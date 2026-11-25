import {shaders} from "./shaders.js";
import {checkAbort} from "../shared/loader.js";
import {startCompletionPump} from "../../localfold/webgpu/runtime/completion-pump.js";

const USAGE_MAP_READ = 0x0001;
const USAGE_COPY_SRC = 0x0004;
const USAGE_COPY_DST = 0x0008;
const USAGE_STORAGE = 0x0080;
const DEFAULT_MAX_POOL_BYTES = 256 * 1024 * 1024;

function fallback(info) {
  return info.isFallbackAdapter===true||/swiftshader|llvmpipe|lavapipe|software|\bwarp\b|basic render/i.test([info.vendor,info.architecture,info.description,info.device].join(" "));
}
export async function acquire({device,allowSoftwareFallback=true,signal}) {
  let adapter;
  if(!device) {
    const gpu=globalThis.navigator?.gpu;
    if(!gpu) throw Error("WebGPU unavailable");
    for(const options of [{powerPreference:"high-performance"},{powerPreference:"high-performance"},{}]) {
      checkAbort(signal);adapter=await gpu.requestAdapter(options);if(adapter) break;
    }
    if(!adapter) throw Error("WebGPU adapter unavailable");
  }
  const raw=adapter?.info??device?.adapterInfo??{};
  const adapterInfo=Object.fromEntries(["vendor","architecture","device","description","isFallbackAdapter"].map(k=>[k,raw[k]]));
  const isFallbackAdapter=fallback(adapterInfo)||adapter?.isFallbackAdapter===true;
  if(!allowSoftwareFallback && (isFallbackAdapter||(!adapterInfo.vendor&&!adapterInfo.description&&!adapterInfo.architecture))) throw Error("Verified hardware WebGPU adapter required");
  const owns=!device;
  if(!device) {
    const requiredFeatures=adapter.features?.has?.("shader-f16")?["shader-f16"]:[];
    const requiredLimits={};
    for(const k of ["maxBufferSize","maxStorageBufferBindingSize","maxComputeWorkgroupsPerDimension"]) {
      if(typeof adapter.limits?.[k]==="number" && Number.isFinite(adapter.limits[k])) requiredLimits[k]=adapter.limits[k];
    }
    try {
      device=await adapter.requestDevice({requiredFeatures,requiredLimits});
    } catch(err) {
      if(!requiredFeatures.length) throw err;
      // Adapter is consumed after the first requestDevice call; re-acquire a fresh adapter before fallback.
      const gpu=globalThis.navigator?.gpu;
      const retryAdapter=await gpu?.requestAdapter({powerPreference:"high-performance"}) ?? await gpu?.requestAdapter();
      if(!retryAdapter) throw err;
      const retryLimits={};
      for(const k of ["maxBufferSize","maxStorageBufferBindingSize","maxComputeWorkgroupsPerDimension"]) {
        if(typeof retryAdapter.limits?.[k]==="number" && Number.isFinite(retryAdapter.limits[k])) retryLimits[k]=retryAdapter.limits[k];
      }
      device=await retryAdapter.requestDevice({requiredLimits:retryLimits});
    }
  }
  const isFirefox=/firefox/i.test(globalThis.navigator?.userAgent??"");
  if(isFirefox && typeof globalThis.GPUBufferUsage!=="undefined") {
    startCompletionPump(device,true);
  }
  return {device,owns,adapterInfo,isFallbackAdapter};
}
export class Resources {
  constructor(connection,{maxPoolBytes=DEFAULT_MAX_POOL_BYTES}={}) {
    Object.assign(this,connection);
    this.supportsPoseBatching=true;
    this.maxPoolBytes=maxPoolBytes;
    this.buffers=new Map();this.residentBytes=0;this.transientBytes=0;this.peakBytes=0;this.pipelines={};
    this.activationPool=new Map();this.pooledBytes=0;
    this.deferredDestroy=[];
    this.paramPool=[];this.startsPool=[];this.atomPool=[];this.stagingPool=new Map();
    this.poolHits=0;this.poolMisses=0;
    this.device.lost?.then(reason=>{this.lost=reason?.message||reason?.reason||"Device lost";});
  }
  alive() {
    if(this.dead||this.lost) throw Error(this.lost??"WebGPU engine destroyed");
  }
  allocate(size,usage,resident=false,data) {
    this.alive();
    size=Math.max(4,Math.ceil(size/4)*4);
    if(!Number.isSafeInteger(size)||size>this.device.limits.maxBufferSize||
      ((usage&USAGE_STORAGE)&&size>this.device.limits.maxStorageBufferBindingSize)) throw Error("GPU allocation exceeds device limits");
    const b=this.device.createBuffer({size,usage,mappedAtCreation:!!data});
    this.buffers.set(b,resident);this[resident?"residentBytes":"transientBytes"]+=size;
    this.peakBytes=Math.max(this.peakBytes,this.residentBytes+this.transientBytes);
    if(data) {
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
      b.unmap();
    }
    return b;
  }
  free(b) {
    if(!this.buffers.has(b)) return;
    this[this.buffers.get(b)?"residentBytes":"transientBytes"]-=b.size;
    this.buffers.delete(b);b.destroy();
  }
  deferFree(b) {
    if(!this.buffers.has(b)) return;
    this.deferredDestroy.push(b);
  }
  flushDeferredDestroy() {
    for(const b of this.deferredDestroy.splice(0)) this.free(b);
  }
  _evictOneActivation() {
    for(const [sz,bucket] of this.activationPool.entries()) {
      if(bucket.length) {
        const b=bucket.pop();
        this.pooledBytes-=b.size;
        if(!bucket.length) this.activationPool.delete(sz);
        // Defer destroy() until after the active GPUOps encoder submits so in-flight commands never hit a destroyed buffer.
        this.deferFree(b);
        return true;
      }
      this.activationPool.delete(sz);
    }
    return false;
  }
  acquireActivation(size) {
    size=Math.max(4,Math.ceil(size/4)*4);
    const bucket=this.activationPool.get(size);
    if(bucket?.length) {
      const b=bucket.pop();
      this.pooledBytes-=b.size;
      if(!bucket.length) this.activationPool.delete(size);
      this.poolHits++;
      return b;
    }
    while(this.pooledBytes+size>this.maxPoolBytes) {
      if(!this._evictOneActivation()) break;
    }
    this.poolMisses++;
    return this.allocate(size,USAGE_STORAGE|USAGE_COPY_SRC|USAGE_COPY_DST,false);
  }
  releaseActivation(b) {
    if(this.dead||this.lost||!this.buffers.has(b)) return;
    if(b.size>this.maxPoolBytes||this.maxPoolBytes<=0) {this.deferFree(b);return;}
    while(this.pooledBytes+b.size>this.maxPoolBytes) {
      if(!this._evictOneActivation()) {this.deferFree(b);return;}
    }
    let bucket=this.activationPool.get(b.size);
    if(!bucket) this.activationPool.set(b.size,bucket=[]);
    bucket.push(b);
    this.pooledBytes+=b.size;
  }
  acquireParamBuffer(byteLength) {
    const idx=this.paramPool.findIndex(b=>b.size>=byteLength);
    if(idx>=0) {
      this.poolHits++;
      return this.paramPool.splice(idx,1)[0];
    }
    this.poolMisses++;
    return this.allocate(byteLength,USAGE_STORAGE|USAGE_COPY_DST,false);
  }
  releaseParamBuffer(b) {
    if(this.dead||this.lost||!this.buffers.has(b)) return;
    this.paramPool.push(b);
  }
  acquireStartsBuffer(starts) {
    const need=Math.max(128,starts.byteLength);
    const idx=this.startsPool.findIndex(b=>b.size>=need);
    let b;
    if(idx>=0) {
      b=this.startsPool.splice(idx,1)[0];
      this.poolHits++;
    } else {
      this.poolMisses++;
      b=this.allocate(need,USAGE_STORAGE|USAGE_COPY_DST,false);
    }
    this.device.queue.writeBuffer(b,0,starts);
    return b;
  }
  releaseStartsBuffer(b) {
    if(this.dead||this.lost||!this.buffers.has(b)) return;
    this.startsPool.push(b);
  }
  acquireAtomBuffer(atoms) {
    const need=atoms.byteLength;
    const idx=this.atomPool.findIndex(b=>b.size>=need);
    let b;
    if(idx>=0) {
      b=this.atomPool.splice(idx,1)[0];
      this.poolHits++;
    } else {
      if(this.atomPool.length>0) this.free(this.atomPool.pop());
      const cap=Math.max(4096,2**Math.ceil(Math.log2(Math.max(4,need))));
      this.poolMisses++;
      b=this.allocate(cap,USAGE_STORAGE|USAGE_COPY_DST,false);
    }
    this.device.queue.writeBuffer(b,0,atoms);
    return b;
  }
  releaseAtomBuffer(b) {
    if(this.dead||this.lost||!this.buffers.has(b)) return;
    this.atomPool.push(b);
  }
  acquireStagingBuffer(size) {
    size=Math.max(4,Math.ceil(size/4)*4);
    const bucket=this.stagingPool.get(size);
    if(bucket?.length) {
      const b=bucket.pop();
      if(!bucket.length) this.stagingPool.delete(size);
      this.poolHits++;
      return b;
    }
    this.poolMisses++;
    return this.allocate(size,USAGE_COPY_DST|USAGE_MAP_READ,false);
  }
  releaseStagingBuffer(b) {
    if(this.dead||this.lost||!this.buffers.has(b)) return;
    let bucket=this.stagingPool.get(b.size);
    if(!bucket) this.stagingPool.set(b.size,bucket=[]);
    bucket.push(b);
  }
  trimPool() {
    this.flushDeferredDestroy();
    for(const bucket of this.activationPool.values()) for(const b of bucket) this.free(b);
    this.activationPool.clear();this.pooledBytes=0;this.deferredDestroy.length=0;
    for(const b of this.paramPool.splice(0)) this.free(b);
    for(const b of this.startsPool.splice(0)) this.free(b);
    for(const b of this.atomPool.splice(0)) this.free(b);
    for(const bucket of this.stagingPool.values()) for(const b of bucket) this.free(b);
    this.stagingPool.clear();
  }
  memoryUsage() {
    const {residentBytes,transientBytes,peakBytes,pooledBytes,poolHits,poolMisses}=this;
    return {residentBytes,transientBytes,peakBytes,pooledBytes,poolHits,poolMisses,
      totalBytes:residentBytes+transientBytes,buffers:this.buffers.size};
  }
  async initialize(tensors,names,signal) {
    const chunks=[], align=n=>Math.ceil(n/4)*4;let size=0;
    function append(bytes) {const offset=size;chunks.push([offset,bytes]);size+=align(bytes.byteLength);return offset/4;}
    this.layout={};
    for(const [key,t] of Object.entries(tensors)) {
      if(!names.includes(key.split("/")[0])) continue;
      const offset=append(t.bytes), scale=t.scale?append(new Uint8Array(t.scale.buffer)):0;
      this.layout[key]={shape:t.shape,offset,scale,kind:{F32:0,F16:1,I8:2}[t.dtype]};
    }
    const bytes=new Uint8Array(size);
    for(const [offset,chunk] of chunks) bytes.set(chunk,offset);
    this.packed=this.allocate(size,USAGE_STORAGE,true,bytes);
    await Promise.all(Object.entries(shaders).map(async ([label,code])=>{
      const module=this.device.createShaderModule({label,code});
      this.pipelines[label]=await this.device.createComputePipelineAsync({label,layout:"auto",compute:{module,entryPoint:"main"}});
    }));
    checkAbort(signal);this.alive();
  }
  destroy() {
    if(this.dead) return;
    this.dead=true;
    this.activationPool.clear();this.pooledBytes=0;this.deferredDestroy.length=0;
    this.paramPool.length=0;this.startsPool.length=0;this.atomPool.length=0;this.stagingPool.clear();
    for(const b of [...this.buffers.keys()]) this.free(b);
    this.pipelines={};if(this.owns) this.device.destroy();
  }
}
