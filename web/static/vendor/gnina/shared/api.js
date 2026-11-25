import {dockPose} from "./docking.js";
import {checkAbort} from "./loader.js";
import {forward,forwardGraph,decodeScores} from "./network.js";
import {prepareGrid,vinaBreakdown,transformPose,centroid} from "./scoring.js";
import {parseReceptor,parseLigands} from "./structure.js";

const POSE_BATCH_SIZE=4;

export function createScorer(backend,models,creationSignal) {
  let closed=false;
  function check(signal) {
    if(closed) throw Error("GNINA engine has been destroyed");
    checkAbort(creationSignal);checkAbort(signal);
  }
  async function scorePairs(pairs,options={}) {
    const {signal,trace,center,contributions=false}=options;
    check(signal);
    if(!pairs.length) return [];
    if(trace || !backend.supportsPoseBatching || pairs.length===1) {
      const out=[];
      for(const {rec,lig} of pairs) out.push(await scorePose(rec,lig,options));
      return out;
    }
    const results=[];
    for(let start=0;start<pairs.length;start+=POSE_BATCH_SIZE) {
      const chunk=pairs.slice(start,start+POSE_BATCH_SIZE);
      const ops=backend.begin(()=>check(signal));
      try {
        if(!ops.readMany) {
          for(const {rec,lig} of chunk) {
            const grid=ops.grid(prepareGrid(rec,lig,center));
            const perModel={};let score=0,affinity=0;
            for(const name of models) {
              check(signal);
              const r=await forward(ops,grid,name,null);
              perModel[name]=r;score+=r.cnnScore;affinity+=r.cnnAffinity;
            }
            ops.release(grid);
            const cnnScore=score/models.length,cnnAffinity=affinity/models.length;
            const breakdown=vinaBreakdown(rec,lig);
            results.push({cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity,vinaScore:breakdown.vinaScore,
              models:perModel,...(contributions?{contributions:breakdown}:{})});
          }
          continue;
        }
        const scoreTensors=[];
        for(const {rec,lig} of chunk) {
          check(signal);
          const grid=ops.grid(prepareGrid(rec,lig,center));
          for(const name of models) {
            check(signal);
            scoreTensors.push(await forwardGraph(ops,grid,name,null));
          }
          ops.release(grid);
        }
        const gpuPromise=ops.readMany(scoreTensors);
        const breakdowns=chunk.map(({rec,lig})=>vinaBreakdown(rec,lig));
        const rawList=await gpuPromise;
        for(const t of scoreTensors) ops.release(t);
        check(signal);
        for(let pIdx=0;pIdx<chunk.length;pIdx++) {
          const perModel={};let score=0,affinity=0;
          for(let mIdx=0;mIdx<models.length;mIdx++) {
            const r=decodeScores(rawList[pIdx*models.length+mIdx]);
            perModel[models[mIdx]]=r;score+=r.cnnScore;affinity+=r.cnnAffinity;
          }
          const cnnScore=score/models.length,cnnAffinity=affinity/models.length;
          const breakdown=breakdowns[pIdx];
          results.push({cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity,vinaScore:breakdown.vinaScore,
            models:perModel,...(contributions?{contributions:breakdown}:{})});
        }
      } finally {ops.finish();}
    }
    return results;
  }
  async function scorePose(receptor,ligand,options={}) {
    const {signal,trace,center,contributions=false}=options;
    check(signal);
    if(typeof receptor==="string") receptor=parseReceptor(receptor,options.receptorFormat);
    if(typeof ligand==="string") {
      const poses=parseLigands(ligand,options.ligandFormat);
      if(poses.length!==1) throw Error("Use scorePoses for multiple poses");
      ligand=poses[0];
    }
    const input=prepareGrid(receptor,ligand,center), ops=backend.begin(()=>check(signal));
    try {
      const grid=ops.grid(input);
      if(trace || !ops.readMany) {
        if(trace) await trace("grid",await ops.read(grid));
        const perModel={};
        let score=0,affinity=0;
        for(const name of models) {
          check(signal);
          const result=await forward(ops,grid,name,trace);
          perModel[name]=result;score+=result.cnnScore;affinity+=result.cnnAffinity;
        }
        ops.release(grid);
        check(signal);
        const cnnScore=score/models.length, cnnAffinity=affinity/models.length;
        const breakdown=vinaBreakdown(receptor,ligand);
        return {cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity,vinaScore:breakdown.vinaScore,
          models:perModel,...(contributions?{contributions:breakdown}:{})};
      }
      const scoreTensors=[];
      for(const name of models) {
        check(signal);
        scoreTensors.push(await forwardGraph(ops,grid,name,null));
      }
      ops.release(grid);
      // Submit single batched GPU pass across all ensemble models and overlap CPU Vina pairwise evaluation.
      const gpuPromise=ops.readMany(scoreTensors);
      const breakdown=vinaBreakdown(receptor,ligand);
      const rawList=await gpuPromise;
      for(const t of scoreTensors) ops.release(t);
      check(signal);
      const perModel={};
      let score=0,affinity=0;
      for(let i=0;i<models.length;i++) {
        const result=decodeScores(rawList[i]);
        perModel[models[i]]=result;score+=result.cnnScore;affinity+=result.cnnAffinity;
      }
      const cnnScore=score/models.length, cnnAffinity=affinity/models.length;
      return {cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity,vinaScore:breakdown.vinaScore,
        models:perModel,...(contributions?{contributions:breakdown}:{})};
    } finally {ops.finish();}
  }
  // Measure a true CNN ablation delta at a fixed grid center. Hiding the only
  // heavy atom leaves an empty channel group, as in the upstream density grid.
  async function maskContributions(rec,lig,{atomIndices=[],residues=[],...options}={}) {
    if(typeof rec==="string") rec=parseReceptor(rec,options.receptorFormat);
    if(typeof lig==="string") [lig]=parseLigands(lig,options.ligandFormat);
    const fixed={...options,center:options.center??centroid(lig)};
    const baseline=await scorePose(rec,lig,fixed);
    const deltas=result=>Object.fromEntries(["cnnScore","cnnAffinity","cnnVS"].map(k=>[k,baseline[k]-result[k]]));
    const uniqueAtoms=[...new Set(atomIndices)];
    for(const index of uniqueAtoms) {
      if(!lig.atoms.some(a=>a.index===index)) throw Error("Unknown ligand atom index "+index);
    }
    const uniqueResidues=[...new Set(residues)];
    for(const residue of uniqueResidues) {
      if(!rec.atoms.some(a=>a.residue===residue)) throw Error("Unknown receptor residue "+residue);
    }
    const atomPairs=uniqueAtoms.map(index=>({rec,lig:{...lig,atoms:lig.atoms.filter(a=>a.index!==index)}}));
    const resPairs=uniqueResidues.map(residue=>({rec:{...rec,atoms:rec.atoms.filter(a=>a.residue!==residue)},lig}));
    const allScored=await scorePairs([...atomPairs,...resPairs],fixed);
    const atoms=uniqueAtoms.map((index,i)=>({index,...deltas(allScored[i])}));
    const perResidue=uniqueResidues.map((residue,i)=>({residue,...deltas(allScored[uniqueAtoms.length+i])}));
    return {baseline,atoms,residues:perResidue};
  }
  async function scorePoses(rec,poses,options={}) {
    if(typeof rec==="string") rec=parseReceptor(rec,options.receptorFormat);
    if(typeof poses==="string") poses=parseLigands(poses,options.ligandFormat);
    return scorePairs(poses.map(lig=>({rec,lig})),options);
  }
  async function minimizePose(rec,lig,options={}) {
    const {iterations=4,translationStep=.5,rotationStep=.12,cnnWeight=1}=options;
    if(!Number.isInteger(iterations)||iterations<0||![translationStep,rotationStep,cnnWeight].every(x=>Number.isFinite(x)&&x>=0)) throw Error("Invalid minimization options");
    if(typeof rec==="string") rec=parseReceptor(rec,options.receptorFormat);
    if(typeof lig==="string") [lig]=parseLigands(lig,options.ligandFormat);
    const energy=s=>s.vinaScore-cnnWeight*Math.log(Math.max(s.cnnScore,1e-30));
    let pose=lig,scores=await scorePose(rec,lig,options),step=1;
    for(let i=0;i<iterations;i++) {
      let best=pose;
      const candidates=[];
      for(let axis=0;axis<6;axis++) for(const sign of [-1,1]) {
        const rotation=[0,0,0],shift=[0,0,0];
        (axis<3?shift:rotation)[axis%3]=sign*step*(axis<3?translationStep:rotationStep);
        candidates.push(transformPose(pose,rotation,shift));
      }
      const candidateScores=await scorePairs(candidates.map(c=>({rec,lig:c})),options);
      for(let k=0;k<candidates.length;k++) {
        const s=candidateScores[k];
        if(energy(s)<energy(scores)) {best=candidates[k];scores=s;}
      }
      if(best===pose) step*=.5;
      pose=best;
    }
    return {pose,scores,objective:energy(scores)};
  }
  return {scorePose,scorePoses,minimizePose,maskContributions,
    dockPose:(rec,lig,options)=>dockPose(rec,lig,options,scorePose,check),ensemble:[...models],adapterInfo:backend.adapterInfo,
    isFallbackAdapter:backend.isFallbackAdapter,memoryUsage:()=>backend.memoryUsage(),
    destroy(){if(!closed){closed=true;backend.destroy();}}};
}
