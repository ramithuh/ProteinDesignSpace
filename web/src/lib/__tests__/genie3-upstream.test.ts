import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';

const VENDOR_DIR = resolve(process.cwd(), 'static/vendor/genie3');
const WORKER_PATH = resolve(process.cwd(), 'src/lib/workers/genie3.worker.ts');

describe('Genie 3 WebGPU integration', () => {
  it('registers genie3 adapter and valid genie3-mpnn-af2 preset', () => {
    const manifest = ADAPTER_REGISTRY.genie3;
    expect(manifest).toBeDefined();
    expect(manifest.stage).toBe('generate');
    expect(manifest.outputs.structure?.type).toBe('StructureArtifact');
    expect(manifest.outputs.sequences?.type).toBe('SequenceSet');

    const defaults = defaultParamsForAdapter('genie3');
    expect(defaults.length).toBe(80);
    expect(defaults.nSampleStep).toBe(100);

    const preset = WORKFLOW_PRESETS.find((p) => p.id === 'genie3-mpnn-af2');
    expect(preset).toBeDefined();
    expect(validateGraph(preset!.graph)).toEqual([]);
  });

  it('has vendored Genie 3 WebGPU entrypoint and fetches FP16 weights from HuggingFace by default', async () => {
    const entryPath = resolve(VENDOR_DIR, 'webgpu/index.js');
    expect(existsSync(entryPath)).toBe(true);

    const mod = await import(pathToFileURL(entryPath).href);
    expect(typeof mod.requestWebGpuDevice).toBe('function');
    expect(typeof mod.fetchGenie3Weights).toBe('function');
    expect(typeof mod.Genie3GeneratorGpu?.create).toBe('function');

    // Keep weights off Cloudflare Pages static bundle; fetch from HuggingFace with env override
    expect(existsSync(resolve(VENDOR_DIR, 'weights'))).toBe(false);
    const workerSrc = readFileSync(WORKER_PATH, 'utf8');
    expect(workerSrc).toContain('https://huggingface.co/ramithuh/genie3-webgpu/resolve/main/fp16');
    expect(workerSrc).toContain('VITE_GENIE3_WEIGHTS_URL');
  });
});
