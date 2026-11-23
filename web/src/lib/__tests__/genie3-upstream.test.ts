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
    // Finding #11 & #23 & #7/#9/#10 & Primitive 2: worker applies unified protein-only HETATM policy,
    // fixes predictSequence=false, and emits shared motif lineage metadata
    expect(workerSrc).toContain('resolveGenie3PolyAlaSequence');
    expect(workerSrc).toContain('evaluateProteinOnlyStructurePolicy');
    expect(workerSrc).toContain('buildMotifLineage');
    expect(workerSrc).toContain('lockedChains');
    expect(workerSrc).toContain('chainRoles');
    expect(workerSrc).toContain('nativeMotifCoords');
    // Wave-2 & #123: honest Cα-only output, directionScale 0.0, fail-fast before ensureLoaded(), drop weightBundle, transferables
    expect(workerSrc).toContain('caOnly: true');
    expect(workerSrc).toContain('weightBundle = null');
    expect(workerSrc).toContain('collectOwnedTransferables');
    const failFastIdx = workerSrc.indexOf('requires a wired structure input');
    const ensureLoadedIdx = workerSrc.indexOf('await ensureLoaded(');
    expect(failFastIdx).toBeGreaterThan(-1);
    expect(failFastIdx).toBeLessThan(ensureLoadedIdx);

    // Verify RFD4 & LocalFold workers also carry Wave-2 VRAM cleanup & fail-fast / transferables
    const rfd4WorkerSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/rfd4.worker.ts'), 'utf8');
    expect(rfd4WorkerSrc).toContain('releaseActivations');
    expect(rfd4WorkerSrc).toContain('collectOwnedTransferables');
    const rfdFailFastIdx = rfd4WorkerSrc.indexOf('requires a wired structure input');
    const rfdEnsureIdx = rfd4WorkerSrc.indexOf('await ensureLoaded(');
    expect(rfdFailFastIdx).toBeGreaterThan(-1);
    expect(rfdFailFastIdx).toBeLessThan(rfdEnsureIdx);

    const lfWorkerSrc = readFileSync(resolve(process.cwd(), 'src/lib/workers/localfold.worker.ts'), 'utf8');
    expect(lfWorkerSrc).toContain('releaseAllWeights');
    expect(lfWorkerSrc).toContain('resolveLocalFoldMemoryBudgetBytes');
    expect(lfWorkerSrc).toContain('collectOwnedTransferables');
  });
});
