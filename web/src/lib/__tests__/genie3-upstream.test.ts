import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';

const VENDOR_DIR = resolve(process.cwd(), 'static/vendor/genie3');

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

  it('has vendored Genie 3 WebGPU entrypoint and <=25 MiB FP16 shards', async () => {
    const entryPath = resolve(VENDOR_DIR, 'webgpu/index.js');
    const manifestPath = resolve(VENDOR_DIR, 'weights/fp16/manifest.json');
    expect(existsSync(entryPath)).toBe(true);
    expect(existsSync(manifestPath)).toBe(true);

    const mod = await import(pathToFileURL(entryPath).href);
    expect(typeof mod.requestWebGpuDevice).toBe('function');
    expect(typeof mod.fetchGenie3Weights).toBe('function');
    expect(typeof mod.Genie3GeneratorGpu?.create).toBe('function');

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    expect(manifest.format).toBe('genie3-safetensors-v1');
    expect(manifest.precision).toBe('fp16');
    expect(manifest.shards.length).toBeGreaterThanOrEqual(2);
    for (const s of manifest.shards) {
      const sp = resolve(VENDOR_DIR, 'weights/fp16', s.file);
      expect(existsSync(sp)).toBe(true);
      // Cloudflare Pages single-file ceiling is 25 MiB
      expect(statSync(sp).size).toBeLessThan(25 * 1024 * 1024);
    }
  });
});
