import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { generateSampleHelixBundlePdb, parseStructureText } from '$lib/protein-ops/pdb';
import { CATALOG_STAGE_GROUPS } from '$lib/engine/catalog';
import { LOCAL_EXAMPLE_PLANS, compileLocalStarterPlan } from '$lib/engine/workflow-planner';
import { validateGraph } from '$lib/engine/graph-utils';
import {
  formatWebGpuAdapterSummary,
  isSoftwareWebGpuAdapter,
  isWebGpuDeviceLostError,
  requestResilientWebGpuAdapter,
} from '$lib/engine/webgpu-device';

const WEB_ROOT = resolve(process.cwd());

describe('Track A + B regressions (#32–#37, #58, catalog & UX)', () => {
  it('#58: generateSampleHelixBundlePdb has valid peptide C(i)–N(i+1) bond distances across all 53 residues including antiparallel Helix 2', () => {
    const pdb = generateSampleHelixBundlePdb();
    const parsed = parseStructureText(pdb);
    expect(parsed.sequence.length).toBe(53);

    const atomsByRes = new Map<number, Record<string, [number, number, number]>>();
    for (const line of pdb.split('\n')) {
      if (!line.startsWith('ATOM')) continue;
      const atom = line.slice(12, 16).trim();
      const resSeq = Number(line.slice(22, 26).trim());
      const x = Number(line.slice(30, 38).trim());
      const y = Number(line.slice(38, 46).trim());
      const z = Number(line.slice(46, 54).trim());
      const entry = atomsByRes.get(resSeq) ?? {};
      entry[atom] = [x, y, z];
      atomsByRes.set(resSeq, entry);
    }

    let maxCnDist = 0;
    let minCnDist = Infinity;
    for (let r = 1; r < 53; r++) {
      const c = atomsByRes.get(r)?.C;
      const nNext = atomsByRes.get(r + 1)?.N;
      expect(c).toBeDefined();
      expect(nNext).toBeDefined();
      const d = Math.hypot(c![0] - nNext![0], c![1] - nNext![1], c![2] - nNext![2]);
      if (d > maxCnDist) maxCnDist = d;
      if (d < minCnDist) minCnDist = d;
      // Helix 2 (residues 20–34) previously had 3.44 Å broken peptide bonds!
      expect(d).toBeGreaterThan(1.1);
      expect(d).toBeLessThan(1.75);
    }
    expect(minCnDist).toBeGreaterThan(1.1);
    expect(maxCnDist).toBeLessThan(1.75);
  });

  it('catalog exposes protpardelle, esmc, and dyna1 in + Add Block categories', () => {
    const allIds = CATALOG_STAGE_GROUPS.flatMap((c) => c.ids);
    expect(allIds).toContain('protpardelle');
    expect(allIds).toContain('esmc');
    expect(allIds).toContain('dyna1');
  });

  it('#32/#33/#34/#35/#36 source guards: ligand rendering, pLDDT/iPAE, Controls padding, maxSequences warning chip', () => {
    const viewerSrc = readFileSync(
      resolve(WEB_ROOT, 'src/lib/viewer/MolSuperposeViewer.svelte'),
      'utf8',
    );
    expect(viewerSrc).toContain("'non-standard'");
    expect(viewerSrc).toContain('non-polymer');

    const foldWorkerSrc = readFileSync(
      resolve(WEB_ROOT, 'src/lib/plugins/localfold-af2/localfold.worker.ts'),
      'utf8',
    );
    expect(foldWorkerSrc).toContain(".fill('LIG')");
    expect(foldWorkerSrc).toContain('paeDim > cleanSeq.length');

    const canvasSrc = readFileSync(resolve(WEB_ROOT, 'src/lib/ui/WorkflowCanvas.svelte'), 'utf8');
    expect(canvasSrc).toContain(': 0.30;');
    expect(canvasSrc).toContain('left: compact ? 0.12 : 0.14');

    const inspectorSrc = readFileSync(resolve(WEB_ROOT, 'src/lib/ui/NodeInspector.svelte'), 'utf8');
    expect(inspectorSrc).toContain('LocalFoldWidget');
    const localFoldWidgetSrc = readFileSync(
      resolve(WEB_ROOT, 'src/lib/ui/inspector/LocalFoldWidget.svelte'),
      'utf8',
    );
    expect(localFoldWidgetSrc).toContain('Fold all');
  });

  it('#37: resilient WebGPU helper classifies software adapters, detects device-loss errors, and falls back across retries', async () => {
    expect(
      isSoftwareWebGpuAdapter({
        vendor: 'google',
        architecture: '',
        device: 'SwiftShader Device',
        description: '',
        isFallbackAdapter: false,
      }),
    ).toBe(true);
    expect(
      isSoftwareWebGpuAdapter({
        vendor: 'nvidia',
        architecture: 'blackwell',
        device: 'RTX 5090',
        description: 'NVIDIA GeForce RTX 5090',
        isFallbackAdapter: false,
      }),
    ).toBe(false);
    expect(
      formatWebGpuAdapterSummary(
        {
          vendor: 'google',
          architecture: '',
          device: 'SwiftShader',
          description: 'SwiftShader',
          isFallbackAdapter: true,
        },
        false,
      ),
    ).toBe('SwiftShader (software, f32)');

    expect(
      isWebGpuDeviceLostError(new Error('mapAsync: A valid external Instance reference no longer exists')),
    ).toBe(true);
    expect(isWebGpuDeviceLostError(new Error('HTTP 404 Not Found'))).toBe(false);

    let calls = 0;
    const mockAdapter = {
      features: new Set(['shader-f16']),
      limits: { maxBufferSize: 1e9, maxStorageBufferBindingSize: 1e9 },
      info: { vendor: 'nvidia', architecture: 'ada', device: 'RTX', description: 'RTX' },
    } as unknown as GPUAdapter;

    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: vi.fn(async (opts?: { powerPreference?: string }) => {
          calls++;
          if (opts?.powerPreference === 'high-performance') return null;
          return mockAdapter;
        }),
      },
    });
    try {
      const res = await requestResilientWebGpuAdapter({ retries: 2, retryDelayMs: 1 });
      expect(res.adapter).toBe(mockAdapter);
      expect(res.isSoftwareFallback).toBe(false);
      expect(res.hasF16).toBe(true);
      expect(calls).toBe(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('#37: vendored engines allow software fallback in production and workers wire CPU/WASM fallbacks', () => {
    for (const rel of [
      'static/vendor/thermompnn/webgpu/index.js',
      'static/vendor/protpardelle/webgpu/device.js',
      'static/vendor/esmc/webgpu/engine.js',
      'static/vendor/antifold-esmif1/webgpu/index.js',
    ]) {
      const src = readFileSync(resolve(WEB_ROOT, rel), 'utf8');
      expect(src).toMatch(/allowSoftwareFallback\s*=\s*true/);
    }

    const thermoWorker = readFileSync(
      resolve(WEB_ROOT, 'src/lib/plugins/thermompnn/thermompnn.worker.ts'),
      'utf8',
    );
    expect(thermoWorker).toContain('/vendor/thermompnn/cpu/index.js');
    expect(thermoWorker).toContain('createThermoMPNNCPU');

    const ligandWorker = readFileSync(
      resolve(WEB_ROOT, 'src/lib/plugins/ligandmpnn/ligandmpnn.worker.ts'),
      'utf8',
    );
    expect(ligandWorker).toContain('/vendor/ligandmpnn/cpu/index.js');
    expect(ligandWorker).toContain('createLigandMpnnCpuEngine');

    const mpnnWorker = readFileSync(resolve(WEB_ROOT, 'src/lib/plugins/web-protein-mpnn/mpnn.worker.ts'), 'utf8');
    expect(mpnnWorker).toContain("executeRun('wasm')");
    expect(mpnnWorker).toContain('isWebGpuDeviceLostError');
  });

  it('Track B UX: simple fast starter chips, stage-colored selection glow, clickable brand logo, returning empty canvas, and vertical split resizer', () => {
    expect(LOCAL_EXAMPLE_PLANS.map((p) => p.id)).toEqual([
      'add-stability-scan',
      'gnina-1stp-pocket',
      'denovo-60aa',
      'stabilize-1ubq',
    ]);
    const gninaStarter = compileLocalStarterPlan(LOCAL_EXAMPLE_PLANS[1].plan);
    expect(validateGraph(gninaStarter.graph)).toEqual([]);
    expect(gninaStarter.graph.nodes).toHaveLength(2);
    expect(gninaStarter.graph.nodes[1].adapterId).toBe('gnina');
    expect(gninaStarter.graph.nodes[1].params.taskMode).toBe('pocket-scan');

    const denovo = compileLocalStarterPlan(LOCAL_EXAMPLE_PLANS[2].plan);
    expect(validateGraph(denovo.graph)).toEqual([]);
    expect(denovo.graph.nodes).toHaveLength(4);

    const ubq = compileLocalStarterPlan(LOCAL_EXAMPLE_PLANS[3].plan);
    expect(validateGraph(ubq.graph)).toEqual([]);
    expect(ubq.graph.nodes).toHaveLength(2);

    const nodeSrc = readFileSync(resolve(WEB_ROOT, 'src/lib/ui/ProteinNode.svelte'), 'utf8');
    expect(nodeSrc).toContain('border-color: var(--stage-color');
    expect(nodeSrc).toContain('color-mix(in srgb, var(--stage-color');

    const pageSrc = readFileSync(resolve(WEB_ROOT, 'src/routes/+page.svelte'), 'utf8');
    expect(pageSrc).toContain('resetToHomeCanvas');
    expect(pageSrc).toContain("localStorage.getItem('denovo-visited')");
    expect(pageSrc).toContain('verticalSplitPct');
    expect(pageSrc).toContain('split-handle is-vertical');
    expect(pageSrc).toContain('row-resize');
  });
});
