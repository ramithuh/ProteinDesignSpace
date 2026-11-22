import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { defaultParamsForAdapter, validateGraph } from '$lib/engine/graph-utils';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { estimateRunRequirements } from '$lib/engine/run-preflight';
import { extractDyna1Backbone3Coords } from '$lib/engine/dyna1-helpers';
import { resolveGenie3CompiledInput } from '$lib/engine/conditioning';
import { encodeGraphToHash, decodeHashToGraph } from '$lib/engine/share';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import type { WorkflowNode } from '$lib/engine/types';

function makeNode(adapterId: string, id: string, paramsOverride: Record<string, unknown> = {}): WorkflowNode {
  return {
    id,
    adapterId,
    label: ADAPTER_REGISTRY[adapterId]?.shortName ?? adapterId,
    params: { ...defaultParamsForAdapter(adapterId), ...paramsOverride },
  };
}

describe('ported-models-audit fixes', () => {
  it('marks Dyna-1 outputs.structure as optional so sequence-only runs pass output contract validation', () => {
    expect(ADAPTER_REGISTRY.dyna1.outputs.structure.required).toBe(false);
    expect(ADAPTER_REGISTRY.dyna1.outputs.track.required).not.toBe(false);
  });

  it('wires tiedPositions on LigandMPNN registry schema and validates without duplicate omitAminoAcids errors', () => {
    const schemaKeys = (ADAPTER_REGISTRY.ligandmpnn.paramSchema ?? []).map((f) => f.key);
    expect(schemaKeys).toContain('tiedPositions');

    const pdb = makeNode('pdb-input', 'p1');
    const lmp = makeNode('ligandmpnn', 'l1', { omitAminoAcids: 'Z' });
    const errs = validateGraph({
      id: 'g-lmp',
      name: 'LigandMPNN omit check',
      nodes: [pdb, lmp],
      edges: [{ fromNode: 'p1', fromPort: 'structure', toNode: 'l1', toPort: 'structure' }],
    });
    const omitErrs = errs.filter((e) => e.toLowerCase().includes('omit'));
    expect(omitErrs).toHaveLength(1);
  });

  it('validates mode-required inputs in validateGraph() before execution', () => {
    const pp = makeNode('protpardelle', 'pp1', {
      taskMode: 'motif',
      motifSelection: 'A1-10',
    });
    const errs = validateGraph({
      id: 'g-pp-missing',
      name: 'Disconnected motif ProtPardelle',
      nodes: [pp],
      edges: [],
    });
    expect(errs.some((e) => /structure/i.test(e))).toBe(true);
  });

  it('allows multi-char mmCIF whole-chain tokens in fixedPositions for schema-validated inverse folders', () => {
    const pdb = makeNode('pdb-input', 'p1');
    const lmp = makeNode('ligandmpnn', 'l1', { fixedPositions: 'AA, AB' });
    const errs = validateGraph({
      id: 'g-multichar',
      name: 'Multi-char chain lock',
      nodes: [pdb, lmp],
      edges: [{ fromNode: 'p1', fromPort: 'structure', toNode: 'l1', toPort: 'structure' }],
    });
    expect(errs).toEqual([]);
  });

  it('builds gap- and chain-break-aware residueIndex in extractDyna1Backbone3Coords', () => {
    const coords = new Float32Array(4 * 12);
    for (let i = 0; i < 4 * 12; i++) coords[i] = i + 1;
    const extracted = extractDyna1Backbone3Coords({
      length: 4,
      backbone4Coords: coords,
      atomGranularity: 'all-atom',
      sourceResidues: [
        { index: 0, authChainId: 'A', projectedChainId: 'A', resNum: 1, insCode: '', resName: 'ALA' },
        { index: 1, authChainId: 'A', projectedChainId: 'A', resNum: 2, insCode: '', resName: 'GLY' },
        { index: 2, authChainId: 'A', projectedChainId: 'A', resNum: 7, insCode: '', resName: 'VAL' },
        { index: 3, authChainId: 'B', projectedChainId: 'B', resNum: 1, insCode: '', resName: 'LEU' },
      ],
    });
    expect(extracted).not.toBeNull();
    expect(Array.from(extracted!.residueIndex)).toEqual([0, 1, 6, 38]);
  });

  it('profiles RFdiffusion taskMode="symmetry" cleanly without double-multiplying length', () => {
    const rfd = makeNode('rfdiffusion', 'r1', {
      taskMode: 'symmetry',
      symmetry: 'c3',
      length: 120,
    });
    const af2 = makeNode('localfold-af2', 'f1', { model: 'monomer' });
    const mpnn = makeNode('web-protein-mpnn', 'm1');
    const findings = validateBioCapabilities({
      id: 'g-rfd-sym',
      name: 'RFdiffusion C3',
      nodes: [rfd, mpnn, af2],
      edges: [
        { fromNode: 'r1', fromPort: 'structure', toNode: 'm1', toPort: 'structure' },
        { fromNode: 'm1', fromPort: 'sequences', toNode: 'f1', toPort: 'sequences' },
      ],
    });
    expect(findings.some((f) => /multimer|oligomer|complex|chain/i.test(f.message))).toBe(true);
  });

  it('keeps Genie 3 taskMode="unconditional" unconditional even if stale lockChain/motifSpan remain', () => {
    const compiled = resolveGenie3CompiledInput(
      { taskMode: 'unconditional', length: 70, lockChain: 'A', motifSpan: 'A1-10' },
      undefined,
      undefined,
    );
    expect(compiled.mode).toBe('unconditional');
    expect(compiled.length).toBe(70);
  });

  it('tracks nonCommercial flags, precision weights, numDesigns, and Dyna-1 VQ-VAE in run-preflight', () => {
    const ff2 = makeNode('foldflow2', 'ff2', { precision: 'fp16', numDesigns: 3 });
    const preFf2 = estimateRunRequirements({ id: 'g-ff2', name: 'FF2', nodes: [ff2], edges: [] });
    expect(preFf2.hasNonCommercialModel).toBe(true);
    expect(preFf2.totalDownloadMB).toBeGreaterThanOrEqual(1300);
    expect(preFf2.plannedBackbones).toBe(3);

    const pdb = makeNode('pdb-input', 'p1');
    const dyna = makeNode('dyna1', 'd1', { precision: 'int8', inputMode: 'auto' });
    const preDyna = estimateRunRequirements({
      id: 'g-dyna',
      name: 'Dyna struct',
      nodes: [pdb, dyna],
      edges: [{ fromNode: 'p1', fromPort: 'structure', toNode: 'd1', toPort: 'structure' }],
    });
    expect(preDyna.totalDownloadMB).toBe(844 + 88);
  });

  it('round-trips all newly short-coded adapters cleanly via share.ts', async () => {
    const pdb = makeNode('pdb-input', 'p1');
    const lmp = makeNode('ligandmpnn', 'l1');
    const dy = makeNode('dyna1', 'd1');
    const g = {
      id: 'g-share-new',
      name: 'Share new adapters',
      nodes: [pdb, lmp, dy],
      edges: [
        { fromNode: 'p1', fromPort: 'structure', toNode: 'l1', toPort: 'structure' },
        { fromNode: 'l1', fromPort: 'sequences', toNode: 'd1', toPort: 'sequences' },
      ],
    };
    const encoded = await encodeGraphToHash(g);
    const decoded = await decodeHashToGraph(encoded.hash);
    expect(decoded?.nodes.map((n) => n.adapterId)).toEqual(['pdb-input', 'ligandmpnn', 'dyna1']);
  });

  it('restricts AntiFold/ESM-IF1 WebGPU autoregressive sampling to canonical 20 AA tokens and updates model cards', () => {
    const afWebGpu = readFileSync(
      resolve(__dirname, '../../../static/vendor/antifold-esmif1/webgpu/index.js'),
      'utf8',
    );
    expect(afWebGpu).not.toContain('var token=34u;for(var j=0u;j<35u;j++)');
    expect(getModelCardForAdapter('dyna1')?.limitations.join(' ')).toContain('VQ-VAE');
    expect(getModelCardForAdapter('la-proteina')?.limitations.join(' ')).toContain('400');
    expect(getModelCardForAdapter('localfold-af2', { model: 'esmfold2-fast-300m' })?.limitations.join(' ')).toContain(
      '64',
    );
  });
});
