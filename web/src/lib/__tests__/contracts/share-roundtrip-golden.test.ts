import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY, WORKFLOW_PRESETS } from '$lib/engine/registry';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import {
  decodeHashToGraph,
  encodeGraphToHash,
  graphToYamlRecipe,
  parseRecipeToGraph,
} from '$lib/engine/share';
import type { WorkflowGraph } from '$lib/engine/types';

const IMMUTABLE_SHORT_CODES: Record<string, string> = {
  'pdb-input': 'p',
  'sequence-input': 'q',
  'rfd4-proteina': 'r',
  genie3: 'g',
  protpardelle: 'pp',
  rfdiffusion: 'd',
  rfd3: 'r3',
  boltzgen: 'b',
  'artios-backbone': 'ab',
  evodiff: 'e',
  'la-proteina': 'lp',
  proteina: 'pr',
  'sid-protein': 'sp',
  genie2: 'g2',
  frameflow: 'ff',
  foldflow2: 'f2',
  'web-protein-mpnn': 'm',
  ligandmpnn: 'lm',
  lasermpnn: 'la',
  'artios-potts': 'ap',
  'proton-pottsmpnn': 'pm',
  'antifold-esmif1': 'af',
  'localfold-af2': 'f',
  thermompnn: 'tm',
  esmc: 'ec',
  dyna1: 'dy',
  gnina: 'gn',
  'top-k-filter': 'tk',
  'designability-scorer': 's',
  'structure-compare': 'c',
  pulchra: 'pu',
  cg2all: 'cg',
};

describe('Phase 0 contract: golden share (#p=/#g=) and denovo: v1 YAML roundtrips', () => {
  it('preserves immutable 1-2 char short codes with zero collisions across all 26 adapters', () => {
    const codes = Object.values(IMMUTABLE_SHORT_CODES);
    expect(new Set(codes).size).toBe(codes.length);
    expect(Object.keys(IMMUTABLE_SHORT_CODES).sort()).toEqual(Object.keys(ADAPTER_REGISTRY).sort());
  });

  it('round-trips every WORKFLOW_PRESET through #p=/#g= hash and denovo: v1 YAML', async () => {
    for (const preset of WORKFLOW_PRESETS) {
      const encoded = await encodeGraphToHash(preset.graph);
      expect(encoded.hash.startsWith('#p=') || encoded.hash.startsWith('#g=')).toBe(true);
      const decodedHash = await decodeHashToGraph(encoded.hash);
      expect(decodedHash, `Failed to decode hash for preset ${preset.id}`).not.toBeNull();
      expect(decodedHash!.nodes.map((n) => n.adapterId)).toEqual(
        preset.graph.nodes.map((n) => n.adapterId),
      );
      expect(decodedHash!.edges.length).toBe(preset.graph.edges.length);

      const yaml = graphToYamlRecipe(preset.graph);
      expect(yaml).toContain('denovo: v1');
      const decodedYaml = parseRecipeToGraph(yaml);
      expect(decodedYaml.nodes.map((n) => n.adapterId)).toEqual(
        preset.graph.nodes.map((n) => n.adapterId),
      );
      expect(decodedYaml.edges.length).toBe(preset.graph.edges.length);
    }
  });

  it('round-trips all 26 adapters in a single custom graph via #g= and YAML', async () => {
    const adapterIds = Object.keys(ADAPTER_REGISTRY).sort();
    const graph: WorkflowGraph = {
      id: 'golden-all-26',
      name: 'Golden All 26 Adapters',
      description: 'Phase 0 golden coverage graph',
      nodes: adapterIds.map((adapterId, i) => ({
        id: `n_${i}`,
        adapterId,
        label: ADAPTER_REGISTRY[adapterId].shortName,
        params: defaultParamsForAdapter(adapterId),
      })),
      edges: [],
    };

    const { hash, mode } = await encodeGraphToHash(graph);
    expect(mode).toBe('compressed-dag');
    expect(hash.startsWith('#g=')).toBe(true);

    const decoded = await decodeHashToGraph(hash);
    expect(decoded).not.toBeNull();
    expect(decoded!.nodes.map((n) => n.adapterId)).toEqual(adapterIds);

    const yaml = graphToYamlRecipe(graph);
    const fromYaml = parseRecipeToGraph(yaml);
    expect(fromYaml.nodes.map((n) => n.adapterId)).toEqual(adapterIds);
  });

  it('preserves localfold-af2 foldModel and sequence-input seqLabel YAML aliases', () => {
    const yaml = [
      'denovo: v1',
      'name: "Alias Golden"',
      'nodes:',
      '  seq1: {"model":"sequence-input","label":"My Seq","seqLabel":"wt_dimer","sequence":"ACDEFGHIKLMNPQRSTVWY"}',
      '  fold1: {"model":"localfold-af2","label":"AF3","foldModel":"af3","msaMode":"single"}',
      'wires:',
      '  - seq1.sequences -> fold1.sequences',
    ].join('\n');

    const parsed = parseRecipeToGraph(yaml);
    const seqNode = parsed.nodes.find((n) => n.id === 'seq1');
    const foldNode = parsed.nodes.find((n) => n.id === 'fold1');
    expect(seqNode?.params.label).toBe('wt_dimer');
    expect(foldNode?.params.model).toBe('af3');
  });
});
