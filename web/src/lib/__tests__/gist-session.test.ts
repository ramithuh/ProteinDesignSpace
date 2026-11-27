import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it, vi } from 'vitest';
import {
  buildGistShareUrl,
  deserializeGistSession,
  deserializeGistSessionAsync,
  extractGistQueryParam,
  formatGistShareParam,
  GIST_SESSION_FILENAME,
  GIST_SESSION_SCHEMA,
  loadGistSession,
  parseGistReference,
  serializeGistSession,
  serializeGistSessionCompressed,
} from '$lib/engine/gist-session';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { generateSampleHelixBundlePdb, parsePdbBackbone } from '$lib/protein-ops/pdb';
import type {
  CanonicalArtifact,
  DesignabilityMetrics,
  FoldedStructure,
  MutationMatrixArtifact,
  ResidueTrackArtifact,
  SequenceSet,
  StructureArtifact,
  TrajectoryFramesArtifact,
  WorkflowGraph,
} from '$lib/engine/types';

const VALID_ID = '0123456789abcdef0123456789abcdef';
const VALID_REV = 'abcdef1234567890abcdef1234567890abcdef12';

function makeStructureArtifact(id: string, pdbText: string, overrides: Partial<StructureArtifact> = {}): StructureArtifact {
  const parsed = parsePdbBackbone(pdbText);
  const geometry = computeGeometryStats(
    parsed.caCoords,
    parsed.chainIds,
    parsed.resNums,
    parsed.insCodes,
  );
  return {
    kind: 'StructureArtifact',
    id,
    label: `${id}.pdb`,
    sourceModelId: 'genie3',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    chainIdMap: parsed.chainIdMap,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    provenance: {
      quantization: 'fp16',
      diffusionStepsScheduled: 50,
      diffusionStepsRun: 50,
      warnings: ['test provenance note'],
    },
    molecularContext: {
      ligands: ['ATP'],
      chainTypes: { A: 'protein' },
    },
    geometry,
    ...overrides,
  };
}

describe('gist-session: parseGistReference / URL formatting', () => {
  it('parses all valid GitHub Gist URLs and shorthand forms', () => {
    expect(parseGistReference(`https://gist.github.com/octocat/${VALID_ID}`)).toEqual({
      owner: 'octocat',
      id: VALID_ID,
    });
    expect(parseGistReference(`https://gist.github.com/octocat/${VALID_ID}/${VALID_REV}`)).toEqual({
      owner: 'octocat',
      id: VALID_ID,
      rev: VALID_REV,
    });
    expect(parseGistReference(`https://gist.github.com/octocat/${VALID_ID}/#file-denovo-session-json`)).toEqual({
      owner: 'octocat',
      id: VALID_ID,
    });
    expect(parseGistReference(`https://gist.github.com/${VALID_ID}`)).toEqual({
      id: VALID_ID,
    });
    expect(
      parseGistReference(`https://gist.githubusercontent.com/octo-cat/${VALID_ID}/raw/${GIST_SESSION_FILENAME}`),
    ).toEqual({
      owner: 'octo-cat',
      id: VALID_ID,
    });
    expect(
      parseGistReference(`https://gist.githubusercontent.com/octo-cat/${VALID_ID}/raw/${VALID_REV}/${GIST_SESSION_FILENAME}`),
    ).toEqual({
      owner: 'octo-cat',
      id: VALID_ID,
      rev: VALID_REV,
    });
    expect(
      parseGistReference(`https://gist.githubusercontent.com/octo-cat/${VALID_ID}/raw`),
    ).toEqual({
      owner: 'octo-cat',
      id: VALID_ID,
    });
    expect(parseGistReference(`octocat/${VALID_ID.toUpperCase()}`)).toEqual({
      owner: 'octocat',
      id: VALID_ID,
    });
    expect(parseGistReference(`octocat/${VALID_ID}@${VALID_REV.slice(0, 12)}`)).toEqual({
      owner: 'octocat',
      id: VALID_ID,
      rev: VALID_REV.slice(0, 12),
    });
    expect(parseGistReference(VALID_ID)).toEqual({ id: VALID_ID });
  });

  it('formats ?gist= parameters and shareable URLs', () => {
    expect(formatGistShareParam({ owner: 'alice', id: VALID_ID, rev: VALID_REV })).toBe(
      `alice/${VALID_ID}@${VALID_REV}`,
    );
    expect(formatGistShareParam({ owner: 'alice', id: VALID_ID })).toBe(`alice/${VALID_ID}`);
    expect(formatGistShareParam({ id: VALID_ID })).toBe(VALID_ID);
    expect(
      buildGistShareUrl('https://denovo.run/', { owner: 'alice', id: VALID_ID }),
    ).toBe(`https://denovo.run/?gist=alice/${VALID_ID}`);
    expect(
      extractGistQueryParam(`https://denovo.run/?gist=alice/${VALID_ID}`),
    ).toBe(`alice/${VALID_ID}`);
    expect(extractGistQueryParam(`?gist=${VALID_ID}`)).toBe(VALID_ID);
  });

  it('rejects non-GitHub URLs, invalid IDs/owners/revs, path traversal, and XSS strings', () => {
    const invalidInputs = [
      '',
      '   ',
      `http://gist.github.com/octocat/${VALID_ID}`,
      `https://github.com/octocat/${VALID_ID}`,
      `https://evil.com/octocat/${VALID_ID}`,
      `https://gist.github.com.evil.com/octocat/${VALID_ID}`,
      `https://user:pass@gist.github.com/octocat/${VALID_ID}`,
      `https://gist.github.com:8443/octocat/${VALID_ID}`,
      `https://gist.github.com/octocat/${VALID_ID}?foo=bar`,
      `https://gist.github.com/octocat/${VALID_ID}/extra/seg`,
      `https://gist.github.com/octocat/${VALID_ID}/revisions`,
      `https://gist.githubusercontent.com/octocat/${VALID_ID}/raw/../../etc/passwd`,
      `https://gist.githubusercontent.com/octocat/${VALID_ID}/notraw/file.json`,
      `https://evil.com/?gist=${VALID_ID}`,
      'javascript:alert(1)',
      '<script>alert(1)</script>',
      `octocat/${VALID_ID}<script>`,
      `-leadingdash/${VALID_ID}`,
      `${'a'.repeat(40)}/${VALID_ID}`,
      'octocat/too_short_id',
      'octocat/zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz',
      `octocat/${VALID_ID}@short`,
      `octocat/${VALID_ID}@nothex123`,
      `${VALID_ID}@${VALID_REV}`,
      'not-a-gist-id',
    ];
    for (const bad of invalidInputs) {
      expect(parseGistReference(bad), `expected null for "${bad}"`).toBeNull();
    }
  });
});

describe('gist-session: serializeGistSession & deserializeGistSession', () => {
  it('round-trips workflow + artifacts, strips & re-derives coordinates, skips TrajectoryFrames with warning', () => {
    const pdb = generateSampleHelixBundlePdb();
    const struct1 = makeStructureArtifact('gen-design-1', pdb);
    const struct2 = makeStructureArtifact('gen-design-2', pdb);
    const structWithCollection: StructureArtifact = {
      ...struct1,
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-1',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items: [struct1, struct2],
      },
    };

    const traj: TrajectoryFramesArtifact = {
      kind: 'TrajectoryFrames',
      id: 'traj-1',
      parentStructureId: struct1.id,
      frameCount: 50,
      ordering: 'ordered-frames',
      pdbText: pdb,
    };

    const seqs: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seqset-1',
      parentStructureId: struct1.id,
      sourceStructureId: struct1.id,
      sourceModelId: 'web-protein-mpnn',
      candidates: [
        {
          id: 'seq-1',
          sequence: struct1.sequence,
          score: -1.23,
          globalScore: -1.18,
          sourceModelId: 'web-protein-mpnn',
          sourceStructureId: struct1.id,
          sampleIndex: 0,
        },
      ],
    };

    const plddt = new Float32Array(struct1.length).fill(88.5);
    const pae = new Float32Array(struct1.length * struct1.length).fill(2.5);
    const folded: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'folds-1',
      parentStructureId: struct1.id,
      sourceStructureId: struct1.id,
      folds: [
        {
          sequenceId: 'seq-1',
          sequence: struct1.sequence,
          foldModelId: 'localfold-af2-monomer',
          sourceStructureId: struct1.id,
          pdbText: struct1.pdbText,
          caCoords: new Float32Array(struct1.caCoords),
          plddt,
          pae,
          meanPlddt: 88.5,
          ptm: 0.87,
          elapsedMs: 1200,
        },
      ],
    };

    const metrics: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'metrics-1',
      passRate: 1.0,
      rows: [
        {
          structureId: struct1.id,
          sequenceId: 'seq-1',
          sequence: struct1.sequence,
          scRmsd: 0.72,
          scTm: 0.94,
          meanPlddt: 88.5,
          ptm: 0.87,
          designable: true,
          alignedFoldPdb: struct1.pdbText,
          designedPdb: struct1.pdbText,
          plddt,
          pae,
          geometry: struct1.geometry,
          foldElapsedMs: 1200,
        },
      ],
      best: null,
    };
    metrics.best = metrics.rows[0];

    const mutMat: MutationMatrixArtifact = {
      kind: 'MutationMatrix',
      id: 'mut-1',
      sourceModelId: 'thermompnn',
      parentStructureId: struct1.id,
      metric: 'ddg-kcal-mol',
      direction: 'lower-is-better',
      residues: struct1.sequence,
      residueIds: Array.from({ length: struct1.length }, (_, i) => `A${i + 1}`),
      alphabet: 'ACDEFGHIKLMNPQRSTVWY',
      values: new Float32Array(struct1.length * 20).fill(-0.4),
      meanPerPosition: new Float32Array(struct1.length).fill(-0.4),
      targetedMutations: [
        { mutation: 'A1G', chainId: 'A', residueId: 'A1', wildType: 'A', mutant: 'G', score: 0.35 },
      ],
    };

    const track: ResidueTrackArtifact = {
      kind: 'ResidueTrack',
      id: 'track-1',
      sourceModelId: 'dyna1',
      parentStructureId: struct1.id,
      quantity: 'exchange-prob',
      residues: struct1.sequence,
      values: new Float32Array(struct1.length).fill(0.12),
      rawLogits: new Float32Array(struct1.length).fill(-1.9),
      summary: { mean: 0.12, max: 0.12, min: 0.12 },
    };

    const graph: WorkflowGraph = {
      id: 'gist-test-graph',
      name: 'Gist Test Workflow',
      nodes: [
        { id: 'gen', adapterId: 'genie3', label: 'Genie 3', params: defaultParamsForAdapter('genie3') },
        { id: 'mpnn', adapterId: 'web-protein-mpnn', label: 'MPNN', params: defaultParamsForAdapter('web-protein-mpnn') },
        { id: 'fold', adapterId: 'localfold-af2', label: 'AF2', params: defaultParamsForAdapter('localfold-af2') },
        { id: 'score', adapterId: 'designability-scorer', label: 'Scorer', params: defaultParamsForAdapter('designability-scorer') },
        { id: 'thermo', adapterId: 'thermompnn', label: 'ThermoMPNN', params: defaultParamsForAdapter('thermompnn') },
        { id: 'dyna', adapterId: 'dyna1', label: 'Dyna-1', params: defaultParamsForAdapter('dyna1') },
      ],
      edges: [
        { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
        { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
        { fromNode: 'gen', fromPort: 'structure', toNode: 'thermo', toPort: 'structure' },
        { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'dyna', toPort: 'sequences' },
      ],
    };

    const nodeOutputs: Record<string, Record<string, CanonicalArtifact>> = {
      gen: { structure: structWithCollection, trajectory: traj },
      mpnn: { sequences: seqs },
      fold: { folds: folded, structure: struct1 },
      score: { metrics },
      thermo: { mutations: mutMat },
      dyna: { track },
    };

    const serialized = serializeGistSession(graph, nodeOutputs);
    expect(serialized.nodeCount).toBe(6);
    expect(serialized.resultCount).toBe(6);
    expect(serialized.bytes).toBeGreaterThan(0);
    expect(serialized.warnings).toHaveLength(1);
    expect(serialized.warnings[0]).toMatch(/TrajectoryFrames|trajectory/i);

    // Stripped keys must NOT appear in serialized JSON
    expect(serialized.json).not.toContain('"caCoords"');
    expect(serialized.json).not.toContain('"backbone4Coords"');
    expect(serialized.json).not.toContain('"traj-1"');

    const restored = deserializeGistSession(serialized.json);
    expect(restored.schema).toBe(GIST_SESSION_SCHEMA);
    expect(restored.graph.nodes).toHaveLength(6);
    expect(restored.outputs.gen.trajectory).toBeUndefined();

    // Re-derived coordinates match originals on StructureArtifact + collection items + FoldedStructure
    const rStruct = restored.outputs.gen.structure as StructureArtifact;
    expect(rStruct.caCoords).toBeInstanceOf(Float32Array);
    expect(rStruct.backbone4Coords).toBeInstanceOf(Float32Array);
    expect(rStruct.caCoords.length).toBe(struct1.caCoords.length);
    expect(rStruct.backbone4Coords.length).toBe(struct1.backbone4Coords.length);
    expect(Array.from(rStruct.caCoords)).toEqual(Array.from(struct1.caCoords));
    expect(rStruct.geometry).toEqual(struct1.geometry);
    expect(rStruct.diagnostics).toEqual(struct1.diagnostics);
    expect(rStruct.provenance).toEqual(struct1.provenance);
    expect(rStruct.molecularContext).toEqual(struct1.molecularContext);
    expect(rStruct.observedBackboneMask).toBeInstanceOf(Uint8Array);

    expect(rStruct.collection?.items).toHaveLength(2);
    expect(rStruct.collection?.items[1].caCoords).toBeInstanceOf(Float32Array);
    expect(Array.from(rStruct.collection!.items[1].caCoords)).toEqual(Array.from(struct2.caCoords));

    const rFolds = restored.outputs.fold.folds as FoldedStructure;
    expect(rFolds.folds[0].caCoords).toBeInstanceOf(Float32Array);
    expect(Array.from(rFolds.folds[0].caCoords)).toEqual(Array.from(struct1.caCoords));
    expect(rFolds.folds[0].plddt).toBeInstanceOf(Float32Array);
    expect(rFolds.folds[0].plddt[0]).toBeCloseTo(88.5, 4);
    expect(rFolds.folds[0].pae).toBeInstanceOf(Float32Array);

    const rMetrics = restored.outputs.score.metrics as DesignabilityMetrics;
    expect(rMetrics.best?.scRmsd).toBeCloseTo(0.72, 4);
    expect(rMetrics.rows[0].plddt).toBeInstanceOf(Float32Array);

    const rMut = restored.outputs.thermo.mutations as MutationMatrixArtifact;
    expect(rMut.values).toBeInstanceOf(Float32Array);
    expect(rMut.meanPerPosition).toBeInstanceOf(Float32Array);
    expect(rMut.targetedMutations).toHaveLength(1);

    const rTrack = restored.outputs.dyna.track as ResidueTrackArtifact;
    expect(rTrack.values).toBeInstanceOf(Float32Array);
    expect(rTrack.rawLogits).toBeInstanceOf(Float32Array);
    expect(rTrack.summary?.mean).toBeCloseTo(0.12, 4);
  });

  it('rejects prototype pollution, wrong schema, invalid JSON, and malformed graphs', () => {
    const validGraph: WorkflowGraph = {
      id: 'g1',
      name: 'G1',
      nodes: [{ id: 'g', adapterId: 'genie3', params: defaultParamsForAdapter('genie3') }],
      edges: [],
    };
    const good = serializeGistSession(validGraph, {});

    expect(() => deserializeGistSession('')).toThrow(/empty/i);
    expect(() => deserializeGistSession('{not json')).toThrow(/Invalid JSON/i);
    expect(() =>
      deserializeGistSession(JSON.stringify({ schema: 'wrong/v0', graph: validGraph, outputs: {} })),
    ).toThrow(/Unsupported Gist session schema/i);

    const polluted = `{"schema":"${GIST_SESSION_SCHEMA}","graph":{"id":"g","name":"g","nodes":[],"edges":[]},"outputs":{},"__proto__":{"polluted":true}}`;
    expect(() => deserializeGistSession(polluted)).toThrow(/Forbidden property key/i);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();

    const nestedPolluted = good.json.replace('"outputs":{}', '"outputs":{"g":{"constructor":{"evil":1}}}');
    expect(() => deserializeGistSession(nestedPolluted)).toThrow(/Forbidden property key/i);

    // Malformed graphs rejected via normalizeWorkflowGraph / validateGraph
    expect(() =>
      deserializeGistSession(
        JSON.stringify({
          schema: GIST_SESSION_SCHEMA,
          createdAt: new Date().toISOString(),
          graph: {
            id: 'bad',
            name: 'Bad',
            nodes: [{ id: 'n1', adapterId: 'unknown-model-xyz', params: {} }],
            edges: [],
          },
          outputs: {},
        }),
      ),
    ).toThrow(/Unknown model/i);

    expect(() =>
      deserializeGistSession(
        JSON.stringify({
          schema: GIST_SESSION_SCHEMA,
          createdAt: new Date().toISOString(),
          graph: {
            id: 'cycle',
            name: 'Cycle',
            nodes: [
              { id: 'a', adapterId: 'rfd4-proteina', params: {} },
              { id: 'b', adapterId: 'rfd4-proteina', params: {} },
            ],
            edges: [
              { fromNode: 'a', fromPort: 'structure', toNode: 'b', toPort: 'structure' },
              { fromNode: 'b', fromPort: 'structure', toNode: 'a', toPort: 'structure' },
            ],
          },
          outputs: {},
        }),
      ),
    ).toThrow(/cycle/i);
  });

  it('seedMemoFromOutputs lets ExecutionScheduler skip Gist-hydrated nodes without running workers', async () => {
    const pdb = generateSampleHelixBundlePdb();
    const struct = makeStructureArtifact('gen-1', pdb);
    const seqs: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seq-1',
      parentStructureId: struct.id,
      sourceStructureId: struct.id,
      sourceModelId: 'web-protein-mpnn',
      candidates: [
        {
          id: 'c1',
          sequence: struct.sequence,
          score: -1.0,
          globalScore: -1.0,
          sourceModelId: 'web-protein-mpnn',
        },
      ],
    };
    const graph: WorkflowGraph = {
      id: 'memo-seed',
      name: 'Memo Seed',
      nodes: [
        { id: 'gen', adapterId: 'genie3', params: defaultParamsForAdapter('genie3') },
        {
          id: 'mpnn',
          adapterId: 'web-protein-mpnn',
          params: { ...defaultParamsForAdapter('web-protein-mpnn'), family: 'ca' },
        },
      ],
      edges: [{ fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' }],
    };

    const scheduler = new ExecutionScheduler();
    const genSeqs: SequenceSet = {
      kind: 'SequenceSet',
      id: 'gen-seq-1',
      parentStructureId: struct.id,
      sourceStructureId: struct.id,
      sourceModelId: 'genie3',
      candidates: [
        {
          id: 'g-c1',
          sequence: struct.sequence,
          score: null,
          globalScore: null,
          sourceModelId: 'genie3',
        },
      ],
    };
    const seeded = scheduler.seedMemoFromOutputs(graph, {
      gen: { structure: struct, sequences: genSeqs },
      mpnn: { sequences: seqs },
    });
    expect(seeded).toBe(2);

    const events: string[] = [];
    const res = await scheduler.run(graph, (ev) => {
      events.push(`${ev.nodeId}:${ev.status}`);
    });
    expect(res.outputsByNode.gen.structure.id).toBe(struct.id);
    expect(res.outputsByNode.mpnn.sequences.id).toBe(seqs.id);
    expect(events).toEqual(['gen:cached', 'mpnn:cached']);
  });
});

describe('gist-session: loadGistSession', () => {
  const sampleGraph: WorkflowGraph = {
    id: 'gist-load',
    name: 'Gist Load',
    nodes: [{ id: 'g', adapterId: 'genie3', params: defaultParamsForAdapter('genie3') }],
    edges: [],
  };
  const pdb = generateSampleHelixBundlePdb();
  const sampleSerialized = serializeGistSession(sampleGraph, {
    g: { structure: makeStructureArtifact('g-struct', pdb) },
  });

  it('uses direct gist.githubusercontent.com raw fetch first when owner is known (zero API hit)', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      expect(String(input)).toBe(
        `https://gist.githubusercontent.com/alice/${VALID_ID}/raw/${VALID_REV}/${GIST_SESSION_FILENAME}`,
      );
      return new Response(sampleSerialized.json, { status: 200 });
    });

    const loaded = await loadGistSession(`alice/${VALID_ID}@${VALID_REV}`, {
      fetchFn: fetchMock as unknown as typeof fetch,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(loaded.ref).toEqual({ owner: 'alice', id: VALID_ID, rev: VALID_REV });
    expect(loaded.sourceUrl).toContain('gist.githubusercontent.com');
    expect((loaded.session.outputs.g.structure as StructureArtifact).caCoords).toBeInstanceOf(Float32Array);
  });

  it('falls back to api.github.com when raw fast-path returns 404 or owner is absent', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith('https://gist.githubusercontent.com/')) {
        return new Response('Not Found', { status: 404 });
      }
      expect(url).toBe(`https://api.github.com/gists/${VALID_ID}`);
      return new Response(
        JSON.stringify({
          files: {
            'custom-name.json': {
              filename: 'custom-name.json',
              truncated: false,
              content: sampleSerialized.json,
            },
          },
        }),
        { status: 200 },
      );
    });

    const loaded = await loadGistSession(`alice/${VALID_ID}`, {
      fetchFn: fetchMock as unknown as typeof fetch,
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(loaded.sourceUrl).toBe(`https://api.github.com/gists/${VALID_ID}`);
    expect(loaded.session.graph.id).toBe('gist-load');

    // Owner-less ID skips raw fast-path and goes straight to api.github.com
    fetchMock.mockClear();
    const ownerless = await loadGistSession(VALID_ID, {
      fetchFn: fetchMock as unknown as typeof fetch,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(ownerless.ref).toEqual({ id: VALID_ID });
  });

  it('follows raw_url when truncated is true and enforces host allowlist on raw_url', async () => {
    const allowedRawUrl = `https://gist.githubusercontent.com/alice/${VALID_ID}/raw/${VALID_REV}/denovo-session.json`;
    const fetchTruncated = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `https://api.github.com/gists/${VALID_ID}`) {
        return new Response(
          JSON.stringify({
            files: {
              [GIST_SESSION_FILENAME]: {
                filename: GIST_SESSION_FILENAME,
                truncated: true,
                raw_url: allowedRawUrl,
              },
            },
          }),
          { status: 200 },
        );
      }
      expect(url).toBe(allowedRawUrl);
      return new Response(sampleSerialized.json, { status: 200 });
    });

    const loaded = await loadGistSession(VALID_ID, {
      fetchFn: fetchTruncated as unknown as typeof fetch,
    });
    expect(fetchTruncated).toHaveBeenCalledTimes(2);
    expect(loaded.sourceUrl).toBe(allowedRawUrl);

    // Host allowlist enforcement on malicious raw_url
    for (const evilRaw of [
      'https://evil.com/payload.json',
      'http://gist.githubusercontent.com/alice/id/raw/denovo-session.json',
      'https://user@gist.githubusercontent.com/alice/id/raw/denovo-session.json',
    ]) {
      const evilFetch = vi.fn(async () =>
        new Response(
          JSON.stringify({
            files: {
              [GIST_SESSION_FILENAME]: {
                truncated: true,
                raw_url: evilRaw,
              },
            },
          }),
          { status: 200 },
        ),
      );
      await expect(
        loadGistSession(VALID_ID, { fetchFn: evilFetch as unknown as typeof fetch }),
      ).rejects.toThrow(/Blocked/i);
      expect(evilFetch).toHaveBeenCalledTimes(1);
    }
  });

  it('enforces maxBytes across headers, streaming bodies, and inline API content', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(sampleSerialized.json, {
        status: 200,
        headers: { 'content-length': String(sampleSerialized.bytes) },
      }),
    );

    await expect(
      loadGistSession(`alice/${VALID_ID}`, {
        fetchFn: fetchMock as unknown as typeof fetch,
        maxBytes: 128,
      }),
    ).rejects.toThrow(/exceeds maximum allowed size/i);

    // Streaming overflow without content-length header
    const streamResp = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sampleSerialized.json));
          controller.close();
        },
      }),
      { status: 200 },
    );
    await expect(
      loadGistSession(`alice/${VALID_ID}`, {
        fetchFn: vi.fn(async () => streamResp) as unknown as typeof fetch,
        maxBytes: 64,
      }),
    ).rejects.toThrow(/exceeds maximum allowed size/i);

    // Inline API content exceeding maxBytes (while metadata passes if maxBytes is tight on content)
    const bigContent = sampleSerialized.json;
    const apiFetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      url: `https://api.github.com/gists/${VALID_ID}`,
      headers: new Headers(),
      text: async () =>
        JSON.stringify({
          files: {
            [GIST_SESSION_FILENAME]: {
              truncated: false,
              content: bigContent,
            },
          },
        }),
    }));
    await expect(
      loadGistSession(VALID_ID, {
        fetchFn: apiFetch as unknown as typeof fetch,
        maxBytes: Math.max(100, bigContent.length - 10),
      }),
    ).rejects.toThrow(/exceeds maximum allowed size/i);
  });
});

describe('gist-session: UI & zero-local-storage invariants', () => {
  it('GistSessionPopover.svelte and +page.svelte parse cleanly and persist zero gist data in local storage', () => {
    const gistCoreSrc = readFileSync(resolve(process.cwd(), 'src/lib/engine/gist-session.ts'), 'utf8');
    const popoverSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/GistSessionPopover.svelte'), 'utf8');
    const pageSrc = readFileSync(resolve(process.cwd(), 'src/routes/+page.svelte'), 'utf8');

    expect(parse(popoverSrc, { modern: true }).fragment).toBeDefined();
    expect(parse(pageSrc, { modern: true }).fragment).toBeDefined();

    for (const src of [gistCoreSrc, popoverSrc]) {
      expect(src).not.toMatch(/\b(localStorage|sessionStorage|indexedDB)\b/);
    }
    expect(pageSrc).toContain('GistSessionPopover');
    expect(pageSrc).toContain('applyGistQueryParam');
    expect(pageSrc).toContain('hydrateGistSession');
    expect(pageSrc).toContain('seedMemoFromOutputs');
  });

  it('strips redundant cifText / collection[0] text and round-trips gzip+base64 compressed sessions', async () => {
    const graph: WorkflowGraph = {
      id: 'gz-test',
      name: 'Compressed Test',
      description: '',
      nodes: [
        { id: 'gen_1', adapterId: 'rfd4-proteina', label: 'RFD4', params: {}, position: { x: 0, y: 0 } },
      ],
      edges: [],
    };
    const hugeCif = 'data_BIG\n' + '# '.repeat(4000) + '\n';
    const samplePdb = generateSampleHelixBundlePdb();
    const item1 = makeStructureArtifact('d1', samplePdb, { cifText: hugeCif });
    const item2 = makeStructureArtifact('d2', samplePdb, { cifText: hugeCif });
    const batch: StructureArtifact = {
      ...item1,
      id: 'batch-2',
      collection: {
        kind: 'Collection',
        schema: 'denovo.collection/v1',
        id: 'col-2',
        familyKey: 'fam-2',
        collectionKind: 'independent-candidates',
        requestedCount: 2,
        complete: true,
        items: [item1, item2],
        keys: ['d1', 'd2'],
      },
    };

    const plain = serializeGistSession(graph, { gen_1: { structure: batch } });
    // Redundant cifText was stripped when pdbText is present
    expect(plain.json).not.toContain('data_BIG');

    const compressed = await serializeGistSessionCompressed(graph, { gen_1: { structure: batch } });
    expect(compressed.bytes).toBeLessThan(plain.bytes);
    const outer = JSON.parse(compressed.json) as { schema: string; encoding: string; data: string };
    expect(outer.schema).toBe(GIST_SESSION_SCHEMA);
    expect(outer.encoding).toBe('gzip+base64');

    const restored = await deserializeGistSessionAsync(compressed.json);
    const restoredStruct = restored.outputs.gen_1?.structure as StructureArtifact;
    expect(restoredStruct.pdbText).toContain('ATOM');
    expect(restoredStruct.caCoords).toBeInstanceOf(Float32Array);
    expect(restoredStruct.caCoords.length).toBe(item1.caCoords.length);
    expect(restoredStruct.collection?.items.length).toBe(2);
    expect(restoredStruct.collection?.items[1].caCoords.length).toBe(item2.caCoords.length);

    // Regression: payloads > 512 KB must not deadlock CompressionStream backpressure
    const bigPdb = Array.from({ length: 120 }, () => samplePdb).join('');
    const bigItem = makeStructureArtifact('big1', bigPdb);
    const bigCompressed = await serializeGistSessionCompressed(graph, { gen_1: { structure: bigItem } });
    expect(bigCompressed.bytes).toBeLessThan(bigPdb.length / 4);
    const bigRestored = await deserializeGistSessionAsync(bigCompressed.json);
    expect((bigRestored.outputs.gen_1?.structure as StructureArtifact).pdbText.length).toBe(bigPdb.length);
  });
});
