import { describe, expect, it, vi } from 'vitest';
import {
  computeCampaignSummary,
  computeCollectionSnapshotId,
  computeDesignPassSummaries,
  computeStructurePayloadDigest,
  makeDesignKey,
  makeFoldKey,
  makeSequenceKey,
  normalizeToCollection,
  resolveExactParentStructure,
  selectRepresentativeMember,
  sha256Hex,
  validateCollectionInvariants,
  type CollectionMember,
  type DesignabilityCollection,
  type DesignCollection,
  type FoldCollection,
  type FoldEvaluationRecord,
  type ResidueTrackCollection,
  type SequenceCollection,
  type StructureCollection,
} from '$lib/engine/collections';
import {
  deriveOrdinalStageSeed,
  deriveStableItemSeed,
  estimateFanoutBounds,
  planGeneratorDesigns,
  planLocalFoldFanout,
  planMpnnFanout,
  planTopKFilter,
  SEED_DOMAIN_SPAN,
} from '$lib/engine/work-plan';
import {
  ArtifactBlobStore,
  clearCollectionCheckpoints,
  commitCollectionItemAtomic,
  decodePayloadBytes,
  encodePayloadBytes,
  loadCollectionCheckpoint,
  MemoryArtifactBackend,
  PinBudgetExceededError,
  saveCollectionCheckpoint,
  StorageQuotaError,
  type CollectionCheckpointRecord,
} from '$lib/engine/artifact-store';
import {
  buildCollectionBundleEntries,
  exportCollectionCsv,
  exportCollectionFasta,
  exportCollectionMultiModelPdb,
  HeterogeneousTopologyError,
  type CollectionBundleManifest,
} from '$lib/engine/export-collection';
import type {
  DesignabilityMetrics,
  DesignabilityRow,
  FoldedCandidate,
  FoldedStructure,
  ResidueTrackArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

// ============================================================================
// Test Fixtures
// ============================================================================

function makeSyntheticPdb(
  length: number,
  options: { chainId?: string; coordShift?: number; atomsPerRes?: readonly string[] } = {},
): string {
  const chain = options.chainId ?? 'A';
  const shift = options.coordShift ?? 0;
  const atoms = options.atomsPerRes ?? ['N', 'CA', 'C', 'O'];
  const lines: string[] = [];
  let serial = 1;
  for (let i = 0; i < length; i++) {
    const resSeq = i + 1;
    for (let a = 0; a < atoms.length; a++) {
      const atomName = atoms[a].padEnd(3, ' ');
      const x = (i * 3.8 + a * 0.5 + shift).toFixed(3).padStart(8, ' ');
      const y = (a * 1.1 + shift * 0.25).toFixed(3).padStart(8, ' ');
      const z = (Math.sin(i + shift) * 2.0).toFixed(3).padStart(8, ' ');
      lines.push(
        `ATOM  ${String(serial).padStart(5, ' ')}  ${atomName} ALA ${chain}${String(resSeq).padStart(4, ' ')}    ${x}${y}${z}  1.00 85.00           C`,
      );
      serial++;
    }
  }
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

function makeSyntheticStructure(
  id: string,
  length: number,
  options: {
    coordShift?: number;
    sequence?: string;
    cifText?: string;
    lockedChains?: string[];
    fixedResidueIndices?: number[];
    motifIndices?: number[];
    chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
    atomsPerRes?: readonly string[];
  } = {},
): StructureArtifact {
  const shift = options.coordShift ?? 0;
  const caCoords = new Float32Array(length * 3);
  const backbone4Coords = new Float32Array(length * 4 * 3);
  for (let i = 0; i < length; i++) {
    caCoords[i * 3] = i * 3.8 + shift;
    caCoords[i * 3 + 1] = shift * 0.5;
    caCoords[i * 3 + 2] = Math.cos(i + shift);
    for (let a = 0; a < 4; a++) {
      backbone4Coords[(i * 4 + a) * 3] = i * 3.8 + a * 0.4 + shift;
      backbone4Coords[(i * 4 + a) * 3 + 1] = a * 0.6;
      backbone4Coords[(i * 4 + a) * 3 + 2] = 0;
    }
  }
  const sequence = options.sequence ?? 'A'.repeat(length);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'rfd4-proteina',
    length,
    sequence,
    caCoords,
    backbone4Coords,
    pdbText: makeSyntheticPdb(length, { coordShift: shift, atomsPerRes: options.atomsPerRes }),
    ...(options.cifText ? { cifText: options.cifText } : {}),
    ...(options.lockedChains ? { lockedChains: options.lockedChains } : {}),
    ...(options.fixedResidueIndices ? { fixedResidueIndices: options.fixedResidueIndices } : {}),
    ...(options.motifIndices ? { motifIndices: options.motifIndices } : {}),
    ...(options.chainRoles ? { chainRoles: options.chainRoles } : {}),
    geometry: {
      length,
      rg: 12.4,
      meanCaDist: 3.8,
      chainBreaks: 0,
      clashes: 0,
      helixPct: 50,
      sheetPct: 20,
      coilPct: 30,
    },
  };
}

function buildStructureCollection(
  familyKey: string,
  structures: StructureArtifact[],
): StructureCollection {
  const members: CollectionMember<StructureArtifact>[] = structures.map((s, idx) => {
    const designKey = makeDesignKey(familyKey, idx);
    const payloadDigest = computeStructurePayloadDigest(s);
    return {
      key: designKey,
      ordinal: idx,
      memberKind: 'StructureArtifact',
      lineage: {
        designKey,
        designIndex: idx,
        sourceStructure: { artifactId: s.id, payloadDigest, designKey },
        parentKeys: [],
      },
      provenance: {
        seedScheme: 'domain-v2',
        seed: deriveStableItemSeed(42, 'generation', designKey),
      },
      payload: s,
      summary: {
        label: s.label,
        residueCount: s.length,
        sequenceDigest: sha256Hex(s.sequence),
      },
    };
  });
  const outcomes = members.map((m) => ({
    key: m.key,
    status: 'completed' as const,
    attempt: 1,
    lineage: m.lineage,
  }));
  return {
    kind: 'Collection',
    schema: 'denovo.collection/v1',
    id: computeCollectionSnapshotId(familyKey, members, outcomes),
    familyKey,
    revision: 1,
    memberKind: 'StructureArtifact',
    collectionKind: 'independent-candidates',
    label: 'Generated Backbones',
    sourceModelId: 'rfd4-proteina',
    members,
    outcomes,
    requestedCount: members.length,
    complete: true,
  };
}

// ============================================================================
// RFC Gate #2: Extension Stability (4 -> 8 designs)
// ============================================================================

describe('Wave-6A Gate #2: Extension stability (4 -> 8)', () => {
  it('keeps identical familyKey, designKeys, and seeds for items 0..3 when extending numDesigns 4 -> 8', () => {
    const baseParams = {
      modelId: 'rfd4-proteina',
      baseSeed: 1337,
      scientificParams: { contig: '60-80', steps: 60, temperature: 0.45 },
      inputDigest: sha256Hex('cond-v1'),
    };

    const plan4 = planGeneratorDesigns({ ...baseParams, numDesigns: 4 });
    const cachedSet = new Set(plan4.items.map((i) => i.key));
    const plan8 = planGeneratorDesigns({ ...baseParams, numDesigns: 8, cachedKeys: cachedSet });

    expect(plan8.familyKey).toBe(plan4.familyKey);
    expect(plan8.items).toHaveLength(8);
    expect(plan8.cachedCount).toBe(4);
    expect(plan8.newCount).toBe(4);

    // Designs 0..3 must preserve exact keys, ordinals, and seeds
    for (let i = 0; i < 4; i++) {
      expect(plan8.items[i].key).toBe(plan4.items[i].key);
      expect(plan8.items[i].seed).toBe(plan4.items[i].seed);
      expect(plan8.items[i].ordinal).toBe(i);
      expect(plan8.items[i].cached).toBe(true);
    }

    // Designs 4..7 are fresh, unique, and uncached
    const allKeys = new Set(plan8.items.map((i) => i.key));
    const allSeeds = new Set(plan8.items.map((i) => i.seed));
    expect(allKeys.size).toBe(8);
    expect(allSeeds.size).toBe(8);
    for (let i = 4; i < 8; i++) {
      expect(plan8.items[i].cached).toBe(false);
    }
  });

  it('rejects out-of-range numDesigns outside 1..16', () => {
    expect(() =>
      planGeneratorDesigns({ modelId: 'rfd4-proteina', numDesigns: 0, baseSeed: 42 }),
    ).toThrow(RangeError);
    expect(() =>
      planGeneratorDesigns({ modelId: 'rfd4-proteina', numDesigns: 17, baseSeed: 42 }),
    ).toThrow(RangeError);
  });

  it('preserves existing MPNN sequence keys and seeds when extending backbones or sequencesPerBackbone', () => {
    const structs4 = Array.from({ length: 4 }, (_, i) =>
      makeSyntheticStructure(`bb-${i}`, 64 + i, { coordShift: i }),
    );
    const structs8 = [
      ...structs4,
      ...Array.from({ length: 4 }, (_, i) =>
        makeSyntheticStructure(`bb-${i + 4}`, 68 + i, { coordShift: i + 4 }),
      ),
    ];
    const fam = sha256Hex('gen-fam');
    const col4 = buildStructureCollection(fam, structs4);
    const col8 = buildStructureCollection(fam, structs8);

    const mpnn4 = planMpnnFanout(col4, { sequencesPerBackbone: 2, baseSeed: 99, temperature: 0.1 });
    const mpnn8 = planMpnnFanout(col8, {
      sequencesPerBackbone: 4,
      baseSeed: 99,
      temperature: 0.1,
      cachedKeys: new Set(mpnn4.items.map((i) => i.key)),
    });

    expect(mpnn8.cachedCount).toBe(8);
    expect(mpnn8.newCount).toBe(24);
    for (let b = 0; b < 4; b++) {
      for (let q = 0; q < 2; q++) {
        expect(mpnn8.perBackbonePlans[b].items[q].key).toBe(mpnn4.perBackbonePlans[b].items[q].key);
        expect(mpnn8.perBackbonePlans[b].items[q].seed).toBe(mpnn4.perBackbonePlans[b].items[q].seed);
        expect(mpnn8.perBackbonePlans[b].items[q].cached).toBe(true);
      }
    }
  });
});

// ============================================================================
// RFC Gates #3 & #4: Partial Checkpoint Recovery & Idempotent Atomic Commits
// ============================================================================

describe('Wave-6A Gates #3 & #4: Partial checkpoint recovery & atomic idempotent commits', () => {
  it('recovers committed items after mid-campaign cancellation and resumes only uncommitted work', async () => {
    const persistentBackend = new MemoryArtifactBackend({ isDurable: true });
    const store1 = new ArtifactBlobStore({ backend: persistentBackend });

    const plan = planGeneratorDesigns({ modelId: 'rfd4-proteina', numDesigns: 4, baseSeed: 77 });
    const plannedKeys = plan.items.map((i) => i.key);

    const emptyBase: StructureCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'initial',
      familyKey: plan.familyKey,
      revision: 0,
      memberKind: 'StructureArtifact',
      collectionKind: 'independent-candidates',
      label: 'Run 1',
      sourceModelId: 'rfd4-proteina',
      members: [],
      outcomes: plannedKeys.map((k) => ({ key: k, status: 'pending', attempt: 0 })),
      requestedCount: 4,
      complete: false,
    };

    // Commit items 0 and 1, then simulate cancellation during item 2
    for (let idx = 0; idx < 2; idx++) {
      const planned = plan.items[idx];
      const payload = makeSyntheticStructure(`design-${idx}`, 64, { coordShift: idx * 3 });
      const res = await commitCollectionItemAtomic(store1, {
        runId: 'run-alpha',
        nodeId: 'gen-1',
        baseCollection: emptyBase,
        member: {
          key: planned.key,
          ordinal: planned.ordinal,
          memberKind: 'StructureArtifact',
          lineage: {
            ...planned.lineage,
            sourceStructure: {
              artifactId: payload.id,
              payloadDigest: computeStructurePayloadDigest(payload),
              designKey: planned.key,
            },
          },
          provenance: { seedScheme: 'domain-v2', seed: planned.seed },
          summary: { label: payload.label, residueCount: payload.length },
        },
        payload,
        plannedKeys,
        cancelled: idx === 1,
      });
      expect(res.committed).toBe(true);
      expect(res.alreadyCommitted).toBe(false);
    }

    // Simulate reload: fresh ArtifactBlobStore pointing at the same persistent backend
    const store2 = new ArtifactBlobStore({ backend: persistentBackend });
    const recovered = await loadCollectionCheckpoint(store2, 'run-alpha');
    expect(recovered).not.toBeNull();
    expect(recovered!.committedKeys).toEqual([plannedKeys[0], plannedKeys[1]]);
    expect(recovered!.pendingKeys).toEqual([plannedKeys[2], plannedKeys[3]]);
    expect(recovered!.collection.complete).toBe(false);
    expect(recovered!.cancelled).toBe(true);

    // Checkpoints stay dehydrated (no inline heavy typed arrays on manifest members)
    for (const m of recovered!.collection.members) {
      expect(m.payload).toBeUndefined();
      expect(m.payloadRef).toBeDefined();
    }

    // Hydrate item 1 and verify Float32Array coordinates survived round-trip losslessly
    const hydrated1 = await store2.hydratePayload<StructureArtifact>(
      recovered!.collection.members[1] as CollectionMember<StructureArtifact>,
    );
    expect(hydrated1).not.toBeNull();
    expect(hydrated1!.caCoords).toBeInstanceOf(Float32Array);
    expect(hydrated1!.caCoords.length).toBe(64 * 3);
    expect(hydrated1!.caCoords[0]).toBeCloseTo(3, 5);
  });

  it('guarantees idempotent atomic commits under concurrent duplicate writes (no double commit)', async () => {
    const backend = new MemoryArtifactBackend({ isDurable: true });
    const store = new ArtifactBlobStore({ backend });
    const plan = planGeneratorDesigns({ modelId: 'genie3', numDesigns: 2, baseSeed: 5 });
    const plannedKeys = plan.items.map((i) => i.key);
    const base: StructureCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'init',
      familyKey: plan.familyKey,
      revision: 0,
      memberKind: 'StructureArtifact',
      collectionKind: 'independent-candidates',
      label: 'Genie',
      sourceModelId: 'genie3',
      members: [],
      outcomes: [],
      requestedCount: 2,
      complete: false,
    };

    const struct0 = makeSyntheticStructure('g-0', 50, { coordShift: 1 });
    const member0: CollectionMember<StructureArtifact> = {
      key: plan.items[0].key,
      ordinal: 0,
      memberKind: 'StructureArtifact',
      lineage: plan.items[0].lineage,
      summary: { label: 'g-0', residueCount: 50 },
    };

    // Fire two concurrent commits for the exact same memberKey
    const [r1, r2] = await Promise.all([
      commitCollectionItemAtomic(store, {
        runId: 'run-race',
        nodeId: 'n1',
        baseCollection: base,
        member: member0,
        payload: struct0,
        plannedKeys,
      }),
      commitCollectionItemAtomic(store, {
        runId: 'run-race',
        nodeId: 'n1',
        baseCollection: base,
        member: member0,
        payload: struct0,
        plannedKeys,
      }),
    ]);

    expect(r1.committed).toBe(true);
    expect(r1.alreadyCommitted).toBe(false);
    expect(r2.committed).toBe(false);
    expect(r2.alreadyCommitted).toBe(true);

    const cp = await loadCollectionCheckpoint(store, 'run-race');
    expect(cp!.collection.members).toHaveLength(1);
    expect(cp!.revision).toBe(1);

    // Verify saveCollectionCheckpoint & clearCollectionCheckpoints
    const saved = await saveCollectionCheckpoint(store, cp as CollectionCheckpointRecord);
    expect(saved.runId).toBe('run-race');
    await clearCollectionCheckpoints(store, 'run-race');
    expect(await loadCollectionCheckpoint(store, 'run-race')).toBeNull();
  });

  it('losslessly round-trips nested Float32Array, Uint8Array, and Int32Array via binary codec', () => {
    const payload = {
      id: 'fold-1',
      pae: new Float32Array([0.5, 1.25, 12.75, 25.0]),
      mask: new Uint8Array([1, 0, 1, 1]),
      indices: new Int32Array([-1, 0, 42, 1024]),
      nested: { label: 'ok', values: new Float64Array([Math.PI, Math.E]) },
    };
    const encoded = encodePayloadBytes(payload);
    const decoded = decodePayloadBytes<typeof payload>(encoded);
    expect(decoded.pae).toBeInstanceOf(Float32Array);
    expect([...decoded.pae]).toEqual([...payload.pae]);
    expect(decoded.mask).toBeInstanceOf(Uint8Array);
    expect([...decoded.mask]).toEqual([...payload.mask]);
    expect(decoded.indices).toBeInstanceOf(Int32Array);
    expect([...decoded.indices]).toEqual([...payload.indices]);
    expect(decoded.nested.values).toBeInstanceOf(Float64Array);
    expect(decoded.nested.values[0]).toBeCloseTo(Math.PI, 12);
  });
});

// ============================================================================
// RFC Gates #5 & #6: Exact Parent Lookup, Variable Lengths & Cross-Parent Rejection
// ============================================================================

describe('Wave-6A Gates #5 & #6: Exact parent resolution & variable-length designs', () => {
  it('resolves every fold to its exact declared parent even when equal-length designs are reordered', () => {
    const fam = sha256Hex('equal-len-fam');
    const structA = makeSyntheticStructure('struct-A', 64, { coordShift: 0 });
    const structB = makeSyntheticStructure('struct-B', 64, { coordShift: 50 });
    const structC = makeSyntheticStructure('struct-C', 64, { coordShift: 100 });
    const col = buildStructureCollection(fam, [structA, structB, structC]);

    // Deliberately reverse the array order in the lookup pool
    const reorderedPool = [...col.members].reverse();

    for (const parentMember of col.members) {
      const seqKey = makeSequenceKey(fam, parentMember.key, 0);
      const foldKey = makeFoldKey(fam, seqKey, 0, 'af2-monomer');
      const foldMember: CollectionMember<FoldedCandidate> = {
        key: foldKey,
        ordinal: 0,
        memberKind: 'FoldedCandidate',
        lineage: {
          designKey: parentMember.key,
          designIndex: parentMember.ordinal,
          sequenceKey: seqKey,
          foldKey,
          sourceStructure: parentMember.lineage.sourceStructure,
          parentKeys: [seqKey],
        },
        summary: { label: 'fold', residueCount: 64 },
      };

      const resolved = resolveExactParentStructure(foldMember, reorderedPool);
      expect(resolved.ok).toBe(true);
      if (resolved.ok) {
        expect(resolved.structure.id).toBe(parentMember.payload!.id);
        expect(resolved.structure.caCoords[0]).toBe(parentMember.payload!.caCoords[0]);
      }
    }
  });

  it('supports coexisting 63-aa and 78-aa designs and rejects cross-parent / length / digest / unlinked lookups', () => {
    const fam = sha256Hex('varlen-fam');
    const s63 = makeSyntheticStructure('design-63', 63, { coordShift: 1 });
    const s78 = makeSyntheticStructure('design-78', 78, { coordShift: 20 });
    const structs = buildStructureCollection(fam, [s63, s78]);

    const d63 = structs.members[0];
    const d78 = structs.members[1];

    // Valid 63-aa and 78-aa folds resolve cleanly to their respective parents
    const ok63 = resolveExactParentStructure(
      {
        lineage: {
          designKey: d63.key,
          sequenceKey: makeSequenceKey(fam, d63.key, 0),
          sourceStructure: d63.lineage.sourceStructure,
          parentKeys: [d63.key],
        },
        expectedLength: 63,
      },
      structs,
    );
    expect(ok63.ok).toBe(true);
    if (ok63.ok) expect(ok63.structure.length).toBe(63);

    const ok78 = resolveExactParentStructure(
      {
        lineage: {
          designKey: d78.key,
          sequenceKey: makeSequenceKey(fam, d78.key, 0),
          sourceStructure: d78.lineage.sourceStructure,
          parentKeys: [d78.key],
        },
        expectedLength: 78,
      },
      structs,
    );
    expect(ok78.ok).toBe(true);
    if (ok78.ok) expect(ok78.structure.length).toBe(78);

    // Cross-parent aliasing: designKey of 63-aa paired with sourceStructure of 78-aa -> rejected!
    const crossParent = resolveExactParentStructure(
      {
        lineage: {
          designKey: d63.key,
          sourceStructure: d78.lineage.sourceStructure,
          parentKeys: [d63.key],
        },
      },
      structs,
    );
    expect(crossParent.ok).toBe(false);
    if (!crossParent.ok) expect(crossParent.code).toBe('cross-parent-conflict');

    // Length mismatch within declared parent (63 vs 78) -> rejected, never truncated!
    const lenMismatch = resolveExactParentStructure(
      {
        lineage: {
          designKey: d63.key,
          sourceStructure: d63.lineage.sourceStructure,
          parentKeys: [d63.key],
        },
        expectedLength: 78,
      },
      structs,
    );
    expect(lenMismatch.ok).toBe(false);
    if (!lenMismatch.ok) expect(lenMismatch.code).toBe('length-mismatch');

    // Tampered payloadDigest -> rejected!
    const digestMismatch = resolveExactParentStructure(
      {
        lineage: {
          designKey: d63.key,
          sourceStructure: {
            artifactId: s63.id,
            payloadDigest: sha256Hex('wrong-digest'),
            designKey: d63.key,
          },
          parentKeys: [d63.key],
        },
      },
      structs,
    );
    expect(digestMismatch.ok).toBe(false);
    if (!digestMismatch.ok) expect(digestMismatch.code).toBe('digest-mismatch');

    // Standalone unlinked sequence -> rejected, never falls back to array[0] or equal-length!
    const unlinked = resolveExactParentStructure(
      {
        lineage: { parentKeys: [] },
        expectedLength: 63,
      },
      structs,
    );
    expect(unlinked.ok).toBe(false);
    if (!unlinked.ok) expect(unlinked.code).toBe('unlinked-parent');
  });

  it('rejects duplicate member keys, reused sequenceKey as foldKey, and cross-parent aliasing in validateCollectionInvariants', () => {
    const fam = sha256Hex('inv-fam');
    const s0 = makeSyntheticStructure('s0', 40, { coordShift: 0 });
    const s1 = makeSyntheticStructure('s1', 40, { coordShift: 10 });
    const structs = buildStructureCollection(fam, [s0, s1]);

    expect(validateCollectionInvariants(structs).valid).toBe(true);

    const dupKey = makeSequenceKey(fam, structs.members[0].key, 0);
    const badFolds: FoldCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'bad-folds',
      familyKey: fam,
      revision: 1,
      memberKind: 'FoldedCandidate',
      collectionKind: 'multi-sample',
      label: 'Bad Folds',
      sourceModelId: 'af2',
      members: [
        {
          key: dupKey,
          ordinal: 0,
          memberKind: 'FoldedCandidate',
          lineage: {
            designKey: structs.members[0].key,
            sequenceKey: dupKey,
            foldKey: dupKey, // illegal reuse of sequenceKey
            sourceStructure: structs.members[1].lineage.sourceStructure, // cross-parent alias!
            parentKeys: [dupKey],
          },
          summary: { label: 'f1' },
        },
        {
          key: dupKey, // duplicate key!
          ordinal: 1,
          memberKind: 'FoldedCandidate',
          lineage: {
            designKey: structs.members[0].key,
            sequenceKey: dupKey,
            foldKey: makeFoldKey(fam, dupKey, 1, 'af2'),
            sourceStructure: structs.members[0].lineage.sourceStructure,
            parentKeys: [dupKey],
          },
          summary: { label: 'f2' },
        },
      ],
      outcomes: [],
      requestedCount: 2,
      complete: true,
    };

    const validation = validateCollectionInvariants(badFolds, { knownStructures: structs });
    expect(validation.valid).toBe(false);
    const codes = new Set(validation.errors.map((e) => e.code));
    expect(codes.has('fold-reused-sequence-key')).toBe(true);
    expect(codes.has('duplicate-member-key')).toBe(true);
    expect(codes.has('cross-parent-aliasing')).toBe(true);
  });
});

// ============================================================================
// RFC Gates #7, #8, #10: Stable Seeds, Top-K Filter, Fold Fan-out & Design Locks
// ============================================================================

describe('Wave-6A Gates #7, #8, #10: Stable seeds, Top-K filter, fold fan-out & per-design locks', () => {
  function makeSequenceCollectionFixture(): {
    structs: StructureCollection;
    seqs: SequenceCollection;
  } {
    const fam = sha256Hex('campaign-4x2');
    const rawStructs = [
      makeSyntheticStructure('bb-0', 60, {
        coordShift: 0,
        lockedChains: ['A'],
        fixedResidueIndices: [1, 2, 3],
        motifIndices: [10, 11],
      }),
      makeSyntheticStructure('bb-1', 72, {
        coordShift: 10,
        lockedChains: ['B'],
        fixedResidueIndices: [5, 6],
        motifIndices: [20, 21, 22],
      }),
      makeSyntheticStructure('bb-2', 65, { coordShift: 20 }),
      makeSyntheticStructure('bb-3', 80, { coordShift: 30 }),
    ];
    const structs = buildStructureCollection(fam, rawStructs);
    const mpnnPlan = planMpnnFanout(structs, { sequencesPerBackbone: 2, baseSeed: 42 });

    // Assign intentionally varied MPNN scores (lower is better)
    const scores = [1.8, 1.1, 1.5, 1.5, null, 1.3, 1.2, 2.4];
    const members: CollectionMember<SequenceCandidate>[] = mpnnPlan.items.map((item, idx) => ({
      key: item.key,
      ordinal: item.ordinal,
      memberKind: 'SequenceCandidate',
      lineage: item.lineage,
      provenance: { seedScheme: 'domain-v2', seed: item.seed },
      payload: {
        id: `seq-${idx}`,
        key: item.key,
        sequence: 'G'.repeat(item.backboneLength),
        score: scores[idx],
        globalScore: scores[idx] !== null ? scores[idx]! + 0.1 : null,
        sourceModelId: 'proteinmpnn',
        sourceStructureId: item.sourceStructure.artifactId,
        sampleIndex: item.sequenceIndex,
        seed: item.seed,
      },
      summary: {
        label: `seq-${idx}`,
        residueCount: item.backboneLength,
        score: scores[idx],
        globalScore: scores[idx] !== null ? scores[idx]! + 0.1 : null,
      },
    }));

    const outcomes = members.map((m) => ({
      key: m.key,
      status: 'completed' as const,
      attempt: 1,
      lineage: m.lineage,
    }));

    const seqs: SequenceCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: computeCollectionSnapshotId(mpnnPlan.familyKey, members, outcomes),
      familyKey: mpnnPlan.familyKey,
      revision: 1,
      memberKind: 'SequenceCandidate',
      collectionKind: 'independent-candidates',
      label: 'MPNN Sequences',
      sourceModelId: 'proteinmpnn',
      members,
      outcomes,
      requestedCount: members.length,
      complete: true,
    };

    return { structs, seqs };
  }

  it('derives disjoint seeds across domains and keeps fold seeds invariant to sorting, Top-K, and cache skips (Gate #7)', () => {
    const key = 'seq:abcdef0123456789:q0';
    const sGen = deriveStableItemSeed(42, 'generation', key);
    const sMpnn = deriveStableItemSeed(42, 'mpnn', key);
    const sFold = deriveStableItemSeed(42, 'fold', key);
    const sEns = deriveStableItemSeed(42, 'ensemble', key);

    expect(Math.floor(sGen / SEED_DOMAIN_SPAN)).toBe(1);
    expect(Math.floor(sMpnn / SEED_DOMAIN_SPAN)).toBe(2);
    expect(Math.floor(sFold / SEED_DOMAIN_SPAN)).toBe(3);
    expect(Math.floor(sEns / SEED_DOMAIN_SPAN)).toBe(4);
    expect(new Set([sGen, sMpnn, sFold, sEns]).size).toBe(4);

    // Ordinal stage seed helper rejects out-of-range ordinals instead of wrapping
    expect(deriveOrdinalStageSeed('fold', 10, 5)).toBe(3 * SEED_DOMAIN_SPAN + 15);
    expect(() => deriveOrdinalStageSeed('fold', 10, SEED_DOMAIN_SPAN)).toThrow(RangeError);

    const { seqs } = makeSequenceCollectionFixture();

    const planA = planLocalFoldFanout(seqs, {
      foldModelId: 'af2-monomer',
      baseSeed: 2026,
      perDesignTopK: 2,
      maxFolds: 8,
    });

    // Reverse input order + change perDesignTopK to 1 + mark half cached
    const reversedSeqs: SequenceCollection = {
      ...seqs,
      members: [...seqs.members].reverse(),
    };
    const planB = planLocalFoldFanout(reversedSeqs, {
      foldModelId: 'af2-monomer',
      baseSeed: 2026,
      perDesignTopK: 1,
      maxFolds: 4,
      cachedFoldKeys: new Set(planA.selected.slice(0, 4).map((s) => s.foldKey)),
    });

    const seedByFoldKeyA = new Map(planA.selected.map((s) => [s.foldKey, s.seed]));
    for (const itemB of planB.selected) {
      expect(seedByFoldKeyA.get(itemB.foldKey)).toBe(itemB.seed);
    }
  });

  it('produces exact path summary, round-robin fairness, and identical selected/excluded plans (Gate #8)', () => {
    const { seqs } = makeSequenceCollectionFixture();

    // 4 backbones x 2 sequences -> perDesignTopK=1 -> 4 selected -> 4 refolds
    const plan = planLocalFoldFanout(seqs, {
      foldModelId: 'af2-monomer',
      baseSeed: 101,
      perDesignTopK: 1,
      samplesPerSequence: 1,
      maxFolds: 8,
    });

    expect(plan.pathSummary).toBe('4 backbones → 8 sequences → 4 selected → 4 refolds');
    expect(plan.selected).toHaveLength(4);
    expect(plan.excluded).toHaveLength(4);

    // Design 0: score 1.1 (ordinal 1) beats score 1.8 (ordinal 0)
    expect(plan.selected[0].sequenceMember.summary.score).toBe(1.1);
    // Design 1: tie 1.5 vs 1.5 breaks by creation ordinal 0
    expect(plan.selected[1].sequenceMember.ordinal).toBe(0);
    // Design 2: score 1.3 beats null score (nulls sort last!)
    expect(plan.selected[2].sequenceMember.summary.score).toBe(1.3);
    // Design 3: score 1.2 beats 2.4
    expect(plan.selected[3].sequenceMember.summary.score).toBe(1.2);

    // Marking cached folds MUST NOT alter selected/excluded scientific items or planDigest
    const planWithCache = planLocalFoldFanout(seqs, {
      foldModelId: 'af2-monomer',
      baseSeed: 101,
      perDesignTopK: 1,
      samplesPerSequence: 1,
      maxFolds: 8,
      cachedFoldKeys: new Set([plan.selected[0].foldKey, plan.selected[2].foldKey]),
    });
    expect(planWithCache.planDigest).toBe(plan.planDigest);
    expect(planWithCache.selected.map((s) => s.foldKey)).toEqual(plan.selected.map((s) => s.foldKey));
    expect(planWithCache.cachedFoldCount).toBe(2);
    expect(planWithCache.newFoldCount).toBe(2);
    expect(planWithCache.cacheSummary).toBe('2 refolds cached · 2 to run');

    // RFC §5.1 example: N=8, K=4, perDesignTopK=2, S=2, maxFolds=24 -> 32 seqs -> 24 refolds
    const bounds = estimateFanoutBounds({
      numDesigns: 8,
      sequencesPerBackbone: 4,
      perDesignTopK: 2,
      samplesPerSequence: 2,
      maxFolds: 24,
    });
    expect(bounds.maxSequences).toBe(32);
    expect(bounds.maxSelectedSequences).toBe(16);
    expect(bounds.maxFolds).toBe(24);
  });

  it('supports planTopKFilter in both per-backbone (round-robin maxTotal) and global modes', () => {
    const { seqs } = makeSequenceCollectionFixture();

    const perBackbone = planTopKFilter(seqs, {
      mode: 'per-backbone',
      topK: 2,
      maxTotal: 5,
      metric: 'score',
      direction: 'asc',
    });
    expect(perBackbone.selected).toHaveLength(5);
    expect(perBackbone.excluded).toHaveLength(3);
    // First 4 slots must go one-each to the 4 designs via round-robin
    const first4Designs = new Set(perBackbone.selected.slice(0, 4).map((s) => s.designKey));
    expect(first4Designs.size).toBe(4);

    const globalTop3 = planTopKFilter(seqs, {
      mode: 'global',
      topK: 3,
      metric: 'score',
      direction: 'asc',
    });
    expect(globalTop3.selected).toHaveLength(3);
    expect(globalTop3.selected.map((s) => s.metricValue)).toEqual([1.1, 1.2, 1.3]);
    expect(globalTop3.excluded).toHaveLength(5);
  });

  it('preserves design-specific lockedChains, fixedResidueIndices, and motifIndices in planMpnnFanout (Gate #10)', () => {
    const { structs } = makeSequenceCollectionFixture();
    const mpnn = planMpnnFanout(structs, { sequencesPerBackbone: 2, baseSeed: 7 });

    expect(mpnn.perBackbonePlans[0].lockedChains).toEqual(['A']);
    expect(mpnn.perBackbonePlans[0].fixedResidueIndices).toEqual([1, 2, 3]);
    expect(mpnn.perBackbonePlans[0].motifIndices).toEqual([10, 11]);

    expect(mpnn.perBackbonePlans[1].lockedChains).toEqual(['B']);
    expect(mpnn.perBackbonePlans[1].fixedResidueIndices).toEqual([5, 6]);
    expect(mpnn.perBackbonePlans[1].motifIndices).toEqual([20, 21, 22]);

    expect(mpnn.perBackbonePlans[2].lockedChains).toEqual([]);
    expect(mpnn.perBackbonePlans[2].fixedResidueIndices).toEqual([]);
    expect(mpnn.perBackbonePlans[2].motifIndices).toEqual([]);
  });
});

// ============================================================================
// RFC Gate #9: Campaign & Per-Design Summaries (Null Denominators & Failures)
// ============================================================================

describe('Wave-6A Gate #9: Campaign summaries, failures, exclusions & null denominators', () => {
  it('returns null (never fabricated 0) when denominators are zero or only confidence-less/failed/pending items exist', () => {
    const empty = computeCampaignSummary([], { requestedDesigns: 0 });
    expect(empty.foldPassRate).toBeNull();
    expect(empty.designHitRate).toBeNull();
    expect(empty.meanPerDesignFoldPassRate).toBeNull();
    expect(empty.coverage.completedFraction).toBeNull();
    expect(empty.coverage.evaluableFraction).toBeNull();
    expect(empty.conservativeCampaignYield).toBeNull();

    const records: FoldEvaluationRecord[] = [
      {
        foldKey: 'f1',
        designKey: 'd1',
        sequenceKey: 's1',
        status: 'completed',
        confidenceAvailable: false, // confidence-less ensemble sample -> ineligible, NOT failed!
        designable: null,
      },
      {
        foldKey: 'f2',
        designKey: 'd1',
        sequenceKey: 's2',
        status: 'failed',
      },
      {
        foldKey: 'f3',
        designKey: 'd1',
        sequenceKey: 's3',
        status: 'excluded',
      },
      {
        foldKey: 'f4',
        designKey: 'd2',
        sequenceKey: 's4',
        status: 'pending',
      },
    ];

    const summary = computeCampaignSummary(records, { requestedDesigns: 2 });
    expect(summary.evaluableFolds).toBe(0);
    expect(summary.ineligibleFolds).toBe(1);
    expect(summary.failedFolds).toBe(1);
    expect(summary.excludedFolds).toBe(1);
    expect(summary.pendingFolds).toBe(1);
    expect(summary.plannedFolds).toBe(3); // excludes the 'excluded' record
    expect(summary.foldPassRate).toBeNull();
    expect(summary.designHitRate).toBeNull();
    expect(summary.meanPerDesignFoldPassRate).toBeNull();
    expect(summary.conservativeCampaignYield).toBe(0); // 0 passing / 2 requested
    expect(summary.complete).toBe(false);
  });

  it('computes exact foldPassRate, designHitRate, meanPerDesignFoldPassRate, coverage & conservativeCampaignYield', () => {
    const records: FoldEvaluationRecord[] = [
      // Design 1: 2 evaluable folds -> 2 pass (rate = 1.0), 1 excluded
      { foldKey: 'd1-f1', designKey: 'd1', sequenceKey: 'd1-s1', status: 'completed', designable: true },
      { foldKey: 'd1-f2', designKey: 'd1', sequenceKey: 'd1-s2', status: 'completed', designable: true },
      { foldKey: 'd1-f3', designKey: 'd1', sequenceKey: 'd1-s3', status: 'excluded' },
      // Design 2: 2 evaluable folds -> 0 pass (rate = 0.0), 1 failed
      { foldKey: 'd2-f1', designKey: 'd2', sequenceKey: 'd2-s1', status: 'completed', designable: false },
      { foldKey: 'd2-f2', designKey: 'd2', sequenceKey: 'd2-s2', status: 'completed', designable: false },
      { foldKey: 'd2-f3', designKey: 'd2', sequenceKey: 'd2-s3', status: 'failed' },
      // Design 3: 1 pending fold (unfinished)
      { foldKey: 'd3-f1', designKey: 'd3', sequenceKey: 'd3-s1', status: 'pending' },
    ];

    const perDesign = computeDesignPassSummaries(records, { designKeys: ['d1', 'd2', 'd3', 'd4'] });
    expect(perDesign).toHaveLength(4);
    expect(perDesign[0].foldPassRate).toBe(1);
    expect(perDesign[0].hasPassingSequence).toBe(true);
    expect(perDesign[0].complete).toBe(true);
    expect(perDesign[1].foldPassRate).toBe(0);
    expect(perDesign[1].hasPassingSequence).toBe(false);
    expect(perDesign[1].complete).toBe(true);
    expect(perDesign[2].foldPassRate).toBeNull();
    expect(perDesign[2].complete).toBe(false);
    expect(perDesign[3].foldPassRate).toBeNull();

    const campaign = computeCampaignSummary(records, {
      requestedDesigns: 4,
      designKeys: ['d1', 'd2', 'd3', 'd4'],
      samplesPerSequence: 2,
      predictorModelId: 'af2-monomer',
    });

    expect(campaign.foldPassRate).toBeCloseTo(2 / 4, 6);
    expect(campaign.designHitRate).toBeCloseTo(1 / 2, 6); // 1 passing / 2 evaluated designs
    expect(campaign.meanPerDesignFoldPassRate).toBeCloseTo((1 + 0) / 2, 6);
    expect(campaign.conservativeCampaignYield).toBeCloseTo(1 / 4, 6); // includes unfinished d3 & d4!
    expect(campaign.coverage.plannedFolds).toBe(6);
    expect(campaign.coverage.completedFolds).toBe(4);
    expect(campaign.coverage.evaluableFolds).toBe(4);
    expect(campaign.coverage.completedFraction).toBeCloseTo(4 / 6, 6);
    expect(campaign.aggregationPolicy).toBe('at-least-one-of-2-passes');
    expect(campaign.complete).toBe(false);
  });
});

// ============================================================================
// RFC Gates #11 & #12: Bounded Pin Memory & Quota-Exceeded Resilience
// ============================================================================

describe('Wave-6A Gates #11 & #12: Bounded hydrated memory, pinning & quota errors', () => {
  it('evicts unpinned hydrated PAE payloads while preserving pinned keys within memory budget (Gate #11)', async () => {
    const backend = new MemoryArtifactBackend({ isDurable: false });
    // Each payload has a 16 KiB Float32Array; allow ~64 KiB unpinned, ~128 KiB total
    const store = new ArtifactBlobStore({
      backend,
      maxHydratedBytes: 64 * 1024,
      maxTotalBytes: 128 * 1024,
      maxHydratedEntries: 16,
    });

    // In-memory fallback must explicitly report isDurable === false
    expect(store.isDurable).toBe(false);

    const refs = [];
    for (let i = 0; i < 60; i++) {
      const pae = new Float32Array(4096); // 16 KiB
      pae[0] = i;
      const ref = await store.putBlob({ id: `fold-${i}`, pae });
      refs.push(ref);
      store.registerMemberBlob(`fold-key-${i}`, ref);
    }

    // Storing 60 blobs does NOT auto-hydrate them into RAM
    expect(store.getMemoryStats().hydratedCount).toBe(0);

    // Hydrate and pin key 0 and key 1
    await store.getBlob(refs[0]);
    await store.getBlob(refs[1]);
    store.pinKeys(['fold-key-0', 'fold-key-1']);

    // Now hydrate 30 more blobs: unpinned entries must evict while pinned 0 & 1 stay resident
    for (let i = 2; i < 32; i++) {
      const loaded = await store.getBlob<{ id: string; pae: Float32Array }>(refs[i]);
      expect(loaded?.pae[0]).toBe(i);
    }

    const stats = store.getMemoryStats();
    expect(stats.hydratedBytes).toBeLessThanOrEqual(stats.maxTotalBytes);
    expect(stats.unpinnedBytes).toBeLessThanOrEqual(stats.maxHydratedBytes);
    expect(stats.evictionCount).toBeGreaterThan(20);
    expect(store.isHydrated('fold-key-0')).toBe(true);
    expect(store.isHydrated('fold-key-1')).toBe(true);

    // Unpinning releases protection and enforces budget immediately
    store.unpinKeys(['fold-key-0', 'fold-key-1']);
    expect(store.getMemoryStats().pinnedCount).toBe(0);

    // Attempting to pin more than maxTotalBytes throws PinBudgetExceededError
    const hugeRef = await store.putBlob({ huge: new Float32Array(40_000) }); // 160 KiB > 128 KiB
    const hugeMember: CollectionMember<unknown> = {
      key: 'huge-member',
      ordinal: 99,
      memberKind: 'FoldedCandidate',
      lineage: { parentKeys: [] },
      payloadRef: hugeRef,
      summary: { label: 'huge' },
    };
    await expect(store.hydratePayload(hugeMember, { pin: true })).rejects.toThrow(
      PinBudgetExceededError,
    );
  });

  it('surfaces StorageQuotaError cleanly and flags quotaExceeded without corrupting prior blobs (Gate #12)', async () => {
    const onQuota = vi.fn();
    const backend = new MemoryArtifactBackend({ maxQuotaBytes: 32 * 1024 });
    const store = new ArtifactBlobStore({ backend, onQuotaExceeded: onQuota });

    const firstRef = await store.putBlob({ ok: true, data: new Uint8Array(8 * 1024) });
    expect(await store.hasBlob(firstRef)).toBe(true);
    expect(store.quotaExceeded).toBe(false);

    // Exceed 32 KiB quota
    await expect(store.putBlob({ big: new Uint8Array(40 * 1024) })).rejects.toThrow(
      StorageQuotaError,
    );
    expect(store.quotaExceeded).toBe(true);
    expect(onQuota).toHaveBeenCalledTimes(1);

    // Prior committed blob remains intact and readable
    const loaded = await store.getBlob<{ ok: boolean; data: Uint8Array }>(firstRef);
    expect(loaded?.ok).toBe(true);
    expect(loaded?.data.byteLength).toBe(8 * 1024);
  });
});

// ============================================================================
// RFC Gate #13: Snapshot Export Round-Trip & Heterogeneous Topology Guard
// ============================================================================

describe('Wave-6A Gate #13: Snapshot streaming export & heterogeneous topology guard', () => {
  it('exports FASTA with stable keys, model, seed, and chain boundaries', () => {
    const fam = sha256Hex('fasta-fam');
    const designKey = makeDesignKey(fam, 0);
    const seqKey = makeSequenceKey(fam, designKey, 0);
    const seqCol: SequenceCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'snap-fasta',
      familyKey: fam,
      revision: 1,
      memberKind: 'SequenceCandidate',
      collectionKind: 'independent-candidates',
      label: 'MPNN',
      sourceModelId: 'proteinmpnn',
      members: [
        {
          key: seqKey,
          ordinal: 0,
          memberKind: 'SequenceCandidate',
          lineage: { designKey, sequenceKey: seqKey, parentKeys: [designKey] },
          provenance: { seedScheme: 'domain-v2', seed: 555123 },
          payload: {
            id: 'mpnn-0',
            key: seqKey,
            sequence: 'ACDEFGHIKL',
            score: 1.2345,
            globalScore: 1.3,
            sourceModelId: 'proteinmpnn',
            chainLengths: [4, 6],
          },
          summary: { label: 'mpnn-0', residueCount: 10, score: 1.2345 },
        },
      ],
      outcomes: [],
      requestedCount: 1,
      complete: true,
    };

    const fasta = exportCollectionFasta(seqCol);
    expect(fasta).toContain(`>${seqKey} | design=${designKey} | model=proteinmpnn | ordinal=0 | chains=4,6 | seed=555123 | score=1.2345`);
    expect(fasta).toContain('ACDE:FGHIKL');
  });

  it('exports CSV with lineage keys, exact reference ID, exclusions, thresholds & confidence availability', () => {
    const fam = sha256Hex('csv-fam');
    const d0 = makeDesignKey(fam, 0);
    const s0 = makeSequenceKey(fam, d0, 0);
    const f0 = makeFoldKey(fam, s0, 0, 'af2-monomer');
    const s1 = makeSequenceKey(fam, d0, 1);

    const scoreRow: DesignabilityRow = {
      structureId: 'bb-0',
      sequenceId: s0,
      candidateKey: s0,
      sourceStructureId: 'bb-0',
      sequence: 'A'.repeat(63),
      scRmsd: 1.12,
      scTm: 0.91,
      meanPlddt: 88.4,
      confidenceAvailable: true,
      ptm: 0.87,
      iptm: null,
      designable: true,
      referenceMode: 'self-consistency',
      alignedFoldPdb: '',
      designedPdb: '',
      plddt: new Float32Array(63).fill(88.4),
      geometry: {
        length: 63,
        rg: 11,
        meanCaDist: 3.8,
        chainBreaks: 0,
        clashes: 0,
        helixPct: 60,
        sheetPct: 10,
        coilPct: 30,
      },
      foldElapsedMs: 1200,
    };

    const scores: DesignabilityCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'snap-csv-1',
      familyKey: fam,
      revision: 1,
      memberKind: 'DesignabilityRow',
      collectionKind: 'multi-sample',
      label: 'Scores',
      sourceModelId: 'af2-monomer',
      members: [
        {
          key: f0,
          ordinal: 0,
          memberKind: 'DesignabilityRow',
          lineage: {
            designKey: d0,
            sequenceKey: s0,
            foldKey: f0,
            sourceStructure: { artifactId: 'bb-0', payloadDigest: 'deadbeef' },
            parentKeys: [s0],
          },
          provenance: { seedScheme: 'domain-v2', seed: 999 },
          payload: scoreRow,
          summary: {
            label: 'score-0',
            residueCount: 63,
            confidence: 88.4,
            confidenceAvailable: true,
            designable: true,
            scRmsd: 1.12,
          },
        },
      ],
      outcomes: [
        { key: f0, status: 'completed', attempt: 1 },
        {
          key: s1,
          status: 'excluded',
          attempt: 0,
          reason: 'Excluded by perDesignTopK=1',
          lineage: { designKey: d0, sequenceKey: s1, parentKeys: [d0] },
        },
      ],
      requestedCount: 2,
      complete: true,
      aggregationPolicy: 'at-least-one-of-1-passes',
    };

    const csv = exportCollectionCsv(scores, {
      thresholds: { scRmsd: 2.0, plddt: 70 },
    });
    const lines = csv.trim().split('\n');
    expect(lines).toHaveLength(3); // header + 1 completed + 1 excluded
    expect(lines[0]).toContain('reference_structure_id');
    expect(lines[0]).toContain('confidence_available');
    expect(lines[1]).toContain(d0);
    expect(lines[1]).toContain(f0);
    expect(lines[1]).toContain('bb-0');
    expect(lines[1]).toContain('self-consistency');
    expect(lines[1]).toContain(',2,70,'); // thresholds
    expect(lines[2]).toContain('excluded');
    expect(lines[2]).toContain('Excluded by perDesignTopK=1');
  });

  it('exports homogeneous multi-MODEL PDB in explicit order and REFUSES heterogeneous variable-length topologies', () => {
    const fam = sha256Hex('pdb-fam');
    const homA = makeSyntheticStructure('h-a', 20, { coordShift: 0 });
    const homB = makeSyntheticStructure('h-b', 20, { coordShift: 5 });
    const homCol = buildStructureCollection(fam, [homA, homB]);

    // Explicit reversed selection order
    const selectedOrder = [homCol.members[1].key, homCol.members[0].key];
    const multi = exportCollectionMultiModelPdb(homCol, {
      selectedKeys: selectedOrder,
      title: 'HOMOGENEOUS OVERLAY',
    });
    expect(multi.manifest.modelCount).toBe(2);
    expect(multi.manifest.homogeneousTopology).toBe(true);
    expect(multi.manifest.models.map((m) => m.key)).toEqual(selectedOrder);
    expect(multi.pdbText).toContain('MODEL        1');
    expect(multi.pdbText).toContain('MODEL        2');
    expect(multi.pdbText).toContain(`KEY=${selectedOrder[0]}`);

    // Variable-length 63 vs 78 aa -> MUST throw HeterogeneousTopologyError
    const het63 = makeSyntheticStructure('v-63', 63, { coordShift: 0 });
    const het78 = makeSyntheticStructure('v-78', 78, { coordShift: 2 });
    const hetCol = buildStructureCollection(fam, [het63, het78]);
    expect(() => exportCollectionMultiModelPdb(hetCol)).toThrow(HeterogeneousTopologyError);

    // Same length but different atom topology (CA-only vs N/CA/C/O) -> MUST also refuse
    const caOnly = makeSyntheticStructure('ca-20', 20, { atomsPerRes: ['CA'] });
    const mixedAtomCol = buildStructureCollection(fam, [homA, caOnly]);
    expect(() => exportCollectionMultiModelPdb(mixedAtomCol)).toThrow(HeterogeneousTopologyError);
  });

  it('streams heterogeneous bundle entries one at a time, preserving mmCIF, tracks with units, and manifest round-trip', async () => {
    const backend = new MemoryArtifactBackend();
    const store = new ArtifactBlobStore({ backend, maxHydratedEntries: 2 });

    const fam = sha256Hex('bundle-fam');
    const s63 = makeSyntheticStructure('holo-63', 63, {
      coordShift: 0,
      cifText: 'data_holo63\n#\nloop_\n_atom_site.id\n1\n',
    });
    const s78 = makeSyntheticStructure('apo-78', 78, { coordShift: 15 });

    const ref63 = await store.putBlob(s63);
    const ref78 = await store.putBlob(s78);

    const d63 = makeDesignKey(fam, 0);
    const d78 = makeDesignKey(fam, 1);

    // Store dehydrated structure collection (only BlobRefs on manifest)
    const structures: StructureCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'snap-bundle-1',
      familyKey: fam,
      revision: 1,
      memberKind: 'StructureArtifact',
      collectionKind: 'independent-candidates',
      label: 'Mixed Structures',
      sourceModelId: 'rfd4-proteina',
      members: [
        {
          key: d63,
          ordinal: 0,
          memberKind: 'StructureArtifact',
          lineage: {
            designKey: d63,
            designIndex: 0,
            sourceStructure: { artifactId: s63.id, payloadDigest: computeStructurePayloadDigest(s63), designKey: d63 },
            parentKeys: [],
          },
          payloadRef: ref63,
          summary: { label: 'holo-63', residueCount: 63 },
        },
        {
          key: d78,
          ordinal: 1,
          memberKind: 'StructureArtifact',
          lineage: {
            designKey: d78,
            designIndex: 1,
            sourceStructure: { artifactId: s78.id, payloadDigest: computeStructurePayloadDigest(s78), designKey: d78 },
            parentKeys: [],
          },
          payloadRef: ref78,
          summary: { label: 'apo-78', residueCount: 78 },
        },
      ],
      outcomes: [
        { key: d63, status: 'completed', attempt: 1 },
        { key: d78, status: 'completed', attempt: 1 },
      ],
      requestedCount: 2,
      complete: true,
    };

    const track: ResidueTrackArtifact = {
      kind: 'ResidueTrack',
      id: 'dyna-1',
      sourceModelId: 'dyna1',
      quantity: 'exchange-probability',
      unit: 'probability',
      residues: 'ACD',
      residueIds: ['A1', 'A2', 'A3'],
      values: new Float32Array([0.1, 0.5, 0.9]),
    };
    const residueTracks: ResidueTrackCollection = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'tracks-1',
      familyKey: fam,
      revision: 1,
      memberKind: 'ResidueTrack',
      collectionKind: 'independent-candidates',
      label: 'Tracks',
      sourceModelId: 'dyna1',
      members: [
        {
          key: 'track:0',
          ordinal: 0,
          memberKind: 'ResidueTrack',
          lineage: { designKey: d63, parentKeys: [d63] },
          payload: track,
          summary: { label: 'dyna-1', residueCount: 3 },
        },
      ],
      outcomes: [],
      requestedCount: 1,
      complete: true,
    };

    const loadCalls: string[] = [];
    const emitted = new Map<string, string>();
    const gen = buildCollectionBundleEntries(
      {
        snapshotId: 'snap-bundle-1',
        familyKey: fam,
        structures,
        residueTracks,
        thresholds: { scRmsd: 2.0, plddt: 70 },
      },
      {
        loadPayload: async (member) => {
          loadCalls.push(member.key);
          return store.hydratePayload(member);
        },
      },
    );

    // Consume lazily one entry at a time
    while (true) {
      const next = await gen.next();
      if (next.done) break;
      const entry = next.value;
      expect(typeof entry.content).toBe('string');
      emitted.set(entry.path, entry.content as string);
    }

    // Verify lazy streaming order and preserved mmCIF + separate PDB for heterogeneous designs
    expect(loadCalls).toContain(d63);
    expect(loadCalls).toContain(d78);
    expect([...emitted.keys()].some((p) => p.endsWith('.cif'))).toBe(true);
    expect([...emitted.keys()].some((p) => p.endsWith('.pdb'))).toBe(true);
    expect([...emitted.keys()].some((p) => p.startsWith('tracks/') && p.endsWith('.csv'))).toBe(true);
    expect(emitted.has('manifest.json')).toBe(true);

    // Round-trip inspect manifest.json
    const manifest = JSON.parse(emitted.get('manifest.json')!) as CollectionBundleManifest;
    expect(manifest.schema).toBe('denovo.export-manifest/v1');
    expect(manifest.snapshotId).toBe('snap-bundle-1');
    expect(manifest.heterogeneousTopology).toBe(true);
    expect(manifest.counts.structures).toBe(2);
    expect(manifest.counts.residueTracks).toBe(1);
    for (const idxEntry of manifest.entries) {
      expect(emitted.has(idxEntry.path)).toBe(true);
      const actualDigest = sha256Hex(emitted.get(idxEntry.path)!);
      expect(actualDigest).toBe(idxEntry.digest);
    }

    // Residue track preserves unit
    const trackCsv = [...emitted.entries()].find(([k]) => k.startsWith('tracks/'))![1];
    expect(trackCsv).toContain('exchange-probability,probability');
  });
});

// ============================================================================
// Legacy Normalization & Representative Selection
// ============================================================================

describe('Wave-6A Legacy normalization & representative selection', () => {
  it('lifts legacy StructureArtifact, SequenceSet, FoldedStructure, and DesignabilityMetrics cleanly', () => {
    const struct = makeSyntheticStructure('legacy-s1', 30);
    const structCol = normalizeToCollection(struct);
    expect(structCol.kind).toBe('Collection');
    expect(structCol.members).toHaveLength(1);
    expect(structCol.members[0].payload).toBe(struct);
    expect(validateCollectionInvariants(structCol).valid).toBe(true);

    // Calling normalizeToCollection on an existing collection returns it unchanged
    expect(normalizeToCollection(structCol)).toBe(structCol);

    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seqset-1',
      parentStructureId: 'legacy-s1',
      sourceStructureId: 'legacy-s1',
      sourceModelId: 'proteinmpnn',
      candidates: [
        {
          id: 'mpnn-1',
          sequence: 'A'.repeat(30),
          score: 1.4,
          globalScore: 1.5,
          sourceModelId: 'proteinmpnn',
          sampleIndex: 0,
        },
      ],
    };
    const seqCol = normalizeToCollection(seqSet);
    expect(seqCol.members).toHaveLength(1);
    expect(seqCol.members[0].lineage.designKey).toBe('legacy-s1');

    const folded: FoldedStructure = {
      kind: 'FoldedStructure',
      id: 'fold-legacy-1',
      parentStructureId: 'legacy-s1',
      folds: [
        {
          sequenceId: 'mpnn-1',
          sequence: 'A'.repeat(30),
          foldModelId: 'af2',
          pdbText: struct.pdbText,
          caCoords: struct.caCoords,
          plddt: new Float32Array(30).fill(75),
          meanPlddt: 75,
          ptm: 0.72,
          elapsedMs: 500,
        },
        {
          sequenceId: 'mpnn-2',
          sequence: 'A'.repeat(30),
          foldModelId: 'af2',
          pdbText: struct.pdbText,
          caCoords: struct.caCoords,
          plddt: new Float32Array(30).fill(92),
          meanPlddt: 92,
          ptm: 0.9,
          elapsedMs: 510,
        },
      ],
      excludedCandidates: [
        { key: 'mpnn-3', id: 'mpnn-3', index: 2, reason: 'Excluded by maxSequences=2' },
      ],
    };
    const foldCol = normalizeToCollection(folded);
    expect(foldCol.members).toHaveLength(2);
    expect(foldCol.representativeSelection?.selectedKey).toBe(foldCol.members[1].key);
    expect(foldCol.outcomes.some((o) => o.status === 'excluded' && o.key === 'mpnn-3')).toBe(true);

    const metrics: DesignabilityMetrics = {
      kind: 'DesignabilityMetrics',
      id: 'metrics-1',
      rows: [],
      best: null,
      passRate: 0,
    };
    const metricsCol = normalizeToCollection(metrics);
    expect(metricsCol.members).toHaveLength(0);
    expect(selectRepresentativeMember(metricsCol)).toBeNull();
  });

  it('honors explicit key, ordered-frames final frame, and confidence-less ordinal fallback in selectRepresentativeMember', () => {
    const framesCol: DesignCollection<unknown> = {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: 'traj-1',
      familyKey: 'fam',
      revision: 1,
      memberKind: 'StructureArtifact',
      collectionKind: 'ordered-frames',
      label: 'Denoising',
      sourceModelId: 'rfd4',
      members: [
        { key: 'f-0', ordinal: 0, memberKind: 'StructureArtifact', lineage: { parentKeys: [] }, summary: { label: '0' } },
        { key: 'f-1', ordinal: 1, memberKind: 'StructureArtifact', lineage: { parentKeys: [] }, summary: { label: '1' } },
        { key: 'f-2', ordinal: 2, memberKind: 'StructureArtifact', lineage: { parentKeys: [] }, summary: { label: '2' } },
      ],
      outcomes: [],
      requestedCount: 3,
      complete: true,
    };

    const finalFrame = selectRepresentativeMember(framesCol);
    expect(finalFrame?.member.key).toBe('f-2');
    expect(finalFrame?.selection.criterion).toBe('final-committed-frame');

    const explicit = selectRepresentativeMember(framesCol, { explicitKey: 'f-0' });
    expect(explicit?.member.key).toBe('f-0');
    expect(explicit?.selection.criterion).toBe('explicit-selection');

    const confLessFolds: DesignCollection<unknown> = {
      ...framesCol,
      id: 'bioemu-1',
      memberKind: 'FoldedCandidate',
      collectionKind: 'multi-sample',
      members: [
        {
          key: 's-1',
          ordinal: 1,
          memberKind: 'FoldedCandidate',
          lineage: { parentKeys: [] },
          summary: { label: 's1', confidence: null, confidenceAvailable: false },
        },
        {
          key: 's-0',
          ordinal: 0,
          memberKind: 'FoldedCandidate',
          lineage: { parentKeys: [] },
          summary: { label: 's0', confidence: null, confidenceAvailable: false },
        },
      ],
    };
    const fallback = selectRepresentativeMember(confLessFolds);
    expect(fallback?.member.key).toBe('s-0');
    expect(fallback?.selection.scoreValue).toBeNull();
  });
});
