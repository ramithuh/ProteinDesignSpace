import { describe, expect, it } from 'vitest';
import {
  ARTIFACT_ENVELOPE_SCHEMA_VERSION,
  PluginRegistry,
  computeBoltzmannWeights,
  runIterativeLoop,
  type ArtifactEnvelope,
  type ConditionIntent,
  type DomainPack,
  type ModelPlugin,
} from '$lib/kernel';
import { PROTEIN_DOMAIN_ID, proteinBiopolymerDomainPack } from '$lib/domains/protein-biopolymer';

describe('Modern Structural CompBio Kernel Extensibility (2025–2027 Paradigms)', () => {
  it('1. Closed-loop iterative optimization (BindCraft / BoltzDesign / active learning) stamps round lineage & converges deterministically', async () => {
    const scores = [0.42, 0.71, 0.89, 0.91];
    const result = await runIterativeLoop<{ seq: string }, { ipTM: number }>({
      spec: {
        loopId: 'bindcraft-loop',
        maxRounds: 6,
        baseSeed: 1337,
        objectiveDirection: 'maximize',
      },
      executeRound: async ({ roundIndex, seed, prevState }) => {
        const seq = prevState ? `${prevState.seq}A` : 'MKT';
        const ipTM = scores[roundIndex] ?? 0.91;
        const env: ArtifactEnvelope<'FoldedStructure', { seq: string; seed: number }> = {
          kind: 'FoldedStructure',
          schemaVersion: ARTIFACT_ENVELOPE_SCHEMA_VERSION,
          id: `fold-r${roundIndex}`,
          sourceModelId: 'localfold-af2',
          domainId: PROTEIN_DOMAIN_ID,
          payload: { seq, seed },
        };
        return {
          state: { seq },
          artifacts: { fold: env },
          objectiveScore: ipTM,
          metrics: { ipTM },
        };
      },
      stopWhen: (r) => (r.objectiveScore ?? 0) >= 0.85,
    });

    expect(result.status).toBe('converged');
    expect(result.rounds).toHaveLength(3);
    expect(result.bestRound?.roundIndex).toBe(2);
    expect(result.bestRound?.objectiveScore).toBe(0.89);
    expect(result.rounds[0].artifacts.fold.lineage?.roundIndex).toBe(0);
    expect(result.rounds[0].artifacts.fold.lineage?.parentRoundIndex).toBeUndefined();
    expect(result.rounds[2].artifacts.fold.lineage?.roundIndex).toBe(2);
    expect(result.rounds[2].artifacts.fold.lineage?.parentRoundIndex).toBe(1);
    // Distinct deterministic per-round seeds
    expect(new Set(result.rounds.map((r) => r.seed)).size).toBe(3);
  });

  it('2. Conformational ensembles & thermodynamics (BioEmu / MD / MLIP) compute numerically stable Boltzmann weights', () => {
    // Include very large energy offset (-5000 kcal/mol) to prove log-sum-exp stability (no underflow/NaN)
    const energies = new Float32Array([-5000.0, -4999.404, -4998.0]);
    const weights = computeBoltzmannWeights(energies, 300, 'kcal/mol');

    expect(weights).toHaveLength(3);
    const total = weights[0] + weights[1] + weights[2];
    expect(total).toBeCloseTo(1.0, 5);
    expect(weights[0]).toBeGreaterThan(weights[1]);
    expect(weights[1]).toBeGreaterThan(weights[2]);
    // At 300K, kT ≈ 0.596 kcal/mol -> deltaE = 0.596 kcal/mol gives ratio ≈ e^-1 ≈ 0.368
    expect(weights[1] / weights[0]).toBeCloseTo(Math.exp(-1), 2);

    const ensembleEnv: ArtifactEnvelope<'TrajectoryFrames', { frames: number }> = {
      kind: 'TrajectoryFrames',
      schemaVersion: ARTIFACT_ENVELOPE_SCHEMA_VERSION,
      id: 'bioemu-ens-1',
      sourceModelId: 'bioemu',
      domainId: PROTEIN_DOMAIN_ID,
      lineage: {
        collectionKind: 'conformational-ensemble',
        ensembleMeta: {
          ensembleKind: 'conformational-ensemble',
          frameCount: 3,
          weights,
          energies,
          energyUnit: 'kcal/mol',
          temperatureK: 300,
        },
      },
      payload: { frames: 3 },
    };
    expect(ensembleEnv.lineage?.ensembleMeta?.weights?.[0]).toBeCloseTo(weights[0], 6);
  });

  it('3. Atom-level ConditionIntent expresses covalent warheads, cyclic peptides, metal coordination, restraints & symmetry (RFD3 / Boltz-2 / Chai)', () => {
    const intent: ConditionIntent<{ targetName: string }> = {
      domainId: PROTEIN_DOMAIN_ID,
      intentKind: 'all-atom-covalent-enzyme',
      payload: { targetName: 'Mpro-covalent' },
      entities: [
        {
          entityId: 'A',
          entityKind: 'protein',
          sequence: 'SGFRKMAFPS',
          modifications: [{ position: 1, ccdCode: 'ACE' }],
        },
        {
          entityId: 'LIG',
          entityKind: 'ligand-smiles',
          smiles: 'O=C(NC)C#N',
        },
        {
          entityId: 'ZN1',
          entityKind: 'ion',
          ccdCode: 'ZN',
        },
      ],
      bonds: [
        {
          kind: 'covalent',
          from: { entityId: 'A', chain: 'A', resSeq: 145, atomName: 'SG' },
          to: { entityId: 'LIG', atomName: 'C12' },
          bondOrder: 1,
          targetDistanceAngstrom: 1.82,
        },
        {
          kind: 'metal-coordination',
          from: { entityId: 'A', chain: 'A', resSeq: 41, atomName: 'NE2' },
          to: { entityId: 'ZN1', atomName: 'ZN' },
          targetDistanceAngstrom: 2.05,
        },
      ],
      restraints: [
        {
          kind: 'distance',
          selectors: [
            { entityId: 'A', chain: 'A', resSeq: 41, atomName: 'ND1' },
            { entityId: 'A', chain: 'A', resSeq: 145, atomName: 'SG' },
          ],
          min: 2.8,
          max: 3.6,
          unit: 'angstrom',
        },
      ],
      symmetry: { pointGroup: 'C2', copies: 2 },
    };

    expect(intent.entities).toHaveLength(3);
    expect(intent.bonds?.[0].from.atomName).toBe('SG');
    expect(intent.restraints?.[0].max).toBe(3.6);
    expect(intent.symmetry?.pointGroup).toBe('C2');
  });

  it('4. Multi-role omni foundation models (Boltz-2 Fold+Affinity, ESM3) register multiple stages & planner roles cleanly', () => {
    const propertyPack: DomainPack = {
      id: 'property-prediction',
      displayName: 'Property Prediction',
      stages: [{ id: 'property-pred', label: 'Property Prediction', order: 6 }],
      portTypes: [
        {
          typeId: 'PropertyTable',
          domainId: 'property-prediction',
          label: 'Property Table',
          color: { bg: '#22c55e', fg: '#fff', border: '#16a34a' },
          peekPriority: 75,
          summarize: (art) => ({ id: art.id }),
          viewerId: 'prop-table-viewer',
        },
      ],
      artifactViewers: [
        {
          id: 'prop-table-viewer',
          domainId: 'property-prediction',
          label: 'Property Table',
          componentLoader: async () => ({ default: {} }),
        },
      ],
    };

    const reg = new PluginRegistry();
    reg.registerDomainPack(proteinBiopolymerDomainPack);
    reg.registerDomainPack(propertyPack);

    const boltz2Omni: ModelPlugin = {
      id: 'boltz2-affinity',
      domainId: PROTEIN_DOMAIN_ID,
      manifest: {
        id: 'boltz2-affinity',
        displayName: 'Boltz-2 (Structure + Affinity)',
        shortName: 'Boltz-2',
        stage: 'fold',
        secondaryStages: ['score', 'property-pred'],
        runtime: 'wgsl-custom',
        estimatedVramMB: 1800,
        approxWeightsMB: 450,
        licenseNote: 'MIT',
        description: 'Joint co-folding and small-molecule binding affinity prediction',
        inputs: { sequences: { type: 'SequenceSet', required: true } },
        outputs: {
          folds: { type: 'FoldedStructure' },
          affinity: { type: 'PropertyTable' },
        },
      },
      execution: {
        runtime: 'wgsl-custom',
        fallback: { kind: 'none', reason: 'WebGPU' },
        createWorker: () => ({}) as Worker,
      },
      cachePolicy: { cacheStoragePrefixes: ['boltz2-'] },
      share: { shortCode: 'b2' },
      catalog: { categoryId: 'fold', conciseRole: 'Joint 3D co-folding + binding affinity' },
      display: { formatCanvasSummary: () => 'Boltz-2 (Fold + ΔG)' },
      preflight: () => ({ estimatedVramMB: 1800, estimatedDownloadMB: 450, workItems: 1 }),
      planner: {
        plannerRoles: ['predictor', 'scorer'],
        keywords: ['boltz2', 'affinity', 'ic50', 'kd'],
        capabilities: ['co-fold', 'binding-affinity'],
      },
    };

    reg.registerPlugin(boltz2Omni);
    reg.freeze();

    expect(reg.findPluginsByStage('fold').map((p) => p.id)).toContain('boltz2-affinity');
    expect(reg.findPluginsByStage('score').map((p) => p.id)).toContain('boltz2-affinity');
    expect(reg.findPluginsByStage('property-pred').map((p) => p.id)).toContain('boltz2-affinity');
    expect(reg.findPluginsByPlannerRole('predictor').map((p) => p.id)).toContain('boltz2-affinity');
    expect(reg.findPluginsByPlannerRole('scorer').map((p) => p.id)).toContain('boltz2-affinity');
  });
});
