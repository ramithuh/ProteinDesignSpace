// Layer 3 Native ModelPlugin — gnina (GNINA 1.3 Ligand Docking & Affinity)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { resolveGninaLigandSource } from '$lib/protein-ops/gnina-prep';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.gnina as unknown as KernelAdapterManifest;

export const gninaPlugin: ModelPlugin = {
  id: 'gnina',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./gnina.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['gnina-v1'],
    opfsDirs: ['gnina-v1'],
  },
  share: {
    shortCode: 'gn',
  },
  catalog: {
    categoryId: 'score',
    conciseRole:
      '3D CNN protein–ligand docking score, pocket attribution & pose minimization (GNINA 1.3)',
    searchKeywords: [
      'gnina',
      'gnina1.3',
      'docking',
      'dock',
      'ligand',
      'affinity',
      'cnnscore',
      'cnnaffinity',
      'cnn_vs',
      'vina',
      'pocket',
      'minimize',
      'small molecule',
    ],
    buildSubGroups: () => [
      {
        id: 'gnina-dock',
        label: 'GNINA 1.3 · Dock Ligand (Flexible)',
        description:
          'Flexible-torsion MC + BFGS pocket docking with 3-model WebGPU CNN rescoring (~1–3 s)',
        paramsPatch: { taskMode: 'dock', precision: 'fp16', exhaustiveness: 8, numModes: 5 },
        nodeLabel: 'GNINA 1.3 (Dock)',
        approxMB: 4,
      },
      {
        id: 'gnina-score',
        label: 'GNINA 1.3 · Score Pose',
        description: '3-model 3D CNN ensemble (CNNscore, CNNaffinity pK, CNN_VS) + Vina (~65 ms)',
        paramsPatch: { taskMode: 'score', precision: 'fp16' },
        nodeLabel: 'GNINA 1.3',
        approxMB: 4,
      },
      {
        id: 'gnina-pocket-scan',
        label: 'GNINA 1.3 · Pocket Contribution Scan',
        description:
          'Per-residue CNN ablation (ΔCNN_VS) painted onto 3D binding pocket & residue track',
        paramsPatch: { taskMode: 'pocket-scan', precision: 'fp16' },
        nodeLabel: 'GNINA 1.3 (Pocket Scan)',
        approxMB: 4,
      },
      {
        id: 'gnina-minimize',
        label: 'GNINA 1.3 · Minimize Pose',
        description: 'Rigid-body pose refinement minimizing Vina − w·log(CNNscore)',
        paramsPatch: { taskMode: 'minimize', precision: 'fp16' },
        nodeLabel: 'GNINA 1.3 (Minimize)',
        approxMB: 4,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('gnina', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const { taskMode: tm, ligandCode, dockLigandCcd: ccd, source } = resolveGninaLigandSource(params);
      const prec = String(params.precision ?? 'fp16').toUpperCase();
      const lig = ligandCode.toUpperCase();
      const ligTag = source === 'external' ? ' · SDF' : lig ? ` · ${lig}` : '';
      if (tm === 'dock') {
        const exh = Number(params.exhaustiveness ?? 8);
        const modes = Number(params.numModes ?? 5);
        const dockLigTag = ccd ? ` · ${ccd}` : ligTag;
        return `Dock (exh=${exh}, ${modes}p)${dockLigTag} · ${prec}`;
      }
      if (tm === 'pocket-scan') {
        const rad = Number(params.pocketRadius ?? 5.0);
        return `Pocket Scan (${rad} Å)${ligTag} · ${prec}`;
      }
      if (tm === 'minimize') {
        const iters = Number(params.minimizeIterations ?? 4);
        return `Minimize (${iters} iters)${ligTag} · ${prec}`;
      }
      return `Score Pose${ligTag} · ${prec}`;
    },
    resolveSlots: (params) => {
      const mode = String(params.taskMode ?? 'score');
      return [
        {
          portName: 'structure',
          direction: 'out',
          availability: 'active',
          viewportRole: 'receptor',
        },
        {
          portName: 'track',
          direction: 'out',
          availability: mode === 'pocket-scan' ? 'active' : 'inactive-in-mode',
          ...(mode !== 'pocket-scan'
            ? {
                inactiveReason:
                  'Switch GNINA mode to Pocket Contribution Scan to compute per-residue ΔCNN_VS attribution.',
              }
            : {}),
        },
      ];
    },
  },
  preflight: (ctx) => {
    const precision = String(ctx.params.precision ?? 'fp16').toLowerCase();
    const approxMB = manifest.precisionWeightsMB?.[precision] ?? manifest.approxWeightsMB;
    const upBackbones = ctx.upstreamProfiles.structure?.itemCount ?? 1;
    const mode = String(ctx.params.taskMode ?? 'score');
    return {
      estimatedVramMB: manifest.estimatedVramMB,
      estimatedDownloadMB: approxMB,
      workItems: upBackbones,
      outputCardinality: {
        structure: upBackbones,
        ...(mode === 'pocket-scan' ? { track: 1 } : {}),
      },
      stageBreakdown: { score: upBackbones },
    };
  },
};
