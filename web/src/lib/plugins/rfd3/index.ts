// Layer 3 Native ModelPlugin — rfd3 (RFdiffusion3 All-Atom Diffusion · Baker Lab / Foundry)

import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { getLegacyModelCardForAdapter } from '$lib/engine/model-cards';
import { PROTEIN_DOMAIN_ID } from '$lib/domains/protein-biopolymer';
import type { KernelAdapterManifest, KernelModelCardSpec, ModelPlugin } from '$lib/kernel';

const manifest = ADAPTER_REGISTRY.rfd3 as unknown as KernelAdapterManifest;

export const rfd3Plugin: ModelPlugin = {
  id: 'rfd3',
  domainId: PROTEIN_DOMAIN_ID,
  manifest,
  execution: {
    runtime: 'wgsl-custom',
    fallback: { kind: 'none', reason: 'Requires WebGPU (or Host GPU server)' },
    createWorker: async () => new (await import('./rfd3.worker?worker')).default(),
    supportsCooperativeAbort: true,
    abortGraceMs: 1500,
    vramLeaseMode: 'exclusive',
  },
  cachePolicy: {
    cacheStorageNames: ['rfd3-v1'],
    opfsDirs: ['rfd3-v1'],
  },
  share: {
    shortCode: 'r3',
  },
  catalog: {
    categoryId: 'generate',
    conciseRole:
      'All-atom biomolecular diffusion (168M) for monomers, atomic/tip-atom motifs, binders, ligands, cyclic peptides & Cn symmetry',
    searchKeywords: [
      'rfd3',
      'rfdiffusion3',
      'rfdiffusion',
      'all-atom',
      'enzyme',
      'ligand',
      'motif',
      'binder',
      'cyclic',
      'symmetry',
      'unindexed',
      'tip-atom',
      'partial',
      'baker',
      'foundry',
      'hbond',
      'dna',
      'rna',
      'co-design',
    ],
    buildSubGroups: () => [
      {
        id: 'rfd3-unconditional',
        label: 'RFdiffusion3 · Unconditional All-Atom',
        description: 'De novo all-atom monomer & sequence co-design (Recommended default · ~337 MB)',
        paramsPatch: { mode: 'unconditional', precision: 'fp16' },
        nodeLabel: 'RFdiffusion3',
        approxMB: 337,
      },
      {
        id: 'rfd3-motif',
        label: 'RFdiffusion3 · Atomic / Tip-Atom Motif & Enzyme',
        description: 'Indexed or unindexed all-atom motif scaffolding with optional ligand & H-bond conditioning',
        paramsPatch: { mode: 'motif', precision: 'fp16' },
        nodeLabel: 'RFdiffusion3 (Motif)',
        approxMB: 337,
      },
      {
        id: 'rfd3-binder',
        label: 'RFdiffusion3 · PPI Binder (Hotspots)',
        description: 'All-atom de novo binder design against a connected target structure and hotspots',
        paramsPatch: { mode: 'binder', precision: 'fp16' },
        nodeLabel: 'RFdiffusion3 (Binder)',
        approxMB: 337,
      },
      {
        id: 'rfd3-partial',
        label: 'RFdiffusion3 · Partial Diffusion',
        description: 'Refine or diversify a connected input structure from noise level partialT',
        paramsPatch: { mode: 'partial', partialT: 5.0, precision: 'fp16' },
        nodeLabel: 'RFdiffusion3 (Partial)',
        approxMB: 337,
      },
      {
        id: 'rfd3-cyclic',
        label: 'RFdiffusion3 · Cyclic Peptide',
        description: 'Head-to-tail de novo cyclic peptide co-design (16–40 aa)',
        paramsPatch: { mode: 'unconditional', cyclic: true, length: 16, precision: 'fp16' },
        nodeLabel: 'RFdiffusion3 (Cyclic)',
        approxMB: 337,
      },
      {
        id: 'rfd3-symmetry',
        label: 'RFdiffusion3 · Symmetric Oligomer (C3)',
        description: 'All-atom Cn cyclic symmetric assembly generation (C2–C6)',
        paramsPatch: { mode: 'unconditional', symmetry: 'C3', length: 48, precision: 'fp16' },
        nodeLabel: 'RFdiffusion3 (C3)',
        approxMB: 337,
      },
    ],
  },
  modelCard: {
    resolveCard: (params) =>
      getLegacyModelCardForAdapter('rfd3', params) as unknown as KernelModelCardSpec,
  },
  display: {
    formatCanvasSummary: (params) => {
      const mode = String(params.mode ?? 'unconditional').trim().toLowerCase();
      const len = Number(params.length ?? 80);
      const steps = Number(params.steps ?? 50);
      const prec = String(params.precision ?? 'fp16').trim().toUpperCase();
      const n = Number(params.numDesigns ?? 1);
      const nTag = n > 1 ? ` · ×${n}` : '';
      const lig = String(params.ligand ?? '').trim();
      const ligTag = lig ? ` + ${lig}` : '';

      if (mode === 'binder') {
        const hotCount = String(params.hotspots ?? '')
          .split(/[,\s;]+/)
          .filter(Boolean).length;
        const hotTag = hotCount > 0 ? ` · ${hotCount} hot` : '';
        return `Binder · ${len} aa${hotTag}${ligTag} · ${steps} steps${nTag}`;
      }
      if (mode === 'motif') {
        const unidx = String(params.unindex ?? '').trim();
        const contig = String(params.contig ?? '').trim();
        if (unidx) {
          return `Unindexed (${unidx}) · ${len} aa${ligTag} · ${steps} steps${nTag}`;
        }
        return `Motif${contig ? ` (${contig})` : ` · ${len} aa`}${ligTag} · ${steps} steps${nTag}`;
      }
      if (mode === 'partial') {
        const pt = Number(params.partialT ?? 5.0);
        return `Partial (σ=${Number.isFinite(pt) ? pt : 5.0})${ligTag} · ${steps} steps${nTag}`;
      }
      const sym = String(params.symmetry ?? 'none').trim().toUpperCase();
      if (sym && sym !== 'NONE') {
        return `${sym} · ${len} aa/u · ${steps} steps · ${prec}${nTag}`;
      }
      if (params.cyclic) {
        return `Cyclic · ${len} aa · ${steps} steps · ${prec}${nTag}`;
      }
      return `${len} aa · ${steps} steps · ${prec}${nTag}`;
    },
  },
  preflight: (ctx) => {
    const prec = String(ctx.params.precision ?? 'fp16').trim().toLowerCase() === 'int8' ? 'int8' : 'fp16';
    const n = Math.max(1, Math.min(8, Math.round(Number(ctx.params.numDesigns ?? 1) || 1)));
    const fp16MB = manifest.precisionWeightsMB?.fp16 ?? manifest.approxWeightsMB ?? 337;
    const int8MB = manifest.precisionWeightsMB?.int8 ?? 170;
    const fp16Vram = manifest.estimatedVramMB ?? 1400;
    const int8Vram = 1100;
    return {
      estimatedVramMB: prec === 'int8' ? int8Vram : fp16Vram,
      estimatedDownloadMB: prec === 'int8' ? int8MB : fp16MB,
      workItems: n,
      outputCardinality: { structure: n, sequences: n },
      stageBreakdown: { generate: n },
      precisionOptions: {
        fp16: { weightsMB: fp16MB, vramMB: fp16Vram },
        int8: { weightsMB: int8MB, vramMB: int8Vram },
      },
    };
  },
};
