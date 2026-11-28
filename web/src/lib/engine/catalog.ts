// Pure-TypeScript catalog grouping, multi-variant flyout sub-items, and search filtering
// for the `/` and `+ Add Block` workflow block picker.

import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_VARIANT_OPTIONS,
} from './registry';
import {
  AME_BENCHMARK_TARGETS,
  BENCHMARK_TARGETS,
} from './benchmarks';
import { DEFAULT_ALPHAFOLD_DB_UNIPROT_ID, QUICK_ALPHAFOLD_DB_EXAMPLES } from '$lib/protein-ops/alphafold-db';
import { PORT_COLORS } from './graph-utils';

export interface CatalogInsertRequest {
  adapterId: string;
  params?: Record<string, unknown>;
  label?: string;
}

export interface CatalogSubItem {
  key: string;
  label: string;
  subtitle?: string;
  meta?: string;
  badge?: string;
  nonCommercial?: boolean;
  insert: CatalogInsertRequest;
  searchBlob: string;
}

export interface CatalogSubGroup {
  title: string;
  items: CatalogSubItem[];
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  /** Render as a compact nested hover/click flyout submenu when not actively searching. */
  nestedFlyout?: 'ame41' | 'benchmarks';
}

export interface CatalogItem {
  id: string;
  group: string;
  shortName: string;
  displayName: string;
  stage: string;
  role: string;
  subCountLabel?: string;
  subGroups: CatalogSubGroup[];
  identityBlob: string;
  ownSearchBlob: string;
  searchBlob: string;
}

export interface FilteredCatalogEntry {
  item: CatalogItem;
  directParentMatch: boolean;
  matchingSubGroups: CatalogSubGroup[];
  matchingSubItems: CatalogSubItem[];
  hasNarrowedSubMatch: boolean;
}

export const CATALOG_STAGE_GROUPS: { title: string; ids: string[] }[] = [
  { title: 'Inputs', ids: ['pdb-input', 'sequence-input'] },
  {
    title: 'Structure Generation',
    ids: [
      'rfd4-proteina',
      'rfd3',
      'la-proteina',
      'proteina',
      'sid-protein',
      'genie3',
      'genie2',
      'rfdiffusion',
      'frameflow',
      'foldflow2',
      'boltzgen',
      'artios-backbone',
      'protpardelle',
    ],
  },
  {
    title: 'Sequence Design',
    ids: ['web-protein-mpnn', 'ligandmpnn', 'lasermpnn', 'artios-potts', 'proton-pottsmpnn', 'antifold-esmif1', 'evodiff'],
  },
  { title: 'Structure Prediction', ids: ['localfold-af2'] },
  { title: 'Refinement / Backmapping', ids: ['pulchra', 'cg2all'] },
  { title: 'Property Prediction', ids: ['thermompnn', 'esmc', 'dyna1', 'gnina'] },
  { title: 'Evaluation', ids: ['top-k-filter', 'designability-scorer', 'structure-compare'] },
];

export const CONCISE_BLOCK_ROLES: Record<string, string> = {
  'pdb-input': 'Fetch by PDB ID or AlphaFold DB (UniProt), upload/paste PDB & mmCIF, or 3-helix demo',
  'sequence-input': 'Paste amino acid sequences (plain text or FASTA)',
  'rfd4-proteina': 'All-atom backbone and sequence co-design',
  rfd3: 'All-atom biomolecular diffusion (168M) for monomers, atomic/tip-atom motifs, binders, ligands, cyclic peptides & Cn symmetry',
  'la-proteina': 'All-atom latent diffusion (LD1–LD7 + AE1/AE3) for de novo co-design & motif scaffolding',
  proteina: 'Scalable 200M transformer flow matching for de novo Cα backbones',
  'sid-protein': 'Distilled 16-step Proteina sampler for rapid Cα backbone generation',
  genie3: 'Fast de novo backbone diffusion (23.3M parameters)',
  genie2: 'Compact 15.7M Cα diffusion with optional soft motif scaffolding',
  rfdiffusion: 'Original RFdiffusion backbone diffusion, motif contigs & Cn/Dn symmetry',
  frameflow: 'Compact 16.8M rigid-frame flow matching for 5-atom backbones',
  foldflow2: 'ReFT rigid-frame flow matching with frozen ESM-2 650M encoder',
  boltzgen: 'BoltzGen-1 all-atom unconditional, binder & motif diffusion',
  'artios-backbone': 'All-atom EDM diffusion (136.7M) for de novo backbones, motif scaffolding & ligand-conditioned design',
  protpardelle: 'All-atom & backbone EDM diffusion, motif scaffolding & MiniMPNN co-design',
  'web-protein-mpnn': 'Inverse folding (SolubleMPNN, ProteinMPNN, Cα-only)',
  ligandmpnn: 'Ligand-, metal- & nucleic-acid-aware protein sequence design',
  lasermpnn: 'Ligand-aware sequence & all-atom sidechain rotamer co-design',
  'artios-potts': 'All-atom ligand-aware Potts sequence design & scoring (energy + per-residue pseudo-log-likelihood)',
  'proton-pottsmpnn': 'pH-aware Potts sequence design & scoring over explicit protonation states (pH-selective block descent, E_P − E_D)',
  'antifold-esmif1': 'Antibody CDR (AntiFold) & general protein inverse folding (ESM-IF1)',
  evodiff: 'MSA-free discrete sequence diffusion & masked sequence inpainting (OA-DM 38M)',
  'localfold-af2': 'Predict 3D structures (AlphaFold2, RF3, Boltz-2, AlphaFold 3 & more)',
  thermompnn: 'Full L×20 site-saturation ΔΔG stability scan & B-factor painting',
  esmc: 'ESM-C 300M zero-shot L×20 LLR fitness heatmap, naturalness & masked fill',
  dyna1: 'Millisecond conformational exchange & flexibility prediction (ESM-3 + Dyna-1)',
  gnina: '3D CNN protein–ligand docking score, pocket attribution & pose minimization (GNINA 1.3)',
  'top-k-filter': 'Rank and filter top-K sequences per backbone or globally before folding',
  'designability-scorer': 'Kabsch alignment, scRMSD, scTM & pLDDT pass rate',
  'structure-compare': 'Multi-model 3D Kabsch / motif overlay & pairwise RMSD',
  pulchra: 'Cα / backbone trace → all-atom model (PULCHRA 3.06 backbone, rotamers & H-bonds)',
  cg2all: 'Neural Cα trace → all-atom model (cg2all equivariant transformer, rigid-group side chains)',
};

function buildSubGroupsForAdapter(adapterId: string): {
  subCountLabel?: string;
  subGroups: CatalogSubGroup[];
} {
  if (adapterId === 'localfold-af2') {
    const af2Items: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2').map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: opt.shortLabel,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.shortLabel,
        opt.license,
        'af2',
        'af2-class',
        'alphafold2',
        'monomer',
      ]
        .join(' ')
        .toLowerCase(),
    }));

    const af2MultimerItems: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter(
      (m) => m.family === 'af2-multimer',
    ).map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: opt.shortLabel,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.shortLabel,
        opt.license,
        'af2',
        'af2-multimer',
        'alphafold2',
        'multimer',
        'complex',
        'binder',
        'ppi',
      ]
        .join(' ')
        .toLowerCase(),
    }));

    const af3Items: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3').map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      badge: opt.nonCommercial ? 'Non-Commercial' : undefined,
      nonCommercial: opt.nonCommercial,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: opt.shortLabel,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.shortLabel,
        opt.license,
        'af3',
        'af3-class',
        'alphafold3',
        'all-atom',
        opt.nonCommercial ? 'non-commercial' : '',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase(),
    }));

    const esmfold2Items: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter(
      (m) => m.family === 'esmfold2',
    ).map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      insert: { adapterId: 'localfold-af2', params: { model: opt.id, msaMode: 'single' }, label: opt.shortLabel },
      searchBlob: `${opt.id} ${opt.label} ${opt.shortLabel} esmfold2 esm-c protein multimer single sequence`.toLowerCase(),
    }));

    return {
      subCountLabel: `${LOCALFOLD_MODEL_OPTIONS.length} models`,
      subGroups: [
        { title: 'AF2-Class (Monomer)', items: af2Items },
        { title: 'AF2-Multimer v3 (Complexes)', items: af2MultimerItems },
        { title: 'AF3-Class (All-Atom)', items: af3Items },
        { title: 'ESMFold2 Experimental Fast', items: esmfold2Items },
      ],
    };
  }

  if (adapterId === 'web-protein-mpnn') {
    const items: CatalogSubItem[] = MPNN_VARIANT_OPTIONS.map((v) => ({
      key: v.id,
      label: v.label,
      subtitle: v.description,
      meta: `~${v.approxMB} MB · ${v.license}`,
      insert: {
        adapterId: 'web-protein-mpnn',
        params: { family: v.family, modelName: v.modelName },
        label: v.shortLabel,
      },
      searchBlob: [
        v.id,
        v.family,
        v.label,
        v.shortLabel,
        v.description,
        v.license,
        v.family === 'ca' ? 'ca alpha carbon backbone trace' : '',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase(),
    }));

    return {
      subCountLabel: `${items.length} variants`,
      subGroups: [{ title: 'Model Variants', items }],
    };
  }

  if (adapterId === 'antifold-esmif1') {
    const items: CatalogSubItem[] = [
      {
        key: 'antifold',
        label: 'AntiFold (Antibody CDRs)',
        subtitle: 'Antibody & nanobody CDR inverse folding (IMGT/Kabat regions)',
        meta: '~271 MB · BSD-3-Clause',
        insert: {
          adapterId: 'antifold-esmif1',
          params: { variant: 'antifold' },
          label: 'AntiFold',
        },
        searchBlob:
          'antifold antibody nanobody vhh cdr cdr1 cdr2 cdr3 imgt kabat inverse folding opig bsd-3'.toLowerCase(),
      },
      {
        key: 'esmif1',
        label: 'ESM-IF1',
        subtitle: 'General GVP-Transformer inverse folding (Meta AI · 142M params)',
        meta: '~271 MB · MIT',
        insert: {
          adapterId: 'antifold-esmif1',
          params: { variant: 'esmif1' },
          label: 'ESM-IF1',
        },
        searchBlob:
          'esmif1 esm-if1 esm gvp transformer meta general inverse folding sequence design mit'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} variants`,
      subGroups: [{ title: 'Model Variants', items }],
    };
  }

  if (adapterId === 'rfd3') {
    const items: CatalogSubItem[] = [
      {
        key: 'rfd3-unconditional',
        label: 'RFdiffusion3 · Unconditional All-Atom',
        subtitle: 'De novo all-atom monomer & sequence co-design (Recommended default · ~337 MB)',
        meta: '~337 MB · BSD-3-Clause',
        insert: {
          adapterId: 'rfd3',
          params: { mode: 'unconditional', precision: 'fp16' },
          label: 'RFdiffusion3',
        },
        searchBlob:
          'rfd3 rfdiffusion3 rfdiffusion all-atom unconditional de novo monomer baker foundry co-design'.toLowerCase(),
      },
      {
        key: 'rfd3-motif',
        label: 'RFdiffusion3 · Atomic / Tip-Atom Motif & Enzyme',
        subtitle: 'Indexed or unindexed all-atom motif scaffolding with ligand & H-bond conditioning',
        meta: '~337 MB · BSD-3-Clause',
        insert: {
          adapterId: 'rfd3',
          params: { mode: 'motif', precision: 'fp16' },
          label: 'RFdiffusion3 (Motif)',
        },
        searchBlob:
          'rfd3 rfdiffusion3 motif scaffolding enzyme active site tip-atom unindexed ligand hbond all-atom baker'.toLowerCase(),
      },
      {
        key: 'rfd3-binder',
        label: 'RFdiffusion3 · PPI Binder (Hotspots)',
        subtitle: 'All-atom de novo binder design against a connected target structure and hotspots',
        meta: '~337 MB · BSD-3-Clause',
        insert: {
          adapterId: 'rfd3',
          params: { mode: 'binder', precision: 'fp16' },
          label: 'RFdiffusion3 (Binder)',
        },
        searchBlob: 'rfd3 rfdiffusion3 binder ppi hotspots target all-atom baker'.toLowerCase(),
      },
      {
        key: 'rfd3-partial',
        label: 'RFdiffusion3 · Partial Diffusion',
        subtitle: 'Refine or diversify a connected input structure from noise level partialT',
        meta: '~337 MB · BSD-3-Clause',
        insert: {
          adapterId: 'rfd3',
          params: { mode: 'partial', partialT: 5.0, precision: 'fp16' },
          label: 'RFdiffusion3 (Partial)',
        },
        searchBlob: 'rfd3 rfdiffusion3 partial diffusion refine diversify all-atom baker'.toLowerCase(),
      },
      {
        key: 'rfd3-cyclic',
        label: 'RFdiffusion3 · Cyclic Peptide',
        subtitle: 'Head-to-tail de novo cyclic peptide co-design (16–40 aa)',
        meta: '~337 MB · BSD-3-Clause',
        insert: {
          adapterId: 'rfd3',
          params: { mode: 'unconditional', cyclic: true, length: 16, precision: 'fp16' },
          label: 'RFdiffusion3 (Cyclic)',
        },
        searchBlob: 'rfd3 rfdiffusion3 cyclic peptide macrocycle head-to-tail all-atom baker'.toLowerCase(),
      },
      {
        key: 'rfd3-symmetry',
        label: 'RFdiffusion3 · Symmetric Oligomer (C3)',
        subtitle: 'All-atom Cn cyclic symmetric assembly generation (C2–C6)',
        meta: '~337 MB · BSD-3-Clause',
        insert: {
          adapterId: 'rfd3',
          params: { mode: 'unconditional', symmetry: 'C3', length: 48, precision: 'fp16' },
          label: 'RFdiffusion3 (C3)',
        },
        searchBlob: 'rfd3 rfdiffusion3 symmetry cyclic oligomer c2 c3 c4 c6 all-atom baker'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} modes`,
      subGroups: [{ title: 'Design Modes', items }],
    };
  }

  if (adapterId === 'rfdiffusion') {
    const items: CatalogSubItem[] = [
      {
        key: 'rfd-unconditional',
        label: 'RFdiffusion · Unconditional De Novo',
        subtitle: 'Original rigid-frame backbone diffusion (Base_ckpt.pt · ~69 MB)',
        meta: '~69 MB · BSD',
        insert: {
          adapterId: 'rfdiffusion',
          params: { taskMode: 'unconditional', precision: 'int8' },
          label: 'RFdiffusion',
        },
        searchBlob: 'rfdiffusion rfd rfd1 base unconditional backbone se3 diffusion de novo'.toLowerCase(),
      },
      {
        key: 'rfd-motif',
        label: 'RFdiffusion · Motif Scaffolding (Contig)',
        subtitle: 'Contig-driven motif scaffolding from a connected template PDB',
        meta: '~69 MB · BSD',
        insert: {
          adapterId: 'rfdiffusion',
          params: { taskMode: 'motif', precision: 'int8' },
          label: 'RFdiffusion (Motif)',
        },
        searchBlob: 'rfdiffusion rfd motif scaffolding contig template'.toLowerCase(),
      },
      {
        key: 'rfd-symmetry',
        label: 'RFdiffusion · Symmetric Oligomer (C3)',
        subtitle: 'Cyclic / dihedral symmetric assembly generation (C2–C6, D2–D3)',
        meta: '~69 MB · BSD',
        insert: {
          adapterId: 'rfdiffusion',
          params: { taskMode: 'symmetry', symmetry: 'c3', length: 90, precision: 'int8' },
          label: 'RFdiffusion (C3)',
        },
        searchBlob: 'rfdiffusion rfd symmetry cyclic dihedral oligomer c2 c3 c4 c6 d2 d3'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} modes`,
      subGroups: [{ title: 'Design Modes', items }],
    };
  }

  if (adapterId === 'boltzgen') {
    const items: CatalogSubItem[] = [
      {
        key: 'bg-binder',
        label: 'BoltzGen-1 · Binder Design (Diverse)',
        subtitle: 'All-atom de novo binder generation against a connected target PDB',
        meta: '~482 MB · MIT',
        insert: {
          adapterId: 'boltzgen',
          params: { taskMode: 'binder', variant: 'diverse', precision: 'int8' },
          label: 'BoltzGen (Binder)',
        },
        searchBlob: 'boltzgen boltzgen1 binder design all-atom target ppi diverse'.toLowerCase(),
      },
      {
        key: 'bg-unconditional',
        label: 'BoltzGen-1 · Unconditional De Novo',
        subtitle: 'All-atom de novo monomer generation (boltzgen1_diverse)',
        meta: '~482 MB · MIT',
        insert: {
          adapterId: 'boltzgen',
          params: { taskMode: 'unconditional', variant: 'diverse', precision: 'int8' },
          label: 'BoltzGen-1',
        },
        searchBlob: 'boltzgen boltzgen1 unconditional de novo all-atom diverse'.toLowerCase(),
      },
      {
        key: 'bg-motif',
        label: 'BoltzGen-1 · Motif Scaffolding (Adherence)',
        subtitle: 'High-adherence all-atom motif scaffolding inside a de novo scaffold',
        meta: '~482 MB · MIT',
        insert: {
          adapterId: 'boltzgen',
          params: { taskMode: 'motif', variant: 'adherence', precision: 'int8' },
          label: 'BoltzGen (Motif)',
        },
        searchBlob: 'boltzgen boltzgen1 motif scaffolding adherence all-atom'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} presets`,
      subGroups: [{ title: 'Design Modes', items }],
    };
  }

  if (adapterId === 'evodiff') {
    const items: CatalogSubItem[] = [
      {
        key: 'evodiff-unconditional',
        label: 'EvoDiff-Seq · Unconditional De Novo (38M)',
        subtitle: 'MSA-free order-agnostic discrete sequence diffusion (~37 MB)',
        meta: '~37 MB · MIT / CC-BY-4.0',
        insert: {
          adapterId: 'evodiff',
          params: { taskMode: 'unconditional', precision: 'int8' },
          label: 'EvoDiff-38M',
        },
        searchBlob: 'evodiff oadm 38m sequence diffusion de novo msa-free generator'.toLowerCase(),
      },
      {
        key: 'evodiff-infill',
        label: 'EvoDiff-Seq · Masked Sequence Inpainting',
        subtitle: 'Preserve sequence motif residues and diffuse # / masked scaffold spans',
        meta: '~37 MB · MIT / CC-BY-4.0',
        insert: {
          adapterId: 'evodiff',
          params: { taskMode: 'infill', precision: 'int8' },
          label: 'EvoDiff (Infill)',
        },
        searchBlob: 'evodiff oadm 38m infill inpainting mask motif sequence scaffolding'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} modes`,
      subGroups: [{ title: 'Sampling Modes', items }],
    };
  }

  if (adapterId === 'protpardelle') {
    const items: CatalogSubItem[] = [
      {
        key: 'cc91_epoch383',
        label: 'ProtPardelle-1c · All-Atom (cc91)',
        subtitle: 'All-atom monomer diffusion, motif scaffolding & MiniMPNN co-design',
        meta: '~26 MB · CC-BY-4.0',
        insert: {
          adapterId: 'protpardelle',
          params: { checkpoint: 'cc91_epoch383', sidechainMode: true },
          label: 'ProtPardelle (All-Atom)',
        },
        searchBlob: 'protpardelle 1c cc91 allatom all-atom monomer motif diffusion minimpnn codesign'.toLowerCase(),
      },
      {
        key: 'cc91_tip_epoch480',
        label: 'ProtPardelle-1c · Tip-Atom Motif (cc91_tip)',
        subtitle: 'Sidechain-tip atom motif scaffolding & enzyme active-site framing',
        meta: '~26 MB · CC-BY-4.0',
        insert: {
          adapterId: 'protpardelle',
          params: { checkpoint: 'cc91_tip_epoch480', taskMode: 'motif', sidechainMode: true },
          label: 'ProtPardelle (Tip Motif)',
        },
        searchBlob: 'protpardelle cc91_tip tip atom sidechain motif scaffolding enzyme'.toLowerCase(),
      },
      {
        key: 'cc58_epoch416',
        label: 'ProtPardelle-1c · Backbone (cc58)',
        subtitle: 'Fast backbone diffusion & crop-conditioned motif scaffolding',
        meta: '~26 MB · CC-BY-4.0',
        insert: {
          adapterId: 'protpardelle',
          params: { checkpoint: 'cc58_epoch416', sidechainMode: false },
          label: 'ProtPardelle (Backbone)',
        },
        searchBlob: 'protpardelle cc58 backbone diffusion motif'.toLowerCase(),
      },
      {
        key: 'cc94_epoch3100',
        label: 'ProtPardelle-1c · Multichain All-Atom (cc94)',
        subtitle: 'Multichain all-atom complex generation & conditioning',
        meta: '~26 MB · CC-BY-4.0',
        insert: {
          adapterId: 'protpardelle',
          params: { checkpoint: 'cc94_epoch3100', sidechainMode: true },
          label: 'ProtPardelle (Multichain)',
        },
        searchBlob: 'protpardelle cc94 multichain complex all-atom'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} presets`,
      subGroups: [{ title: 'Checkpoints', items }],
    };
  }

  if (adapterId === 'esmc') {
    const items: CatalogSubItem[] = [
      {
        key: 'esmc-wildtype',
        label: 'ESM-C 300M · Wild-Type LLR (Fast)',
        subtitle: '1-pass L×20 zero-shot fitness heatmap & WT log-p track (~40 ms)',
        meta: '~335 MB · MIT',
        insert: {
          adapterId: 'esmc',
          params: { mode: 'wildtype', precision: 'int8' },
          label: 'ESM-C 300M (WT LLR)',
        },
        searchBlob: 'esmc esm-c esm cambrian 300m llr zero-shot fitness mutation wildtype plm'.toLowerCase(),
      },
      {
        key: 'esmc-masked',
        label: 'ESM-C 300M · Masked-Marginal & PPPL',
        subtitle: 'Full L-pass masked-marginal LLR matrix & pseudo-perplexity naturalness',
        meta: '~335 MB · MIT',
        insert: {
          adapterId: 'esmc',
          params: { mode: 'masked', precision: 'int8' },
          label: 'ESM-C 300M (Masked)',
        },
        searchBlob: 'esmc esm-c masked marginal naturalness pseudo-perplexity pppl llr'.toLowerCase(),
      },
      {
        key: 'esmc-fill',
        label: 'ESM-C 300M · Masked Sequence Fill',
        subtitle: 'Mask selected residue positions and predict top-1 amino acids',
        meta: '~335 MB · MIT',
        insert: {
          adapterId: 'esmc',
          params: { mode: 'fill', precision: 'int8' },
          label: 'ESM-C 300M (Fill)',
        },
        searchBlob: 'esmc esm-c fill mask sequence design infill plm'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} modes`,
      subGroups: [{ title: 'Inference Modes', items }],
    };
  }

  if (adapterId === 'dyna1') {
    const items: CatalogSubItem[] = [
      {
        key: 'dyna1-standard',
        label: 'Dyna-1 · Standard (Recommended)',
        subtitle: 'Millisecond conformational exchange probability (0..1 · fast local profile)',
        meta: '~844 MB · MIT',
        insert: {
          adapterId: 'dyna1',
          params: { precision: 'int8' },
          label: 'Dyna-1 (Dynamics)',
        },
        searchBlob: 'dyna1 dyna-1 dynamics flexibility nmr exchange esm3 esm-3 conformational rigid flexible standard'.toLowerCase(),
      },
      {
        key: 'dyna1-high-precision',
        label: 'Dyna-1 · High Precision',
        subtitle: 'Higher-precision exchange probability profile (for high-memory GPUs)',
        meta: '~1.7 GB · MIT',
        insert: {
          adapterId: 'dyna1',
          params: { precision: 'fp16' },
          label: 'Dyna-1 (High Precision)',
        },
        searchBlob: 'dyna1 dyna-1 dynamics flexibility nmr exchange esm3 high precision'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} presets`,
      subGroups: [{ title: 'Presets', items }],
    };
  }

  if (adapterId === 'gnina') {
    const items: CatalogSubItem[] = [
      {
        key: 'gnina-dock',
        label: 'GNINA 1.3 · Dock Ligand (Flexible)',
        subtitle: 'Flexible-torsion MC + BFGS pocket docking with 3-model WebGPU CNN rescoring (~1–3 s)',
        meta: '~4 MB · Apache-2.0 / GPLv2',
        insert: {
          adapterId: 'gnina',
          params: { taskMode: 'dock', precision: 'fp16', exhaustiveness: 8, numModes: 5 },
          label: 'GNINA 1.3 (Dock)',
        },
        searchBlob:
          'gnina gnina1.3 dock docking flexible torsion monte carlo bfgs vina cnn rescore ligand pocket small molecule'.toLowerCase(),
      },
      {
        key: 'gnina-score',
        label: 'GNINA 1.3 · Score Pose',
        subtitle: '3-model 3D CNN ensemble (CNNscore, CNNaffinity pK, CNN_VS) + Vina (~65 ms)',
        meta: '~4 MB · Apache-2.0 / GPLv2',
        insert: {
          adapterId: 'gnina',
          params: { taskMode: 'score', precision: 'fp16' },
          label: 'GNINA 1.3',
        },
        searchBlob:
          'gnina gnina1.3 docking ligand affinity cnnscore cnnaffinity cnn_vs vina small molecule holo score pose'.toLowerCase(),
      },
      {
        key: 'gnina-pocket-scan',
        label: 'GNINA 1.3 · Pocket Contribution Scan',
        subtitle: 'Per-residue CNN ablation (ΔCNN_VS) painted onto 3D binding pocket & residue track',
        meta: '~4 MB · Apache-2.0 / GPLv2',
        insert: {
          adapterId: 'gnina',
          params: { taskMode: 'pocket-scan', precision: 'fp16' },
          label: 'GNINA 1.3 (Pocket Scan)',
        },
        searchBlob:
          'gnina pocket scan ablation contribution delta cnn_vs binding site hotspot residue attribution'.toLowerCase(),
      },
      {
        key: 'gnina-minimize',
        label: 'GNINA 1.3 · Minimize Pose',
        subtitle: 'Rigid-body pose refinement minimizing Vina − w·log(CNNscore)',
        meta: '~4 MB · Apache-2.0 / GPLv2',
        insert: {
          adapterId: 'gnina',
          params: { taskMode: 'minimize', precision: 'fp16' },
          label: 'GNINA 1.3 (Minimize)',
        },
        searchBlob:
          'gnina minimize refine optimize ligand pose rigid body docking vina cnn'.toLowerCase(),
      },
    ];
    return {
      subCountLabel: `${items.length} modes`,
      subGroups: [{ title: 'Modes', items }],
    };
  }

  return { subGroups: [] };
}

export const CATALOG_ITEMS: CatalogItem[] = CATALOG_STAGE_GROUPS.flatMap((grp) =>
  grp.ids
    .map((id): CatalogItem | null => {
      const m = ADAPTER_REGISTRY[id];
      if (!m) return null;

      const portTokens = [
        ...Object.values(m.inputs),
        ...Object.values(m.outputs),
      ].flatMap((spec) => {
        const pc = PORT_COLORS[spec.type];
        return pc ? [pc.label, pc.short] : [];
      });

      const role = CONCISE_BLOCK_ROLES[id] ?? m.description;
      const { subCountLabel, subGroups } = buildSubGroupsForAdapter(id);

      const identityBlob = [
        m.shortName,
        m.displayName.replace(/\s*\([^)]*\)/g, ''),
        m.stage,
        grp.title,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      const ownSearchBlob = [
        m.id,
        m.displayName,
        m.shortName,
        m.stage,
        grp.title,
        m.description,
        role,
        ...portTokens,
      ]
        .join(' ')
        .toLowerCase();

      const subText = subGroups.flatMap((sg) => [sg.title, ...sg.items.map((s) => s.searchBlob)]).join(' ');
      const searchBlob = `${ownSearchBlob} ${subText}`.toLowerCase();

      return {
        id,
        group: grp.title,
        shortName: m.shortName,
        displayName: m.displayName,
        stage: m.stage,
        role,
        subCountLabel,
        subGroups,
        identityBlob,
        ownSearchBlob,
        searchBlob,
      };
    })
    .filter((x): x is CatalogItem => x !== null),
);

export interface CatalogCategory {
  id: string;
  title: string;
  stage: string;
  subtitle: string;
  countLabel: string;
  footnote?: string;
  subGroups: CatalogSubGroup[];
  identityBlob: string;
  searchBlob: string;
}

export interface FilteredCategoryEntry {
  category: CatalogCategory;
  matchingSubGroups: CatalogSubGroup[];
  matchingSubItems: CatalogSubItem[];
  hasNarrowedSubMatch: boolean;
}

export const CATALOG_CATEGORIES: CatalogCategory[] = [
  (() => {
    const pdb = ADAPTER_REGISTRY['pdb-input'];
    const seq = ADAPTER_REGISTRY['sequence-input'];
    const items: CatalogSubItem[] = [
      {
        key: 'pdb-input',
        label: pdb.shortName,
        subtitle: CONCISE_BLOCK_ROLES['pdb-input'],
        insert: { adapterId: 'pdb-input' },
        searchBlob: `${pdb.id} ${pdb.shortName} ${pdb.displayName} ${CONCISE_BLOCK_ROLES['pdb-input']} pdb mmcif structure upload template input`.toLowerCase(),
      },
      {
        key: 'alphafold-db',
        label: 'AlphaFold DB',
        subtitle: 'Fetch a public predicted structure from EMBL-EBI AlphaFold DB by UniProt accession',
        meta: 'Public · CC-BY-4.0',
        badge: 'UniProt',
        insert: {
          adapterId: 'pdb-input',
          params: {
            mode: 'alphafold-db',
            uniprotId: DEFAULT_ALPHAFOLD_DB_UNIPROT_ID,
            filename: `AF-${DEFAULT_ALPHAFOLD_DB_UNIPROT_ID}-F1.cif`,
            pdbText: '',
          },
          label: `AFDB: ${QUICK_ALPHAFOLD_DB_EXAMPLES[0].name} (${DEFAULT_ALPHAFOLD_DB_UNIPROT_ID})`,
        },
        searchBlob:
          'alphafold db afdb uniprot ebi deepmind predicted structure plddt p69905 p04637 q9nzq7 p00698 cc-by-4.0 input'.toLowerCase(),
      },
      {
        key: 'sequence-input',
        label: seq.shortName,
        subtitle: CONCISE_BLOCK_ROLES['sequence-input'],
        insert: { adapterId: 'sequence-input' },
        searchBlob: `${seq.id} ${seq.shortName} ${seq.displayName} ${CONCISE_BLOCK_ROLES['sequence-input']} fasta amino acid sequence input`.toLowerCase(),
      },
    ];
    const makeBenchmarkItem = (b: (typeof BENCHMARK_TARGETS)[number], extraTokens: string): CatalogSubItem => ({
      key: `bench-${b.id}`,
      label: b.shortLabel,
      subtitle: b.biology,
      meta: `${b.sourceFamily} · ~${b.approxTokens} tok`,
      badge: b.tier === 'fast' ? 'Fast' : 'Standard',
      insert: {
        adapterId: 'pdb-input',
        params: {
          mode: 'pdb-id',
          pdbId: b.pdbId,
          filename: `${b.pdbId}.cif`,
          pdbText: '',
          benchmarkId: b.id,
          ...(b.cropSegments ? { cropSegments: b.cropSegments } : {}),
        },
        label: `PDB: ${b.name} (${b.pdbId})`,
      },
      searchBlob: `${b.id} ${b.mcsaId ?? ''} ${b.name} ${b.shortLabel} ${b.pdbId} ${(b.ligands ?? []).join(' ')} ${b.biology} benchmark ${extraTokens}`.toLowerCase(),
    });

    const binderBenchItems: CatalogSubItem[] = BENCHMARK_TARGETS.filter(
      (b) => b.category === 'binder',
    ).map((b) => makeBenchmarkItem(b, 'ppi binder target'));

    const motifBenchItems: CatalogSubItem[] = BENCHMARK_TARGETS.filter(
      (b) => b.category === 'motif',
    ).map((b) => makeBenchmarkItem(b, 'motif scaffold'));

    const ligandDnaBenchItems: CatalogSubItem[] = BENCHMARK_TARGETS.filter(
      (b) => b.category === 'enzyme-ligand' && !b.id.startsWith('ame_'),
    ).map((b) => makeBenchmarkItem(b, 'ligand dna cofactor'));

    const ame41BenchItems: CatalogSubItem[] = AME_BENCHMARK_TARGETS.map((b) =>
      makeBenchmarkItem(b, 'ame enzyme mcsa catalytic tip-atom ligand'),
    );

    const subGroups: CatalogSubGroup[] = [
      { title: 'Starting Inputs', items },
      { title: 'PPI Binders', items: binderBenchItems, nestedFlyout: 'benchmarks' },
      { title: 'Motif Scaffolding', items: motifBenchItems, nestedFlyout: 'benchmarks' },
      { title: 'Ligands & DNA', items: ligandDnaBenchItems, nestedFlyout: 'benchmarks' },
      { title: 'AME 41 Benchmarks', items: ame41BenchItems, nestedFlyout: 'ame41' },
    ];
    const allItems = [...items, ...binderBenchItems, ...motifBenchItems, ...ligandDnaBenchItems, ...ame41BenchItems];
    const identityBlob = 'inputs input source pdb sequence fasta mmcif benchmark';
    const searchBlob = `${identityBlob} ${allItems.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'inputs',
      title: 'Inputs',
      stage: 'source',
      subtitle: 'PDB / mmCIF, FASTA & curated benchmark targets',
      countLabel: `${allItems.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const rfd4 = ADAPTER_REGISTRY['rfd4-proteina'];
    const rfd3 = ADAPTER_REGISTRY.rfd3;
    const lap = ADAPTER_REGISTRY['la-proteina'];
    const pro = ADAPTER_REGISTRY.proteina;
    const sid = ADAPTER_REGISTRY['sid-protein'];
    const g3 = ADAPTER_REGISTRY.genie3;
    const g2 = ADAPTER_REGISTRY.genie2;
    const rfd = ADAPTER_REGISTRY.rfdiffusion;
    const ff = ADAPTER_REGISTRY.frameflow;
    const ff2 = ADAPTER_REGISTRY.foldflow2;
    const bg = ADAPTER_REGISTRY.boltzgen;
    const ab = ADAPTER_REGISTRY['artios-backbone'];
    const pp = ADAPTER_REGISTRY.protpardelle;
    const items: CatalogSubItem[] = [
      {
        key: 'rfd4-proteina',
        label: rfd4.shortName,
        subtitle: CONCISE_BLOCK_ROLES['rfd4-proteina'],
        meta: `~${rfd4.approxWeightsMB} MB`,
        insert: { adapterId: 'rfd4-proteina' },
        searchBlob: `${rfd4.id} ${rfd4.shortName} ${rfd4.displayName} ${CONCISE_BLOCK_ROLES['rfd4-proteina']} rfdiffusion4 rfd4 proteina generate de novo backbone co-design`.toLowerCase(),
      },
      {
        key: 'rfd3',
        label: rfd3.shortName,
        subtitle: CONCISE_BLOCK_ROLES.rfd3,
        meta: `~${rfd3.approxWeightsMB} MB · BSD-3-Clause`,
        insert: { adapterId: 'rfd3', label: 'RFdiffusion3' },
        searchBlob: `${rfd3.id} ${rfd3.shortName} ${rfd3.displayName} ${CONCISE_BLOCK_ROLES.rfd3} rfd3 rfdiffusion3 all-atom enzyme ligand motif binder cyclic symmetry unindexed tip-atom partial baker foundry hbond dna rna`.toLowerCase(),
      },
      {
        key: 'la-proteina',
        label: lap.shortName,
        subtitle: CONCISE_BLOCK_ROLES['la-proteina'],
        meta: `~${lap.approxWeightsMB} MB · Apache-2.0 / NVIDIA`,
        insert: { adapterId: 'la-proteina', label: 'La-Proteina' },
        searchBlob: `${lap.id} ${lap.shortName} ${lap.displayName} ${CONCISE_BLOCK_ROLES['la-proteina']} laproteina ld1 ld4 ld5 ld6 ld7 ae1 ae3 motif scaffolding tip-atom unindexed latent all-atom co-design diffusion nvidia`.toLowerCase(),
      },
      {
        key: 'proteina',
        label: pro.shortName,
        subtitle: CONCISE_BLOCK_ROLES.proteina,
        meta: `~${pro.approxWeightsMB} MB · NVIDIA Research`,
        ...(pro.nonCommercial ? { badge: 'Non-Commercial', nonCommercial: true } : {}),
        insert: { adapterId: 'proteina', label: 'Proteina 200M' },
        searchBlob: `${pro.id} ${pro.shortName} ${pro.displayName} ${CONCISE_BLOCK_ROLES.proteina} proteina 200m v1.2 flow matching backbone nvidia${pro.nonCommercial ? ' non-commercial' : ''}`.toLowerCase(),
      },
      {
        key: 'sid-protein',
        label: sid.shortName,
        subtitle: CONCISE_BLOCK_ROLES['sid-protein'],
        meta: `~${sid.approxWeightsMB} MB · 16-step distilled`,
        ...(sid.nonCommercial ? { badge: 'Non-Commercial', nonCommercial: true } : {}),
        insert: { adapterId: 'sid-protein', label: 'SiD-Protein' },
        searchBlob: `${sid.id} ${sid.shortName} ${sid.displayName} ${CONCISE_BLOCK_ROLES['sid-protein']} sid spid distilled 16-step fast proteina backbone${sid.nonCommercial ? ' non-commercial' : ''}`.toLowerCase(),
      },
      {
        key: 'genie3',
        label: g3.shortName,
        subtitle: CONCISE_BLOCK_ROLES.genie3,
        meta: `~${g3.approxWeightsMB} MB · Apache-2.0`,
        insert: { adapterId: 'genie3' },
        searchBlob: `${g3.id} ${g3.shortName} ${g3.displayName} ${CONCISE_BLOCK_ROLES.genie3} genie genie3 diffusion generate de novo backbone`.toLowerCase(),
      },
      {
        key: 'genie2',
        label: g2.shortName,
        subtitle: CONCISE_BLOCK_ROLES.genie2,
        meta: `~${g2.approxWeightsMB} MB · Apache-2.0`,
        insert: { adapterId: 'genie2', label: 'Genie 2' },
        searchBlob: `${g2.id} ${g2.shortName} ${g2.displayName} ${CONCISE_BLOCK_ROLES.genie2} genie genie2 motif scaffolding diffusion de novo backbone apache`.toLowerCase(),
      },
      {
        key: 'rfdiffusion',
        label: rfd.shortName,
        subtitle: CONCISE_BLOCK_ROLES.rfdiffusion,
        meta: `~${rfd.approxWeightsMB} MB · BSD`,
        insert: { adapterId: 'rfdiffusion', label: 'RFdiffusion' },
        searchBlob: `${rfd.id} ${rfd.shortName} ${rfd.displayName} ${CONCISE_BLOCK_ROLES.rfdiffusion} rfdiffusion rfd base motif contig symmetry cyclic dihedral backbone diffusion bsd`.toLowerCase(),
      },
      {
        key: 'frameflow',
        label: ff.shortName,
        subtitle: CONCISE_BLOCK_ROLES.frameflow,
        meta: `~${ff.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'frameflow', label: 'FrameFlow' },
        searchBlob: `${ff.id} ${ff.shortName} ${ff.displayName} ${CONCISE_BLOCK_ROLES.frameflow} frameflow flow matching rigid frame backbone microsoft mit`.toLowerCase(),
      },
      {
        key: 'foldflow2',
        label: ff2.shortName,
        subtitle: CONCISE_BLOCK_ROLES.foldflow2,
        meta: `~${ff2.approxWeightsMB} MB · CC BY-NC 4.0`,
        ...(ff2.nonCommercial ? { badge: 'Non-Commercial', nonCommercial: true } : {}),
        insert: { adapterId: 'foldflow2', label: 'FoldFlow 2' },
        searchBlob: `${ff2.id} ${ff2.shortName} ${ff2.displayName} ${CONCISE_BLOCK_ROLES.foldflow2} foldflow foldflow2 reft esm2 flow matching sequence conditioned dreamfold${ff2.nonCommercial ? ' non-commercial' : ''}`.toLowerCase(),
      },
      {
        key: 'boltzgen',
        label: bg.shortName,
        subtitle: CONCISE_BLOCK_ROLES.boltzgen,
        meta: `~${bg.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'boltzgen', label: 'BoltzGen-1' },
        searchBlob: `${bg.id} ${bg.shortName} ${bg.displayName} ${CONCISE_BLOCK_ROLES.boltzgen} boltzgen boltzgen1 all-atom binder motif diverse adherence diffusion mit`.toLowerCase(),
      },
      {
        key: 'artios-backbone',
        label: ab.shortName,
        subtitle: CONCISE_BLOCK_ROLES['artios-backbone'],
        meta: `~${ab.approxWeightsMB} MB · Apache-2.0`,
        insert: { adapterId: 'artios-backbone', label: 'ArtiosBackbone' },
        searchBlob: `${ab.id} ${ab.shortName} ${ab.displayName} ${CONCISE_BLOCK_ROLES['artios-backbone']} artios artiosbackbone ligo biosciences all-atom edm diffusion dit motif scaffolding ligand enzyme rasa burial de novo backbone karras self-conditioning apache`.toLowerCase(),
      },
      {
        key: 'protpardelle',
        label: pp.shortName,
        subtitle: CONCISE_BLOCK_ROLES.protpardelle,
        meta: `~${pp.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'protpardelle', label: 'ProtPardelle-1c' },
        searchBlob: `${pp.id} ${pp.shortName} ${pp.displayName} ${CONCISE_BLOCK_ROLES.protpardelle} protpardelle edm all-atom motif partial minimpnn diffusion mit`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [{ title: 'De Novo Generators', items }];
    const identityBlob = 'structure generation generate generator de novo backbone diffusion flow matching';
    const searchBlob = `${identityBlob} ${items.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'generation',
      title: 'Structure Generation',
      stage: 'generate',
      subtitle: 'RFD4-Proteina, RFdiffusion3, La-Proteina, Proteina, SiD-Protein, Genie 2/3, RFdiffusion, FrameFlow, FoldFlow 2, BoltzGen-1 & ProtPardelle-1c',
      countLabel: `${items.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const mpnn = buildSubGroupsForAdapter('web-protein-mpnn');
    const antifold = buildSubGroupsForAdapter('antifold-esmif1');
    const lig = ADAPTER_REGISTRY.ligandmpnn;
    const laser = ADAPTER_REGISTRY.lasermpnn;
    const potts = ADAPTER_REGISTRY['artios-potts'];
    const proton = ADAPTER_REGISTRY['proton-pottsmpnn'];
    const evo = ADAPTER_REGISTRY.evodiff;
    const allAtomItems: CatalogSubItem[] = [
      {
        key: 'ligandmpnn',
        label: lig.shortName,
        subtitle: CONCISE_BLOCK_ROLES.ligandmpnn,
        meta: `~${lig.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'ligandmpnn', label: 'LigandMPNN' },
        searchBlob:
          `${lig.id} ${lig.shortName} ${lig.displayName} ${CONCISE_BLOCK_ROLES.ligandmpnn} ligand metal ion dna rna holo pocket mpnn mit`.toLowerCase(),
      },
      {
        key: 'lasermpnn',
        label: laser.shortName,
        subtitle: CONCISE_BLOCK_ROLES.lasermpnn,
        meta: `~${laser.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'lasermpnn', label: 'LASErMPNN' },
        searchBlob:
          `${laser.id} ${laser.shortName} ${laser.displayName} ${CONCISE_BLOCK_ROLES.lasermpnn} ligand rotamer sidechain packing all-atom spice polizzi mpnn mit`.toLowerCase(),
      },
      {
        key: 'artios-potts',
        label: potts.shortName,
        subtitle: CONCISE_BLOCK_ROLES['artios-potts'],
        meta: `~${potts.approxWeightsMB} MB · Apache-2.0`,
        insert: { adapterId: 'artios-potts', label: 'ArtiosPotts' },
        searchBlob:
          `${potts.id} ${potts.shortName} ${potts.displayName} ${CONCISE_BLOCK_ROLES['artios-potts']} artios artiospotts potts mrf markov random field couplings gibbs dlmc mcmc ligand cofactor metal active site pocket pll pseudo-log-likelihood energy ligo biosciences apache-2.0`.toLowerCase(),
      },
    ];
    const phItems: CatalogSubItem[] = [
      {
        key: 'proton-pottsmpnn',
        label: proton.shortName,
        subtitle: CONCISE_BLOCK_ROLES['proton-pottsmpnn'],
        meta: `~${proton.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'proton-pottsmpnn', label: 'Proton-PottsMPNN' },
        searchBlob:
          `${proton.id} ${proton.shortName} ${proton.displayName} ${CONCISE_BLOCK_ROLES['proton-pottsmpnn']} proton protonpotts protonpottsmpnn pottsmpnn potts ph ph-selective protonation state titratable histidine his aspartate glutamate pka charge microstate block descent couplings energy rc-foundry jacobsen mit bsd-3-clause`.toLowerCase(),
      },
    ];
    const seqDiffItems: CatalogSubItem[] = [
      {
        key: 'evodiff',
        label: evo.shortName,
        subtitle: CONCISE_BLOCK_ROLES.evodiff,
        meta: `~${evo.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'evodiff', label: 'EvoDiff-38M' },
        searchBlob:
          `${evo.id} ${evo.shortName} ${evo.displayName} ${CONCISE_BLOCK_ROLES.evodiff} evodiff oadm 38m sequence diffusion infill inpainting msa-free mit`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [
      { title: 'ProteinMPNN Family', items: mpnn.subGroups[0]?.items ?? [] },
      { title: 'Ligand & All-Atom Inverse Folding', items: allAtomItems },
      { title: 'pH-Aware & Protonation-State Design', items: phItems },
      { title: 'Antibody & General GVP-Transformer', items: antifold.subGroups[0]?.items ?? [] },
      { title: 'Sequence Diffusion (MSA-Free)', items: seqDiffItems },
    ];
    const total = subGroups.reduce((a, g) => a + g.items.length, 0);
    const identityBlob = 'sequence design inverse fold folding';
    const searchBlob = `${identityBlob} ${subGroups.flatMap((g) => [g.title.toLowerCase(), ...g.items.map((i) => i.searchBlob)]).join(' ')}`;
    return {
      id: 'sequence-design',
      title: 'Sequence Design',
      stage: 'inverse-fold',
      subtitle: 'ProteinMPNN, LigandMPNN, LASErMPNN, AntiFold & EvoDiff-38M',
      countLabel: `${total}`,
      footnote: 'Runs locally over WebGPU — ProteinMPNN, LigandMPNN, LASErMPNN, AntiFold, ESM-IF1 & EvoDiff',
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const lf = buildSubGroupsForAdapter('localfold-af2');
    const total = lf.subGroups.reduce((a, g) => a + g.items.length, 0);
    const identityBlob = 'structure prediction folding fold localfold predict';
    const searchBlob = `${identityBlob} ${lf.subGroups.flatMap((g) => [g.title.toLowerCase(), ...g.items.map((i) => i.searchBlob)]).join(' ')}`;
    return {
      id: 'prediction',
      title: 'Structure Prediction',
      stage: 'fold',
      subtitle: 'AlphaFold2, RoseTTAFold3, Boltz-2, AlphaFold 3…',
      countLabel: `${total}`,
      footnote: 'Powered by LocalFold (sokrypton/localfold)',
      subGroups: lf.subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const th = ADAPTER_REGISTRY.thermompnn;
    const ec = ADAPTER_REGISTRY.esmc;
    const dy = ADAPTER_REGISTRY.dyna1;
    const gn = ADAPTER_REGISTRY.gnina;
    const items: CatalogSubItem[] = [
      {
        key: 'thermompnn',
        label: th.displayName,
        subtitle: CONCISE_BLOCK_ROLES.thermompnn,
        meta: `~${th.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'thermompnn', label: 'ThermoMPNN' },
        searchBlob:
          `${th.id} ${th.shortName} ${th.displayName} ${CONCISE_BLOCK_ROLES.thermompnn} thermompnn ddg stability mutation ssm site saturation heatmap bfactor kuhlman mit property`.toLowerCase(),
      },
      {
        key: 'esmc',
        label: ec.displayName,
        subtitle: CONCISE_BLOCK_ROLES.esmc,
        meta: `~${ec.approxWeightsMB} MB · Apache-2.0`,
        insert: { adapterId: 'esmc', label: 'ESM-C 300M' },
        searchBlob:
          `${ec.id} ${ec.shortName} ${ec.displayName} ${CONCISE_BLOCK_ROLES.esmc} esmc esm-c llr naturalness fitness mutation plm masked fill property`.toLowerCase(),
      },
      {
        key: 'dyna1',
        label: dy.displayName,
        subtitle: CONCISE_BLOCK_ROLES.dyna1,
        meta: `~${dy.approxWeightsMB} MB · MIT`,
        insert: { adapterId: 'dyna1', label: 'Dyna-1' },
        searchBlob:
          `${dy.id} ${dy.shortName} ${dy.displayName} ${CONCISE_BLOCK_ROLES.dyna1} dyna1 dyna-1 dynamics flexibility exchange nmr conformational property`.toLowerCase(),
      },
      {
        key: 'gnina',
        label: gn.displayName,
        subtitle: CONCISE_BLOCK_ROLES.gnina,
        meta: `~${gn.approxWeightsMB} MB · Apache-2.0 / GPLv2`,
        insert: { adapterId: 'gnina', label: 'GNINA 1.3' },
        searchBlob:
          `${gn.id} ${gn.shortName} ${gn.displayName} ${CONCISE_BLOCK_ROLES.gnina} gnina docking ligand affinity cnnscore cnnaffinity cnn_vs vina small molecule pocket scan minimize pose property`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [{ title: 'Stability, Fitness, Dynamics & Docking', items }];
    const identityBlob = 'property prediction properties biophysical stability fitness dynamics flexibility docking ligand affinity ddg llr nmr vina cnn';
    const searchBlob = `${identityBlob} ${items.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'property-prediction',
      title: 'Property Prediction',
      stage: 'score',
      subtitle: 'ThermoMPNN (ΔΔG), ESM-C 300M (LLR), Dyna-1 & GNINA 1.3 (ligand)',
      countLabel: `${items.length}`,
      footnote: 'Runs locally over WebGPU — stability, evolutionary fitness, dynamics & protein–ligand docking',
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const sc = ADAPTER_REGISTRY['designability-scorer'];
    const cmp = ADAPTER_REGISTRY['structure-compare'];
    const tk = ADAPTER_REGISTRY['top-k-filter'];
    const items: CatalogSubItem[] = [
      {
        key: 'top-k-filter',
        label: tk.displayName,
        subtitle: CONCISE_BLOCK_ROLES['top-k-filter'],
        insert: { adapterId: 'top-k-filter' },
        searchBlob: `${tk.id} ${tk.shortName} ${tk.displayName} ${CONCISE_BLOCK_ROLES['top-k-filter']} topk top-k filter ranker rank sequence batch multi-design`.toLowerCase(),
      },
      {
        key: 'designability-scorer',
        label: sc.shortName,
        subtitle: CONCISE_BLOCK_ROLES['designability-scorer'],
        insert: { adapterId: 'designability-scorer' },
        searchBlob: `${sc.id} ${sc.shortName} ${sc.displayName} ${CONCISE_BLOCK_ROLES['designability-scorer']} score evaluation designability kabsch scrmsd sctm plddt`.toLowerCase(),
      },
      {
        key: 'structure-compare',
        label: cmp.shortName,
        subtitle: CONCISE_BLOCK_ROLES['structure-compare'],
        insert: { adapterId: 'structure-compare', label: '3D Compare' },
        searchBlob: `${cmp.id} ${cmp.shortName} ${cmp.displayName} ${CONCISE_BLOCK_ROLES['structure-compare']} compare viewer superpose overlay align motif rmsd multi-model 3d`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [{ title: 'Filtering, Designability & 3D Compare', items }];
    const identityBlob = 'evaluation score scoring metrics filter compare superpose';
    const searchBlob = `${identityBlob} ${items.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'evaluation',
      title: 'Evaluation',
      stage: 'score',
      subtitle: 'Top-K Filter, scRMSD / scTM & 3D Compare',
      countLabel: `${items.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),

  // 6. Refinement / Backmapping (PULCHRA & cg2all)
  (() => {
    const pu = ADAPTER_REGISTRY.pulchra;
    const cg = ADAPTER_REGISTRY.cg2all;
    const items: CatalogSubItem[] = [
      {
        key: 'pulchra',
        label: pu.shortName,
        subtitle: CONCISE_BLOCK_ROLES.pulchra,
        meta: '~1.7 MB · CPU · MIT',
        insert: { adapterId: 'pulchra', label: 'PULCHRA' },
        searchBlob: `${pu.id} ${pu.shortName} ${pu.displayName} ${CONCISE_BLOCK_ROLES.pulchra} refinement backmapping all-atom reconstruct rebuild sidechain backbone ca trace`.toLowerCase(),
      },
      {
        key: 'cg2all',
        label: cg.shortName,
        subtitle: CONCISE_BLOCK_ROLES.cg2all,
        meta: '~8 MB · WebGPU · Apache-2.0 / CC-BY-4.0',
        insert: { adapterId: 'cg2all', label: 'cg2all' },
        searchBlob: `${cg.id} ${cg.shortName} ${cg.displayName} ${CONCISE_BLOCK_ROLES.cg2all} refinement backmapping all-atom neural equivariant ca-fix sidechain backbone`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [{ title: 'All-Atom Reconstruction & Backmapping', items }];
    const identityBlob = 'refinement backmapping reconstruct rebuild all-atom sidechain backbone ca trace';
    const searchBlob = `${identityBlob} ${items.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'refinement',
      title: 'Refinement / Backmapping',
      stage: 'score',
      subtitle: 'PULCHRA & cg2all All-Atom Rebuild',
      countLabel: `${items.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
];

function tokenizeCatalogText(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

function compactCatalogText(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function scoreCatalogSubItem(
  sub: CatalogSubItem,
  groupTitle: string,
  q: string,
  terms: string[],
): number {
  const qCompact = compactCatalogText(q);
  const labelL = sub.label.toLowerCase();
  const insertLabelL = (sub.insert.label ?? '').toLowerCase();
  const keyL = sub.key.toLowerCase();
  const adapterL = sub.insert.adapterId.toLowerCase();
  const subtitleL = (sub.subtitle ?? '').toLowerCase();
  const groupL = groupTitle.toLowerCase();

  const labelWords = tokenizeCatalogText(labelL);
  const insertWords = tokenizeCatalogText(insertLabelL);
  const subtitleWords = tokenizeCatalogText(subtitleL);
  const groupWords = tokenizeCatalogText(groupL);
  const blobWords = tokenizeCatalogText(sub.searchBlob);

  let score = 0;

  if (labelL === q || (insertLabelL && insertLabelL === q)) {
    score += 1200;
  } else if (
    compactCatalogText(labelL) === qCompact ||
    keyL === q ||
    adapterL === q ||
    (insertLabelL && compactCatalogText(insertLabelL) === qCompact)
  ) {
    score += 1100;
  } else if (labelL.startsWith(q) || (insertLabelL && insertLabelL.startsWith(q))) {
    score += 850;
  } else if (
    (qCompact && compactCatalogText(labelL).startsWith(qCompact)) ||
    keyL.startsWith(q) ||
    adapterL.startsWith(q) ||
    (insertLabelL && qCompact && compactCatalogText(insertLabelL).startsWith(qCompact))
  ) {
    score += 750;
  } else if (labelL.includes(q) || (insertLabelL && insertLabelL.includes(q))) {
    score += 600;
  }

  for (const t of terms) {
    if (labelWords.includes(t) || insertWords.includes(t)) {
      score += 200;
    } else if (
      labelWords.some((w) => w.startsWith(t)) ||
      insertWords.some((w) => w.startsWith(t))
    ) {
      score += 140;
    } else if (labelL.includes(t) || insertLabelL.includes(t) || keyL.includes(t)) {
      score += 80;
    } else if (subtitleWords.includes(t) || groupWords.includes(t)) {
      score += 40;
    } else if (
      subtitleWords.some((w) => w.startsWith(t)) ||
      groupWords.some((w) => w.startsWith(t))
    ) {
      score += 25;
    } else if (blobWords.includes(t)) {
      score += 15;
    } else if (sub.searchBlob.includes(t)) {
      score += 5;
    }
  }

  return score;
}

function scoreCatalogCategory(
  cat: CatalogCategory,
  matchingSubGroups: CatalogSubGroup[],
  q: string,
  terms: string[],
): number {
  const catTitleL = cat.title.toLowerCase();
  const catTitleWords = tokenizeCatalogText(catTitleL);
  const identityWords = tokenizeCatalogText(cat.identityBlob);

  let catOwnScore = 0;
  if (catTitleL === q) {
    catOwnScore = 1500;
  } else if (catTitleL.startsWith(q)) {
    catOwnScore = 1000;
  } else if (terms.length > 0 && terms.every((t) => catTitleWords.includes(t))) {
    catOwnScore = 700;
  } else {
    for (const t of terms) {
      if (catTitleWords.includes(t)) catOwnScore += 120;
      else if (catTitleWords.some((w) => w.startsWith(t))) catOwnScore += 70;
      else if (identityWords.includes(t)) catOwnScore += 30;
    }
  }

  let bestItemScore = 0;
  for (const sg of matchingSubGroups) {
    for (const sub of sg.items) {
      const s = scoreCatalogSubItem(sub, sg.title, q, terms);
      if (s > bestItemScore) bestItemScore = s;
    }
  }

  return Math.max(catOwnScore, bestItemScore) + Math.min(catOwnScore, bestItemScore) * 0.01;
}

function scoreCatalogItem(
  item: CatalogItem,
  matchingSubGroups: CatalogSubGroup[],
  q: string,
  terms: string[],
): number {
  const qCompact = compactCatalogText(q);
  const shortL = item.shortName.toLowerCase();
  const dispL = item.displayName.toLowerCase();
  const idL = item.id.toLowerCase();
  const shortWords = tokenizeCatalogText(shortL);
  const dispWords = tokenizeCatalogText(dispL);
  const ownWords = tokenizeCatalogText(item.ownSearchBlob);

  let ownScore = 0;
  if (shortL === q || dispL === q) {
    ownScore += 1200;
  } else if (
    compactCatalogText(shortL) === qCompact ||
    compactCatalogText(dispL) === qCompact ||
    idL === q
  ) {
    ownScore += 1100;
  } else if (shortL.startsWith(q) || dispL.startsWith(q)) {
    ownScore += 850;
  } else if (
    (qCompact && compactCatalogText(shortL).startsWith(qCompact)) ||
    (qCompact && compactCatalogText(dispL).startsWith(qCompact)) ||
    idL.startsWith(q)
  ) {
    ownScore += 750;
  } else if (shortL.includes(q) || dispL.includes(q)) {
    ownScore += 600;
  }

  for (const t of terms) {
    if (shortWords.includes(t) || dispWords.includes(t)) ownScore += 200;
    else if (shortWords.some((w) => w.startsWith(t)) || dispWords.some((w) => w.startsWith(t))) ownScore += 140;
    else if (shortL.includes(t) || dispL.includes(t) || idL.includes(t)) ownScore += 80;
    else if (ownWords.includes(t)) ownScore += 30;
    else if (item.ownSearchBlob.includes(t)) ownScore += 10;
  }

  let bestSubScore = 0;
  for (const sg of matchingSubGroups) {
    for (const sub of sg.items) {
      const s = scoreCatalogSubItem(sub, sg.title, q, terms);
      if (s > bestSubScore) bestSubScore = s;
    }
  }

  return Math.max(ownScore, bestSubScore) + Math.min(ownScore, bestSubScore) * 0.01;
}

export function filterCatalogCategories(
  query: string,
  categories: CatalogCategory[] = CATALOG_CATEGORIES,
): FilteredCategoryEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return categories.map((category) => ({
      category,
      matchingSubGroups: category.subGroups,
      matchingSubItems: category.subGroups.flatMap((g) => g.items),
      hasNarrowedSubMatch: false,
    }));
  }

  const terms = q.split(/\s+/).filter(Boolean);
  const scoredResults: Array<{ entry: FilteredCategoryEntry; score: number; origIdx: number }> = [];

  for (let catIdx = 0; catIdx < categories.length; catIdx++) {
    const category = categories[catIdx];
    if (!terms.every((t) => category.searchBlob.includes(t))) continue;
    const totalSubCount = category.subGroups.reduce((a, g) => a + g.items.length, 0);

    // Pure category-level match (e.g. "prediction", "localfold", "generation") -> show all sub-items
    if (terms.every((t) => category.identityBlob.includes(t))) {
      const entry: FilteredCategoryEntry = {
        category,
        matchingSubGroups: category.subGroups,
        matchingSubItems: category.subGroups.flatMap((g) => g.items),
        hasNarrowedSubMatch: false,
      };
      scoredResults.push({
        entry,
        score: scoreCatalogCategory(category, entry.matchingSubGroups, q, terms),
        origIdx: catIdx,
      });
      continue;
    }

    const subTargetTerms = terms.filter(
      (t) =>
        !category.identityBlob.includes(t) &&
        category.subGroups.some((sg) => {
          const titleL = sg.title.toLowerCase();
          return sg.items.some((sub) => `${titleL} ${sub.searchBlob}`.includes(t));
        }),
    );

    if (subTargetTerms.length > 0) {
      const narrowedWithScore: Array<{ sg: CatalogSubGroup; bestScore: number; origSgIdx: number }> = [];
      for (let sgIdx = 0; sgIdx < category.subGroups.length; sgIdx++) {
        const sg = category.subGroups[sgIdx];
        const titleL = sg.title.toLowerCase();
        const matched = sg.items
          .map((sub, itemIdx) => ({
            sub,
            itemIdx,
            score: scoreCatalogSubItem(sub, sg.title, q, terms),
          }))
          .filter(({ sub }) => {
            const hay = `${titleL} ${sub.searchBlob}`;
            return subTargetTerms.every((t) => hay.includes(t));
          });
        if (matched.length > 0) {
          matched.sort((a, b) => b.score - a.score || a.itemIdx - b.itemIdx);
          narrowedWithScore.push({
            sg: { title: sg.title, items: matched.map((m) => m.sub) },
            bestScore: matched[0].score,
            origSgIdx: sgIdx,
          });
        }
      }
      if (narrowedWithScore.length > 0) {
        narrowedWithScore.sort((a, b) => b.bestScore - a.bestScore || a.origSgIdx - b.origSgIdx);
        const narrowed = narrowedWithScore.map((n) => n.sg);
        const matchingSubItems = narrowed.flatMap((g) => g.items);
        const entry: FilteredCategoryEntry = {
          category,
          matchingSubGroups: narrowed,
          matchingSubItems,
          hasNarrowedSubMatch: matchingSubItems.length < totalSubCount || totalSubCount === 1,
        };
        scoredResults.push({
          entry,
          score: scoreCatalogCategory(category, narrowed, q, terms),
          origIdx: catIdx,
        });
        continue;
      }
    }

    const entry: FilteredCategoryEntry = {
      category,
      matchingSubGroups: category.subGroups,
      matchingSubItems: category.subGroups.flatMap((g) => g.items),
      hasNarrowedSubMatch: false,
    };
    scoredResults.push({
      entry,
      score: scoreCatalogCategory(category, entry.matchingSubGroups, q, terms),
      origIdx: catIdx,
    });
  }

  scoredResults.sort((a, b) => b.score - a.score || a.origIdx - b.origIdx);
  return scoredResults.map((r) => r.entry);
}

export function filterCatalog(query: string, items: CatalogItem[] = CATALOG_ITEMS): FilteredCatalogEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return items.map((item) => {
      const matchingSubItems = item.subGroups.flatMap((g) => g.items);
      return {
        item,
        directParentMatch: true,
        matchingSubGroups: item.subGroups,
        matchingSubItems,
        hasNarrowedSubMatch: false,
      };
    });
  }

  const terms = q.split(/\s+/).filter(Boolean);
  const scoredResults: Array<{ entry: FilteredCatalogEntry; score: number; origIdx: number }> = [];

  for (let itemIdx = 0; itemIdx < items.length; itemIdx++) {
    const item = items[itemIdx];
    if (!terms.every((t) => item.searchBlob.includes(t))) continue;

    const directParentMatch = terms.every((t) => item.ownSearchBlob.includes(t));
    const totalSubCount = item.subGroups.reduce((acc, g) => acc + g.items.length, 0);

    if (item.subGroups.length === 0) {
      const entry: FilteredCatalogEntry = {
        item,
        directParentMatch,
        matchingSubGroups: [],
        matchingSubItems: [],
        hasNarrowedSubMatch: false,
      };
      scoredResults.push({
        entry,
        score: scoreCatalogItem(item, [], q, terms),
        origIdx: itemIdx,
      });
      continue;
    }

    // 1. If every query term matches the block's own identity (e.g. "rfd4", "localfold", "proteinmpnn"),
    // expose all sub-groups without mis-narrowing to a coincidental sub-item match.
    if (terms.every((t) => item.identityBlob.includes(t))) {
      const matchingSubItems = item.subGroups.flatMap((g) => g.items);
      const entry: FilteredCatalogEntry = {
        item,
        directParentMatch,
        matchingSubGroups: item.subGroups,
        matchingSubItems,
        hasNarrowedSubMatch: false,
      };
      scoredResults.push({
        entry,
        score: scoreCatalogItem(item, item.subGroups, q, terms),
        origIdx: itemIdx,
      });
      continue;
    }

    // 2. Narrow across parent + sub-item terms (e.g. "localfold boltz", "proteinmpnn soluble", "rfd4 motif", "rfd4 hotspot")
    // before falling back to directParentMatch.
    const subTargetTerms = terms.filter(
      (t) =>
        !item.identityBlob.includes(t) &&
        item.subGroups.some((sg) => {
          const sgTitleLower = sg.title.toLowerCase();
          return sg.items.some((sub) => `${sgTitleLower} ${sub.searchBlob}`.includes(t));
        }),
    );
    const parentOnlyTerms = terms.filter((t) => !subTargetTerms.includes(t));

    if (subTargetTerms.length > 0 && parentOnlyTerms.every((t) => item.ownSearchBlob.includes(t))) {
      const narrowedWithScore: Array<{ sg: CatalogSubGroup; bestScore: number; origSgIdx: number }> = [];
      for (let sgIdx = 0; sgIdx < item.subGroups.length; sgIdx++) {
        const sg = item.subGroups[sgIdx];
        const sgTitleLower = sg.title.toLowerCase();
        const matched = sg.items
          .map((sub, subIdx) => ({
            sub,
            subIdx,
            score: scoreCatalogSubItem(sub, sg.title, q, terms),
          }))
          .filter(({ sub }) => {
            const hay = `${sgTitleLower} ${sub.searchBlob}`;
            return subTargetTerms.every((t) => hay.includes(t));
          });
        if (matched.length > 0) {
          matched.sort((a, b) => b.score - a.score || a.subIdx - b.subIdx);
          narrowedWithScore.push({
            sg: { title: sg.title, items: matched.map((m) => m.sub) },
            bestScore: matched[0].score,
            origSgIdx: sgIdx,
          });
        }
      }
      if (narrowedWithScore.length > 0) {
        narrowedWithScore.sort((a, b) => b.bestScore - a.bestScore || a.origSgIdx - b.origSgIdx);
        const narrowedGroups = narrowedWithScore.map((n) => n.sg);
        const matchingSubItems = narrowedGroups.flatMap((g) => g.items);
        const entry: FilteredCatalogEntry = {
          item,
          directParentMatch,
          matchingSubGroups: narrowedGroups,
          matchingSubItems,
          hasNarrowedSubMatch: matchingSubItems.length < totalSubCount,
        };
        scoredResults.push({
          entry,
          score: scoreCatalogItem(item, narrowedGroups, q, terms),
          origIdx: itemIdx,
        });
        continue;
      }
    }

    // 3. Fallback: if parent matches directly (e.g. "prediction" or role keywords), expose all sub-groups
    if (directParentMatch) {
      const matchingSubItems = item.subGroups.flatMap((g) => g.items);
      const entry: FilteredCatalogEntry = {
        item,
        directParentMatch,
        matchingSubGroups: item.subGroups,
        matchingSubItems,
        hasNarrowedSubMatch: false,
      };
      scoredResults.push({
        entry,
        score: scoreCatalogItem(item, item.subGroups, q, terms),
        origIdx: itemIdx,
      });
      continue;
    }

    // 4. Cross-boundary query matched multiple distinct sub-items without one unifying sub-item
    const entry: FilteredCatalogEntry = {
      item,
      directParentMatch,
      matchingSubGroups: [],
      matchingSubItems: [],
      hasNarrowedSubMatch: false,
    };
    scoredResults.push({
      entry,
      score: scoreCatalogItem(item, [], q, terms),
      origIdx: itemIdx,
    });
  }

  scoredResults.sort((a, b) => b.score - a.score || a.origIdx - b.origIdx);
  return scoredResults.map((r) => r.entry);
}
