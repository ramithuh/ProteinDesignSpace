// Full motif-scaffolding benchmark suite sourced from Proteina-Complexa
// `configs/design_tasks/motif_dict.yaml` (RFdiffusion 25 + RFD4 all-atom motif tasks).
//
// ── Deduplication from the 77 YAML entries -> 30 canonical tasks ──
// (26 `motif_data_aa/` tasks + their 26 `_TIP` twins + 25 plain `motif_data/` tasks.)
// 1. `_TIP` variants (`1PRW_AA` vs `1PRW_AA_TIP`, …) are the *same* design task with
//    `atom_selection_mode: tip_atoms` instead of `all_atom`. Tip-atom conditioning is a
//    runtime toggle in this app (Protpardelle tip checkpoints / RFD4 `motifAtoms`), not a
//    separate target, so every `_TIP` entry collapses into its canonical task.
// 2. `*_AA` variants duplicate their plain counterpart (`1BCF_AA` vs `1BCF`, …). They differ
//    only in which preprocessed PDB they point at (`motif_data_aa/` vs `motif_data/`), so the
//    pair collapses into one task. Where the two disagree on chain letters we keep the one that
//    matches real RCSB auth numbering (see 3. below).
// 3. The YAML contigs reference *preprocessed* PDB copies whose chains are sometimes renamed to
//    per-segment labels (`A;B;C;D`) and whose residues are sometimes renumbered from 1. Because
//    this app streams structures straight from RCSB, every motif span below was re-resolved to
//    real auth chain/residue IDs and verified against the Genie 2 `data/design25` reference
//    motif PDBs plus the deposited RCSB coordinates. Corrections applied:
//      - 3IXT  `A254-277`        -> `P254-277` (chain P = fusion glycoprotein F1)
//      - 4JHW  `B196-212/A63-69` -> `F196-212/F63-69` (chain F = fusion glycoprotein F0)
//      - 5IUS  `B119-140/A63-82` -> `A119-140/A63-82` (chain A = PD-1; B is a second copy)
//      - 1YCR  `A19-27`          -> `B19-27` (chain B = p53 peptide, A = MDM2)
//      - 7MRX  `A25-46`          -> `B25-46` (chain B = barstar, A = barnase)
//      - 1QJG  `B38/A14/C99`     -> `A38/A14/A99` (single KSI chain)
//      - 2KL8  `A1-7/B28-79`     -> `A1-7/A28-79` (single OR15 chain)
//      - 5YUI  `A93-97/B118-120/C198-200` -> all chain A (single CA II chain)
//      - 1BCF  `A92-99/B123-130/C47-54/D18-25` -> all chain A (one bacterioferritin subunit)
//      - 1PRW  `A1-20/B1-20`     -> `A16-35/A52-71` (renumbered-from-1 copy -> calmodulin)
//      - 6EXZ  `A28-42`          -> `A557-571` (renumbered-from-1 copy -> MEX67 C-term)
// 4. `1QJG` keeps three distinct tasks: the YAML's shuffled-order (`1QJG`), index-ordered
//    (`1QJG_incr_order`) and native-spacing (`1QJG_AA_NATIVE`, 115–135 aa) problems.
// 5. The six pre-existing curated motif IDs (`rfd_5tpn_rsvf_v`, `rfd_1prw_efhand`,
//    `g3_5yui_ca2`, `rfd_1ycr_p53`, `rfd_1bcf_diiron`, `g3_4xoj_trypsin`) are preserved
//    verbatim and cover the 1PRW/1BCF/5TPN/5YUI/1YCR tasks, so those five YAML tasks are not
//    re-added. `4XOJ` is a Genie 3 MotifBench target with no `motif_dict.yaml` counterpart.

import type { BenchmarkTargetMeta } from './benchmarks';

/**
 * The six curated motif benchmarks that shipped before the `motif_dict.yaml` import, kept
 * byte-for-byte so existing IDs, crops, contigs and motif spans stay 100% backward compatible.
 */
const CURATED_MOTIF_BENCHMARKS: BenchmarkTargetMeta[] = [
  {
    id: 'rfd_5tpn_rsvf_v',
    name: 'RSV-F Site V Epitope',
    shortLabel: 'RSV-F Site V (5TPN)',
    category: 'motif',
    pdbId: '5TPN',
    sourceFamily: 'RFdiffusion 25 / Genie 3 RSV-F',
    approxTokens: 75,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Viral prefusion neutralizing helix–strand epitope (A163–181)',
    cropSegments: 'A163-181',
    conditionParams: {
      taskPreset: 'custom',
      contig: '25-25,A163-181,31-31',
      motifSpan: 'A163-181',
    },
  },
  {
    id: 'rfd_1prw_efhand',
    name: 'Calmodulin EF-Hand',
    shortLabel: 'Calmodulin EF-Hand (1PRW)',
    category: 'motif',
    pdbId: '1PRW',
    sourceFamily: 'RFdiffusion / RFD4',
    approxTokens: 76,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Twin Ca²⁺-binding EF-hand loops (A16–35 & A52–71)',
    cropSegments: 'A16-35, A52-71',
    conditionParams: {
      taskPreset: 'custom',
      contig: '10-10,A16-35,16-16,A52-71,10-10',
      motifSpan: 'A16-35, A52-71',
    },
  },
  {
    id: 'g3_5yui_ca2',
    name: 'Carbonic Anhydrase II',
    shortLabel: 'CA II His₃ Site (5YUI)',
    category: 'motif',
    pdbId: '5YUI',
    sourceFamily: 'RFdiffusion / Genie 3 MotifBench',
    approxTokens: 75,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: '3-island catalytic His₃ Zn²⁺ binding site with Thr199 gatekeeper',
    cropSegments: 'A93-97, A118-120, A198-200',
    conditionParams: {
      taskPreset: 'custom',
      contig: '14-14,A93-97,20-20,A118-120,20-20,A198-200,10-10',
      motifSpan: 'A93-97, A118-120, A198-200',
    },
  },
  {
    id: 'rfd_1ycr_p53',
    name: 'p53 Helix (MDM2 Motif)',
    shortLabel: 'p53 Helix Motif (1YCR)',
    category: 'motif',
    pdbId: '1YCR',
    sourceFamily: 'RFdiffusion Classic',
    approxTokens: 65,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: '9-aa cancer-suppressor helical motif (B19–27, F19/W23/L26 triad)',
    cropSegments: 'B19-27',
    conditionParams: {
      taskPreset: 'custom',
      contig: '28-28,B19-27,28-28',
      motifSpan: 'B19-27',
    },
  },
  {
    id: 'rfd_1bcf_diiron',
    name: 'Di-Iron 4-Helix Bundle',
    shortLabel: 'Di-Iron Site (1BCF)',
    category: 'motif',
    pdbId: '1BCF',
    sourceFamily: 'RFdiffusion / Genie 3',
    approxTokens: 125,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: '4-island bacterioferritin di-iron coordinating helices',
    cropSegments: 'A18-25, A47-54, A92-99, A123-130',
    conditionParams: {
      taskPreset: 'custom',
      contig: '12-12,A92-99,24-24,A123-130,24-24,A47-54,23-23,A18-25,10-10',
      motifSpan: 'A18-25, A47-54, A92-99, A123-130',
    },
  },
  {
    id: 'g3_4xoj_trypsin',
    name: 'Trypsin Catalytic Triad',
    shortLabel: 'Trypsin Triad (4XOJ)',
    category: 'motif',
    pdbId: '4XOJ',
    sourceFamily: 'Genie 3 MotifBench',
    approxTokens: 150,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Metal-free Ser–His–Asp catalytic triad + oxyanion hole',
    cropSegments: 'A55, A99, A190-192',
    conditionParams: {
      taskPreset: 'custom',
      contig: '15-15,A55-55,45-45,A99-99,70-70,A190-192,15-15',
      motifSpan: 'A55, A99, A190-192',
    },
  },
];

interface MotifRawTask {
  id: string;
  pdbId: string;
  name: string;
  shortLabel: string;
  sourceFamily: string;
  biology: string;
  /** `motif_dict.yaml` keys this entry canonicalizes, including deduped `_AA`/`_TIP` variants. */
  yamlKeys: string[];
  /**
   * `contig_string` from `motif_dict.yaml` (`/`-delimited), with motif chain letters and residue
   * numbers remapped to real RCSB auth IDs. Flank tokens keep the YAML's `min-max` windows.
   */
  contigString: string;
  /** `[motif_min_length, motif_max_length]` total design-length window from the YAML. */
  designLength: [number, number];
  /**
   * Canonical total designed length. Always inside `designLength` and capped at the 300-token
   * WebGPU budget (only 7K4V's 280–320 window needs the cap).
   */
  targetLength: number;
}

const MOTIF_RAW_TASKS: MotifRawTask[] = [
  // ── Viral epitope scaffolding (RSV F / G) ──
  {
    id: 'rfd_3ixt_rsvf_ii',
    pdbId: '3IXT',
    name: 'RSV-F Site II Epitope',
    shortLabel: 'RSV-F Site II (3IXT)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Motavizumab-bound RSV fusion glycoprotein site II helix–loop epitope (P254–277)',
    yamlKeys: ['3IXT', '3IXT_AA', '3IXT_AA_TIP'],
    contigString: '10-40/P254-277/10-40',
    designLength: [50, 75],
    targetLength: 75,
  },
  {
    id: 'rfd_4jhw_rsvf_zero',
    pdbId: '4JHW',
    name: 'RSV-F Site Ø Epitope',
    shortLabel: 'RSV-F Site Ø (4JHW)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: '2-island prefusion-specific RSV-F site Ø epitope bound by mAb D25 (F63–69 & F196–212)',
    yamlKeys: ['4JHW', '4JHW_AA', '4JHW_AA_TIP'],
    contigString: '10-25/F196-212/15-30/F63-69/10-25',
    designLength: [60, 90],
    targetLength: 90,
  },
  {
    id: 'rfd_4zyp_rsvf_iv',
    pdbId: '4ZYP',
    name: 'RSV-F Site IV Epitope',
    shortLabel: 'RSV-F Site IV (4ZYP)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'RSV fusion glycoprotein site IV β-strand epitope bound by mAb 101F (A422–436)',
    yamlKeys: ['4ZYP', '4ZYP_AA', '4ZYP_AA_TIP'],
    contigString: '10-40/A422-436/10-40',
    designLength: [30, 50],
    targetLength: 50,
  },
  {
    id: 'rfd_5wn9_rsvg_ccr',
    pdbId: '5WN9',
    name: 'RSV-G Central Conserved Region',
    shortLabel: 'RSV-G CCR (5WN9)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Cystine-noose central conserved region of RSV surface glycoprotein G bound by scFv 2D10 (A170–189)',
    yamlKeys: ['5WN9', '5WN9_AA', '5WN9_AA_TIP'],
    contigString: '10-40/A170-189/10-40',
    designLength: [35, 50],
    targetLength: 50,
  },

  // ── Immune-checkpoint interface scaffolding ──
  {
    id: 'rfd_5ius_pd1',
    pdbId: '5IUS',
    name: 'PD-1 Binding Interface',
    shortLabel: 'PD-1 Interface (5IUS)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: '2-island PD-1 IgV interface that engages PD-L1 (A63–82 & A119–140)',
    yamlKeys: ['5IUS', '5IUS_AA', '5IUS_AA_TIP'],
    contigString: '0-30/A119-140/15-40/A63-82/0-30',
    designLength: [57, 142],
    targetLength: 142,
  },

  // ── Enzyme active-site scaffolding (single-residue islands) ──
  {
    id: 'rfd_1qjg_ksi',
    pdbId: '1QJG',
    name: 'Ketosteroid Isomerase Triad',
    shortLabel: 'KSI Triad (1QJG)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: '3 single-residue catalytic islands of Δ⁵-3-ketosteroid isomerase D38N (A14 Tyr, A38 Asn, A99 Asp)',
    yamlKeys: ['1QJG', '1QJG_AA', '1QJG_AA_TIP'],
    contigString: '10-20/A38/15-30/A14/15-30/A99/10-20',
    designLength: [53, 103],
    targetLength: 103,
  },
  {
    id: 'rfd_1qjg_ksi_indexed',
    pdbId: '1QJG',
    name: 'Ketosteroid Isomerase Triad (Index-Ordered)',
    shortLabel: 'KSI Triad · Indexed (1QJG)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'KSI catalytic triad presented in ascending sequence order (A14 → A38 → A99)',
    yamlKeys: ['1QJG_incr_order'],
    contigString: '10-20/A14/15-30/A38/15-30/A99/10-20',
    designLength: [53, 103],
    targetLength: 103,
  },
  {
    id: 'rfdaa_1qjg_ksi_native',
    pdbId: '1QJG',
    name: 'Ketosteroid Isomerase Triad (Native Spacing)',
    shortLabel: 'KSI Triad · Native (1QJG)',
    sourceFamily: 'RFD4 All-Atom / Proteina-Complexa',
    biology: 'KSI catalytic triad with native-length linkers on a 115–135 aa scaffold (A14 → A38 → A99)',
    yamlKeys: ['1QJG_AA_NATIVE', '1QJG_AA_NATIVE_TIP'],
    contigString: '10-20/A14/15-30/A38/50-70/A99/25-30',
    designLength: [115, 135],
    targetLength: 135,
  },

  // ── De novo scaffold reconstruction (fixed termini) ──
  {
    id: 'rfd_2kl8_or15',
    pdbId: '2KL8',
    name: 'De Novo Ferredoxin Fold (Fixed Termini)',
    shortLabel: 'OR15 Fixed Termini (2KL8)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'De novo ferredoxin-like fold OR15 with both native termini fixed and a 20 aa loop rebuilt (A1–7 & A28–79)',
    yamlKeys: ['2KL8', '2KL8_AA', '2KL8_AA_TIP'],
    contigString: 'A1-7/20/A28-79',
    designLength: [79, 79],
    targetLength: 79,
  },

  // ── Barstar binding helix (3 scaffold-length tiers) ──
  {
    id: 'rfd_7mrx_barstar_60',
    pdbId: '7MRX',
    name: 'Barstar Binding Helix (60 aa)',
    shortLabel: 'Barstar Helix · 60 (7MRX)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Barnase-binding barstar helix scaffolded into a 60 aa protein (B25–46)',
    yamlKeys: ['7MRX_60', '7MRX_AA_60', '7MRX_AA_60_TIP'],
    contigString: '0-38/B25-46/0-38',
    designLength: [60, 60],
    targetLength: 60,
  },
  {
    id: 'rfd_7mrx_barstar_85',
    pdbId: '7MRX',
    name: 'Barstar Binding Helix (85 aa)',
    shortLabel: 'Barstar Helix · 85 (7MRX)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Barnase-binding barstar helix scaffolded into an 85 aa protein (B25–46)',
    yamlKeys: ['7MRX_85', '7MRX_AA_85', '7MRX_AA_85_TIP'],
    contigString: '0-63/B25-46/0-63',
    designLength: [85, 85],
    targetLength: 85,
  },
  {
    id: 'rfd_7mrx_barstar_128',
    pdbId: '7MRX',
    name: 'Barstar Binding Helix (128 aa)',
    shortLabel: 'Barstar Helix · 128 (7MRX)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Barnase-binding barstar helix scaffolded into a 128 aa protein (B25–46)',
    yamlKeys: ['7MRX_128', '7MRX_AA_128', '7MRX_AA_128_TIP'],
    contigString: '0-122/B25-46/0-122',
    designLength: [128, 128],
    targetLength: 128,
  },

  // ── De novo NTF2 helix (3 scaffold-length tiers) ──
  {
    id: 'rfd_5trv_ntf2_short',
    pdbId: '5TRV',
    name: 'De Novo NTF2 Helix (56 aa)',
    shortLabel: 'NTF2 Helix · 56 (5TRV)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Long α-helix excised from a de novo NTF2-like fold, rescaffolded at 56 aa (A45–65)',
    yamlKeys: ['5TRV_short', '5TRV_AA_SHORT', '5TRV_AA_SHORT_TIP'],
    contigString: '0-35/A45-65/0-35',
    designLength: [56, 56],
    targetLength: 56,
  },
  {
    id: 'rfd_5trv_ntf2_med',
    pdbId: '5TRV',
    name: 'De Novo NTF2 Helix (86 aa)',
    shortLabel: 'NTF2 Helix · 86 (5TRV)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Long α-helix excised from a de novo NTF2-like fold, rescaffolded at 86 aa (A45–65)',
    yamlKeys: ['5TRV_med', '5TRV_AA_MED', '5TRV_AA_MED_TIP'],
    contigString: '0-65/A45-65/0-65',
    designLength: [86, 86],
    targetLength: 86,
  },
  {
    id: 'rfd_5trv_ntf2_long',
    pdbId: '5TRV',
    name: 'De Novo NTF2 Helix (116 aa)',
    shortLabel: 'NTF2 Helix · 116 (5TRV)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Long α-helix excised from a de novo NTF2-like fold, rescaffolded at 116 aa (A45–65)',
    yamlKeys: ['5TRV_long', '5TRV_AA_LONG', '5TRV_AA_LONG_TIP'],
    contigString: '0-95/A45-65/0-95',
    designLength: [116, 116],
    targetLength: 116,
  },

  // ── Bfd [2Fe-2S] ferredoxin loop (3 scaffold-length tiers) ──
  {
    id: 'rfd_6e6r_bfd_short',
    pdbId: '6E6R',
    name: 'Bfd [2Fe-2S] Loop (48 aa)',
    shortLabel: 'Bfd Loop · 48 (6E6R)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Cys-rich [2Fe-2S] binding loop of bacterioferritin-associated ferredoxin at 48 aa (A23–35)',
    yamlKeys: ['6E6R_short', '6E6R_AA_SHORT', '6E6R_AA_SHORT_TIP'],
    contigString: '0-35/A23-35/0-35',
    designLength: [48, 48],
    targetLength: 48,
  },
  {
    id: 'rfd_6e6r_bfd_med',
    pdbId: '6E6R',
    name: 'Bfd [2Fe-2S] Loop (78 aa)',
    shortLabel: 'Bfd Loop · 78 (6E6R)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Cys-rich [2Fe-2S] binding loop of bacterioferritin-associated ferredoxin at 78 aa (A23–35)',
    yamlKeys: ['6E6R_med', '6E6R_AA_MED', '6E6R_AA_MED_TIP'],
    contigString: '0-65/A23-35/0-65',
    designLength: [78, 78],
    targetLength: 78,
  },
  {
    id: 'rfd_6e6r_bfd_long',
    pdbId: '6E6R',
    name: 'Bfd [2Fe-2S] Loop (108 aa)',
    shortLabel: 'Bfd Loop · 108 (6E6R)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'Cys-rich [2Fe-2S] binding loop of bacterioferritin-associated ferredoxin at 108 aa (A23–35)',
    yamlKeys: ['6E6R_long', '6E6R_AA_LONG', '6E6R_AA_LONG_TIP'],
    contigString: '0-95/A23-35/0-95',
    designLength: [108, 108],
    targetLength: 108,
  },

  // ── MEX67 C-terminal helix (3 scaffold-length tiers) ──
  {
    id: 'rfd_6exz_mex67_short',
    pdbId: '6EXZ',
    name: 'MEX67 C-Terminal Helix (50 aa)',
    shortLabel: 'MEX67 Helix · 50 (6EXZ)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'C-terminal UBA-domain helix of mRNA export factor MEX67 at 50 aa (A557–571)',
    yamlKeys: ['6EXZ_short'],
    contigString: '0-35/A557-571/0-35',
    designLength: [50, 50],
    targetLength: 50,
  },
  {
    id: 'rfd_6exz_mex67_med',
    pdbId: '6EXZ',
    name: 'MEX67 C-Terminal Helix (80 aa)',
    shortLabel: 'MEX67 Helix · 80 (6EXZ)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'C-terminal UBA-domain helix of mRNA export factor MEX67 at 80 aa (A557–571)',
    yamlKeys: ['6EXZ_med'],
    contigString: '0-65/A557-571/0-65',
    designLength: [80, 80],
    targetLength: 80,
  },
  {
    id: 'rfd_6exz_mex67_long',
    pdbId: '6EXZ',
    name: 'MEX67 C-Terminal Helix (110 aa)',
    shortLabel: 'MEX67 Helix · 110 (6EXZ)',
    sourceFamily: 'RFdiffusion 25 / Genie 2 design25',
    biology: 'C-terminal UBA-domain helix of mRNA export factor MEX67 at 110 aa (A557–571)',
    yamlKeys: ['6EXZ_long'],
    contigString: '0-95/A557-571/0-95',
    designLength: [110, 110],
    targetLength: 110,
  },

  // ── All-atom only tasks (no plain `motif_data/` counterpart in the YAML) ──
  {
    id: 'rfdaa_5aou_ra95_triad',
    pdbId: '5AOU',
    name: 'Retro-Aldolase RA95 Triad',
    shortLabel: 'RA95 Triad (5AOU)',
    sourceFamily: 'RFD4 All-Atom / Proteina-Complexa',
    biology: '3 single-residue catalytic islands of retro-aldolase RA95.5-8F on an IGPS scaffold (A1051 Tyr51, A2083 Lys83, A2110 Asn110)',
    yamlKeys: ['5AOU_AA', '5AOU_AA_TIP'],
    contigString: '40-60/A1051/20-40/A2083/20-35/A2110/100-140',
    designLength: [230, 270],
    targetLength: 270,
  },
  {
    id: 'rfdaa_5aou_ra95_tetrad',
    pdbId: '5AOU',
    name: 'Retro-Aldolase RA95 Tetrad',
    shortLabel: 'RA95 Tetrad (5AOU)',
    sourceFamily: 'RFD4 All-Atom / Proteina-Complexa',
    biology: '4-residue retro-aldolase RA95.5-8F catalytic tetrad (A1051 Tyr51, A2083 Lys83, A2110 Asn110, A2180 Tyr180)',
    yamlKeys: ['5AOU_QUAD_AA', '5AOU_QUAD_AA_TIP'],
    contigString: '40-60/A1051/20-40/A2083/20-35/A2110/60-80/A2180/40-60',
    designLength: [230, 270],
    targetLength: 270,
  },
  {
    id: 'rfdaa_7k4v_kemp',
    pdbId: '7K4V',
    name: 'Kemp Eliminase HG3.17 Site',
    shortLabel: 'HG3.17 Site (7K4V)',
    sourceFamily: 'RFD4 All-Atom / Proteina-Complexa',
    biology: '3 single-residue islands of Kemp eliminase HG3.17 on a xylanase scaffold (A44 Trp, A50 Gln, A127 Asp base)',
    yamlKeys: ['7K4V_AA', '7K4V_AA_TIP'],
    contigString: '40-50/A44/3-8/A50/70-85/A127/150-200',
    designLength: [280, 320],
    targetLength: 300,
  },
  {
    id: 'rfdaa_6vw1_ace2',
    pdbId: '6VW1',
    name: 'ACE2 Receptor Helix Pair',
    shortLabel: 'ACE2 Helices (6VW1)',
    sourceFamily: 'RFD4 All-Atom / Proteina-Complexa',
    biology: '2-island ACE2 α-helical hotspot that engages the SARS-CoV-2 receptor-binding domain (A24–42 & A64–82)',
    yamlKeys: ['6VW1_AA', '6VW1_AA_TIP'],
    contigString: '20-30/A24-42/4-10/A64-82/0-5',
    designLength: [62, 83],
    targetLength: 83,
  },
];

interface MotifSegment {
  chain: string;
  start: number;
  end: number;
}

type ContigToken =
  | { kind: 'motif'; seg: MotifSegment }
  | { kind: 'flank'; min: number; max: number };

const MOTIF_TOKEN_RE = /^([A-Za-z])(\d+)(?:-(\d+))?$/;
const FLANK_TOKEN_RE = /^(\d+)(?:-(\d+))?$/;

function parseContigString(task: MotifRawTask): ContigToken[] {
  return task.contigString
    .split('/')
    .map((t) => t.trim())
    .filter(Boolean)
    .map((tok): ContigToken => {
      const motif = MOTIF_TOKEN_RE.exec(tok);
      if (motif) {
        const start = Number(motif[2]);
        const end = motif[3] !== undefined ? Number(motif[3]) : start;
        return { kind: 'motif', seg: { chain: motif[1].toUpperCase(), start, end } };
      }
      const flank = FLANK_TOKEN_RE.exec(tok);
      if (flank) {
        const min = Number(flank[1]);
        return { kind: 'flank', min, max: flank[2] !== undefined ? Number(flank[2]) : min };
      }
      throw new Error(`Unparseable contig token "${tok}" in motif benchmark ${task.id}`);
    });
}

/**
 * Pin each variable-length scaffold flank to a single value so the contig has one deterministic
 * total length. Every flank starts at its YAML minimum, then the leftover scaffold budget is
 * handed out in proportion to each flank's headroom (`max - min`), with any rounding remainder
 * topped up last-flank-first — the same backward-remainder convention
 * `adaptBenchmarkContigForLength` uses when it re-scales these contigs for a user length.
 */
function distributeScaffold(flanks: { min: number; max: number }[], scaffoldTotal: number): number[] {
  const alloc = flanks.map((f) => f.min);
  let remaining = scaffoldTotal - alloc.reduce((s, v) => s + v, 0);

  if (remaining < 0) {
    for (let i = alloc.length - 1; i >= 0 && remaining < 0; i--) {
      const take = Math.min(alloc[i], -remaining);
      alloc[i] -= take;
      remaining += take;
    }
    return alloc;
  }

  const headroom = flanks.map((f) => Math.max(0, f.max - f.min));
  const headroomTotal = headroom.reduce((s, v) => s + v, 0);
  if (headroomTotal > 0 && remaining > 0) {
    const budget = remaining;
    for (let i = 0; i < alloc.length && remaining > 0; i++) {
      const give = Math.min(headroom[i], Math.floor((budget * headroom[i]) / headroomTotal), remaining);
      alloc[i] += give;
      remaining -= give;
    }
    for (let i = alloc.length - 1; i >= 0 && remaining > 0; i--) {
      const give = Math.min(flanks[i].max - alloc[i], remaining);
      alloc[i] += give;
      remaining -= give;
    }
  }
  // All flanks saturated but length unmet: park the remainder on the last flank so the contig
  // still sums to exactly `targetLength`.
  if (remaining > 0) alloc[alloc.length - 1] += remaining;
  return alloc;
}

function formatSegment(seg: MotifSegment): string {
  return seg.start === seg.end ? `${seg.chain}${seg.start}` : `${seg.chain}${seg.start}-${seg.end}`;
}

function buildMotifBenchmarkMeta(raw: MotifRawTask): BenchmarkTargetMeta {
  const tokens = parseContigString(raw);
  const segments = tokens.flatMap((t) => (t.kind === 'motif' ? [t.seg] : []));
  const flanks = tokens.flatMap((t) => (t.kind === 'flank' ? [{ min: t.min, max: t.max }] : []));
  if (segments.length === 0) {
    throw new Error(`Motif benchmark ${raw.id} has no motif segment`);
  }

  const motifTotal = segments.reduce((s, seg) => s + (seg.end - seg.start + 1), 0);
  const alloc = distributeScaffold(flanks, raw.targetLength - motifTotal);

  let flankIdx = 0;
  const contig = tokens
    .map((t) => {
      if (t.kind === 'motif') return `${t.seg.chain}${t.seg.start}-${t.seg.end}`;
      const len = alloc[flankIdx++];
      return len > 0 ? `${len}-${len}` : '';
    })
    .filter(Boolean)
    .join(',');

  // Crops & motif spans are listed in ascending (chain, residue) order regardless of the contig's
  // segment order — matching the pre-existing curated 1BCF entry.
  const span = [...segments]
    .sort((a, b) => a.chain.localeCompare(b.chain) || a.start - b.start)
    .map(formatSegment)
    .join(', ');

  return {
    id: raw.id,
    name: raw.name,
    shortLabel: raw.shortLabel,
    category: 'motif',
    pdbId: raw.pdbId,
    sourceFamily: raw.sourceFamily,
    approxTokens: raw.targetLength,
    tier: raw.targetLength <= 200 ? 'fast' : 'standard',
    supportsGenie3: true,
    entities: ['protein'],
    biology: raw.biology,
    cropSegments: span,
    targetLength: raw.targetLength,
    conditionParams: {
      taskPreset: 'custom',
      contig,
      motifSpan: span,
    },
  };
}

/**
 * Complete motif-scaffolding benchmark suite: the 6 pre-existing curated targets followed by the
 * 25 remaining canonical `motif_dict.yaml` tasks (30 YAML tasks total; 5 of them are already
 * covered by the curated 1PRW/1BCF/5TPN/5YUI/1YCR entries).
 */
export const MOTIF_BENCHMARK_TARGETS: BenchmarkTargetMeta[] = [
  ...CURATED_MOTIF_BENCHMARKS,
  ...MOTIF_RAW_TASKS.map(buildMotifBenchmarkMeta),
];

/** Curated motif IDs shown in the compact top-level PDB-input menu (full suite lives in the flyout). */
export const FEATURED_MOTIF_BENCHMARK_IDS: ReadonlySet<string> = new Set(
  CURATED_MOTIF_BENCHMARKS.map((b) => b.id),
);

/** Motif IDs that only exist as all-atom (`motif_data_aa/`) tasks in `motif_dict.yaml`. */
export const ALL_ATOM_MOTIF_BENCHMARK_IDS: ReadonlySet<string> = new Set(
  MOTIF_RAW_TASKS.filter((t) => t.yamlKeys.every((k) => k.includes('_AA'))).map((t) => t.id),
);

/** `motif_dict.yaml` keys (including deduped `_AA`/`_TIP` variants) covered by each benchmark ID. */
export const MOTIF_BENCHMARK_YAML_KEYS: ReadonlyMap<string, readonly string[]> = new Map([
  ['rfd_1prw_efhand', ['1PRW', '1PRW_AA', '1PRW_AA_TIP']],
  ['rfd_1bcf_diiron', ['1BCF', '1BCF_AA', '1BCF_AA_TIP']],
  ['rfd_5tpn_rsvf_v', ['5TPN', '5TPN_AA', '5TPN_AA_TIP']],
  ['g3_5yui_ca2', ['5YUI', '5YUI_AA', '5YUI_AA_TIP']],
  ['rfd_1ycr_p53', ['1YCR', '1YCR_AA', '1YCR_AA_TIP']],
  ...MOTIF_RAW_TASKS.map((t) => [t.id, t.yamlKeys] as const),
]);
