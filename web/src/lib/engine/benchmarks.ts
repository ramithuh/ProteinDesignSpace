// Curated protein design benchmark targets sized for browser-native WebGPU.
// Compiled from AME, RFdiffusion 25, MotifBench / Genie 3 (RSV-F & BinderBench),
// BindCraft / BoltzGen / Adaptyv Bio, and RFD4-Proteina.

export type BenchmarkCategory = 'motif' | 'binder' | 'enzyme-ligand';

export interface BenchmarkTargetMeta {
  id: string;
  name: string;
  shortLabel: string;
  category: BenchmarkCategory;
  pdbId: string;
  sourceFamily: string;
  approxTokens: number;
  tier: 'fast' | 'standard';
  supportsGenie3: boolean;
  entities: ('protein' | 'ligand' | 'dna')[];
  biology: string;
  /** Optional segment crop applied to the raw RCSB structure so only the benchmark domain/motif is loaded. */
  cropSegments?: string;
  /** Canonical conditioning parameters applied to the Conditioning block in 1 click. */
  conditionParams: {
    taskPreset: 'custom' | string;
    contig: string;
    lockChain?: string;
    hotspots?: string;
    motifSpan?: string;
    unindexedMotif?: boolean;
    hbondMode?: 'none' | 'ACCEPTOR' | 'DONOR' | 'BOTH';
    hbondSelection?: string;
  };
}

export const BENCHMARK_TARGETS: BenchmarkTargetMeta[] = [
  // ── Motif Scaffolding & Catalytic Sites ──
  {
    id: 'rfd_5tpn_rsvf_v',
    name: 'RSV-F Site V Epitope',
    shortLabel: 'RSV-F Site V (5TPN)',
    category: 'motif',
    pdbId: '5TPN',
    sourceFamily: 'RFdiffusion / MotifBench',
    approxTokens: 75,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Viral prefusion neutralizing helix–strand epitope (A163–181)',
    cropSegments: 'A163-181',
    conditionParams: {
      taskPreset: 'custom',
      contig: '25-25,A163-181,30-30',
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
    biology: '3-island catalytic His₃ Zn²⁺ binding site',
    cropSegments: 'A93-97, A118-120, A198-200',
    conditionParams: {
      taskPreset: 'custom',
      contig: '15-15,A93-97,20-20,A118-120,20-20,A198-200,10-10',
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
      contig: '15-15,A18-25,20-20,A47-54,20-20,A92-99,20-20,A123-130,18-18',
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
      contig: '35-35,A55-55,35-35,A99-99,45-45,A190-192,30-30',
      motifSpan: 'A55, A99, A190-192',
    },
  },

  // ── Protein–Protein (PPI) Binder Targets ──
  {
    id: 'pdl1_5o45',
    name: 'PD-L1 (IgV Domain)',
    shortLabel: 'PD-L1 Binder (5O45)',
    category: 'binder',
    pdbId: '5O45',
    sourceFamily: 'RFdiffusion / RFD4 / BindCraft / Genie 3',
    approxTokens: 186,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Immune checkpoint PD-L1 IgV (A17–132) with hydrophobic hotspots',
    cropSegments: 'A17-132',
    conditionParams: {
      taskPreset: 'custom',
      contig: 'A17-132,/0,70-70',
      lockChain: 'A',
      hotspots: 'A54, A56, A66, A115',
    },
  },
  {
    id: 'mdm2_1ycr',
    name: 'MDM2 (p53 Binding Cleft)',
    shortLabel: 'MDM2 Binder (1YCR)',
    category: 'binder',
    pdbId: '1YCR',
    sourceFamily: 'RFdiffusion Classic',
    approxTokens: 145,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Compact 85-aa oncology target (A25–109) with deep p53 pocket',
    cropSegments: 'A25-109',
    conditionParams: {
      taskPreset: 'custom',
      contig: 'A25-109,/0,60-60',
      lockChain: 'A',
      hotspots: 'A54, A61, A62, A67, A72, A93',
    },
  },
  {
    id: 'trka_1www',
    name: 'TrkA (Domain 5)',
    shortLabel: 'TrkA Binder (1WWW)',
    category: 'binder',
    pdbId: '1WWW',
    sourceFamily: 'RFdiffusion / Genie 3',
    approxTokens: 171,
    tier: 'fast',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Neurotrophin receptor TrkA D5 (X282–382)',
    cropSegments: 'X282-382',
    conditionParams: {
      taskPreset: 'custom',
      contig: 'X282-382,/0,70-70',
      lockChain: 'X',
      hotspots: 'X294, X296, X333',
    },
  },
  {
    id: 'insulinr_4zxb',
    name: 'Insulin Receptor (L1)',
    shortLabel: 'InsulinR Binder (4ZXB)',
    category: 'binder',
    pdbId: '4ZXB',
    sourceFamily: 'RFdiffusion / Genie 3',
    approxTokens: 220,
    tier: 'standard',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Insulin receptor L1 domain (E6–155)',
    cropSegments: 'E6-155',
    conditionParams: {
      taskPreset: 'custom',
      contig: 'E6-155,/0,70-70',
      lockChain: 'E',
      hotspots: 'E64, E88, E96',
    },
  },
  {
    id: 'bhrf1_2wh6',
    name: 'EBV BHRF1',
    shortLabel: 'BHRF1 Binder (2WH6)',
    category: 'binder',
    pdbId: '2WH6',
    sourceFamily: 'Genie 3 BinderBench',
    approxTokens: 237,
    tier: 'standard',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Epstein–Barr virus anti-apoptotic Bcl-2 homolog (A2–158)',
    cropSegments: 'A2-158',
    conditionParams: {
      taskPreset: 'custom',
      contig: 'A2-158,/0,80-80',
      lockChain: 'A',
      hotspots: 'A65, A74, A77, A82, A85, A93',
    },
  },
  {
    id: 'egfr_d3_1yy9',
    name: 'EGFR (Domain III)',
    shortLabel: 'EGFR D3 Binder (1YY9)',
    category: 'binder',
    pdbId: '1YY9',
    sourceFamily: 'Adaptyv Bio / BindCraft / BoltzGen',
    approxTokens: 266,
    tier: 'standard',
    supportsGenie3: true,
    entities: ['protein'],
    biology: 'Canonical Adaptyv Bio oncology binder target (A310–505)',
    cropSegments: 'A310-505',
    conditionParams: {
      taskPreset: 'custom',
      contig: 'A310-505,/0,70-70',
      lockChain: 'A',
      hotspots: 'A355, A357, A382, A384, A412',
    },
  },

  // ── AME Enzymes, Cofactors & DNA (All-Atom RFD4) ──
  {
    id: 'ame_1ctt_cda',
    name: 'AME: Cytidine Deaminase',
    shortLabel: 'AME: CDA Zn²⁺ (1CTT)',
    category: 'enzyme-ligand',
    pdbId: '1CTT',
    sourceFamily: 'AME / RFD2–4',
    approxTokens: 160,
    tier: 'fast',
    supportsGenie3: false,
    entities: ['protein', 'ligand'],
    biology: 'Unindexed 4-residue catalytic Zn²⁺ + transition-state analog (DHZ)',
    cropSegments: 'A102, A104, A129, A132, A:ZN, A:DHZ',
    conditionParams: {
      taskPreset: 'custom',
      contig: '150-150',
      motifSpan: 'A102, A104, A129, A132',
      unindexedMotif: true,
    },
  },
  {
    id: 'ame_1os7_taud',
    name: 'AME: TauD Dioxygenase',
    shortLabel: 'AME: TauD Fe²⁺ (1OS7)',
    category: 'enzyme-ligand',
    pdbId: '1OS7',
    sourceFamily: 'AME / RFD2–4',
    approxTokens: 175,
    tier: 'fast',
    supportsGenie3: false,
    entities: ['protein', 'ligand'],
    biology: '2-His-1-carboxylate facial triad coordinating Fe²⁺ + α-ketoglutarate',
    cropSegments: 'A99, A101, A255, A270, A:FE2, A:AKG, A:TAU',
    conditionParams: {
      taskPreset: 'custom',
      contig: '165-165',
      motifSpan: 'A99, A101, A255, A270',
      unindexedMotif: true,
    },
  },
  {
    id: 'sm_7c7m_sam',
    name: 'SAM Cofactor Pocket',
    shortLabel: 'SAM Pocket (7C7M)',
    category: 'enzyme-ligand',
    pdbId: '7C7M',
    sourceFamily: 'RFD4 / RFdiffusionAA',
    approxTokens: 143,
    tier: 'fast',
    supportsGenie3: false,
    entities: ['protein', 'ligand'],
    biology: 'De novo pocket around S-adenosylmethionine (SAM) with H-bonds',
    cropSegments: 'A:SAM',
    conditionParams: {
      taskPreset: 'custom',
      contig: '110-110',
      hbondMode: 'ACCEPTOR',
      hbondSelection: 'A/SAM/*/*',
    },
  },
  {
    id: 'rfd4_7m5w_bdna',
    name: 'B-DNA Binding Protein',
    shortLabel: 'B-DNA Binder (7M5W)',
    category: 'enzyme-ligand',
    pdbId: '7M5W',
    sourceFamily: 'RFD4 / RFD3',
    approxTokens: 128,
    tier: 'fast',
    supportsGenie3: false,
    entities: ['protein', 'dna'],
    biology: 'Sequence-specific dsDNA groove recognition (TGAATGAA)',
    cropSegments: 'B, C',
    conditionParams: {
      taskPreset: 'custom',
      contig: '90-90',
    },
  },
];

export function getBenchmarkById(id: string): BenchmarkTargetMeta | undefined {
  return BENCHMARK_TARGETS.find((b) => b.id === id);
}

/**
 * Adapt a curated benchmark contig to a user-requested length without breaking fixed motif spans
 * or chain-break (`/0`) binder syntax:
 * - Binder (`category === 'binder'` or contains `/0`): updates designed binder length token (after `/0`) to `L-L`.
 * - Enzyme/Ligand (`category === 'enzyme-ligand'`): updates bare length token to `L-L`.
 * - Motif (`category === 'motif'`): preserves fixed motif spans (e.g. `B19-27`) and scales scaffold flanks
 *   proportionally so total scaffold + motif residues equal `L` (remainder allocated from last flank backward).
 */
export function adaptBenchmarkContigForLength(
  contig: string,
  targetLength: number,
  category: BenchmarkCategory,
): string {
  if (!contig || !Number.isFinite(targetLength) || targetLength <= 0) {
    return contig;
  }
  const L = Math.max(10, Math.min(512, Math.round(targetLength)));
  const tokens = contig
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
  if (tokens.length === 0) return contig;

  interface ParsedBare {
    idx: number;
    origLen: number;
    suffix: string;
  }

  const bareTokens: ParsedBare[] = [];
  let motifTotal = 0;
  const chainBreakIdx = tokens.indexOf('/0');

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (tok === '/0') continue;
    const bareMatch = /^(\d+)(?:-(\d+))?([PRDN])?$/i.exec(tok);
    if (bareMatch) {
      const min = Number(bareMatch[1]);
      const max = bareMatch[2] !== undefined ? Number(bareMatch[2]) : min;
      const suffix = bareMatch[3] ? bareMatch[3].toUpperCase() : '';
      bareTokens.push({ idx: i, origLen: Math.max(1, Math.round((min + max) / 2)), suffix });
      continue;
    }
    const spanMatch = /^([A-Za-z_])(-?\d+)(?:-(-?\d+))?$/.exec(tok);
    if (spanMatch) {
      const start = Number(spanMatch[2]);
      const end = spanMatch[3] !== undefined ? Number(spanMatch[3]) : start;
      motifTotal += Math.max(1, end - start + 1);
    }
  }

  if (bareTokens.length === 0) {
    return contig;
  }

  const out = [...tokens];

  if (category === 'binder' || chainBreakIdx >= 0) {
    const afterBreak =
      chainBreakIdx >= 0 ? bareTokens.filter((b) => b.idx > chainBreakIdx) : bareTokens;
    const targets = afterBreak.length > 0 ? afterBreak : bareTokens;
    if (targets.length === 1) {
      const b = targets[0];
      out[b.idx] = `${L}-${L}${b.suffix}`;
      return out.join(',');
    }
    const origSum = Math.max(1, targets.reduce((s, b) => s + b.origLen, 0));
    const alloc = targets.map((b) => Math.max(4, Math.floor((L * b.origLen) / origSum)));
    let diff = L - alloc.reduce((s, v) => s + v, 0);
    for (let k = alloc.length - 1; diff > 0; k = k === 0 ? alloc.length - 1 : k - 1) {
      alloc[k] += 1;
      diff--;
    }
    while (diff < 0) {
      let trimmed = false;
      for (let k = alloc.length - 1; k >= 0 && diff < 0; k--) {
        if (alloc[k] > 4) {
          alloc[k] -= 1;
          diff++;
          trimmed = true;
        }
      }
      if (!trimmed) break;
    }
    for (let k = 0; k < targets.length; k++) {
      out[targets[k].idx] = `${alloc[k]}-${alloc[k]}${targets[k].suffix}`;
    }
    return out.join(',');
  }

  if (category === 'enzyme-ligand') {
    if (bareTokens.length === 1) {
      const b = bareTokens[0];
      out[b.idx] = `${L}-${L}${b.suffix}`;
      return out.join(',');
    }
    const origSum = Math.max(1, bareTokens.reduce((s, b) => s + b.origLen, 0));
    const alloc = bareTokens.map((b) => Math.max(4, Math.floor((L * b.origLen) / origSum)));
    let diff = L - alloc.reduce((s, v) => s + v, 0);
    for (let k = alloc.length - 1; diff > 0; k = k === 0 ? alloc.length - 1 : k - 1) {
      alloc[k] += 1;
      diff--;
    }
    for (let k = 0; k < bareTokens.length; k++) {
      out[bareTokens[k].idx] = `${alloc[k]}-${alloc[k]}${bareTokens[k].suffix}`;
    }
    return out.join(',');
  }

  // category === 'motif'
  const minFlank = 4;
  if (bareTokens.length === 1) {
    const b = bareTokens[0];
    const flank = Math.max(minFlank, L - motifTotal);
    out[b.idx] = `${flank}-${flank}${b.suffix}`;
    return out.join(',');
  }

  const desiredScaffoldTotal = Math.max(bareTokens.length * minFlank, L - motifTotal);
  const origScaffoldTotal = Math.max(1, bareTokens.reduce((s, b) => s + b.origLen, 0));
  const alloc = bareTokens.map((b) =>
    Math.max(minFlank, Math.floor((desiredScaffoldTotal * b.origLen) / origScaffoldTotal)),
  );
  let diff = desiredScaffoldTotal - alloc.reduce((s, v) => s + v, 0);
  for (let k = alloc.length - 1; diff > 0; k = k === 0 ? alloc.length - 1 : k - 1) {
    alloc[k] += 1;
    diff--;
  }
  while (diff < 0) {
    let trimmed = false;
    for (let k = alloc.length - 1; k >= 0 && diff < 0; k--) {
      if (alloc[k] > minFlank) {
        alloc[k] -= 1;
        diff++;
        trimmed = true;
      }
    }
    if (!trimmed) break;
  }
  for (let k = 0; k < bareTokens.length; k++) {
    out[bareTokens[k].idx] = `${alloc[k]}-${alloc[k]}${bareTokens[k].suffix}`;
  }
  return out.join(',');
}
