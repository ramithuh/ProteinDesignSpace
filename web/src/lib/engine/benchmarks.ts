// Curated protein design benchmark targets sized for browser-native WebGPU.
// Compiled from AME, RFdiffusion 25, MotifBench / Genie 3 (RSV-F & BinderBench),
// BindCraft / BoltzGen / Adaptyv Bio, and RFD4-Proteina.

import { AME_BENCHMARK_TARGETS, FEATURED_AME_BENCHMARK_IDS } from './ame-benchmarks';
import {
  ALL_ATOM_MOTIF_BENCHMARK_IDS,
  FEATURED_MOTIF_BENCHMARK_IDS,
  MOTIF_BENCHMARK_TARGETS,
  MOTIF_BENCHMARK_YAML_KEYS,
} from './motif-benchmarks';

export { AME_BENCHMARK_TARGETS, FEATURED_AME_BENCHMARK_IDS };
export {
  ALL_ATOM_MOTIF_BENCHMARK_IDS,
  FEATURED_MOTIF_BENCHMARK_IDS,
  MOTIF_BENCHMARK_TARGETS,
  MOTIF_BENCHMARK_YAML_KEYS,
};

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
  /** Optional M-CSA catalytic site entry ID (e.g. 'M0097') for AME enzyme benchmarks. */
  mcsaId?: string;
  /** Optional ligand CCD codes present in this benchmark (e.g. ['DHZ', 'ZN']). */
  ligands?: string[];
  /** Optional number of catalytic motif residues. */
  motifResidueCount?: number;
  /** Optional canonical designed scaffold length in residues. */
  targetLength?: number;
  /** Canonical conditioning parameters applied to the Conditioning block in 1 click. */
  conditionParams: {
    taskPreset: 'custom' | string;
    contig: string;
    lockChain?: string;
    hotspots?: string;
    motifSpan?: string;
    motifAtoms?: string;
    unindexedMotif?: boolean;
    hbondMode?: 'none' | 'ACCEPTOR' | 'DONOR' | 'BOTH';
    hbondSelection?: string;
    hbondDonorSelection?: string;
    hbondAcceptorSelection?: string;
    rasaBuriedSelection?: string;
    piStackSelection?: string;
  };
}

export const BENCHMARK_TARGETS: BenchmarkTargetMeta[] = [
  // ── Motif Scaffolding & Catalytic Sites (6 curated + 25 `motif_dict.yaml` tasks) ──
  ...MOTIF_BENCHMARK_TARGETS,

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

  // ── AME Enzymes (41 M-CSA tasks), Cofactors & DNA (All-Atom RFD4 / RFD3) ──
  ...AME_BENCHMARK_TARGETS,
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
    biology: 'De novo pocket around S-adenosylmethionine (SAM) with polar donor/acceptor H-bonds',
    cropSegments: 'A:SAM',
    conditionParams: {
      taskPreset: 'custom',
      contig: '110-110',
      hbondDonorSelection: "A/SAM/*/[N,N6,O2',O3']",
      hbondAcceptorSelection: "A/SAM/*/[O,OXT,O4',N1,N3,N7]",
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
    biology: 'Sequence-specific 18-bp dsDNA duplex (containing 8-bp TGAATGAA Capicua core) groove recognition',
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
