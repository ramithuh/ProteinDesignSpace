export type PaperWalkthroughPhase = 'setup' | 'run' | 'result';
export type PaperSpotlightTarget = 'canvas' | 'run' | 'viewer' | 'inspector';

export interface PaperMutationHighlight {
  residues: string[];
  sort?: 'best' | 'worst';
  query?: string;
  limit?: number;
}

export interface PaperWalkthroughStep {
  phase: PaperWalkthroughPhase;
  spotlightTarget: PaperSpotlightTarget;
  title: string;
  body: string;
  /** Result-step navigation target on the graph/inspector. */
  nodeId?: string;
  tab?: 'peek' | 'params';
  outputPort?: string;
  focusResidue?: string;
  mutationHighlight?: PaperMutationHighlight;
}

export interface PaperInsightExample {
  id: string;
  title: string;
  citation: string;
  finding: string;
  canvasNote: string;
  tools: string[];
  walkthrough: PaperWalkthroughStep[];
  /** Verified compact `#g=...` workflow hash (alphanumeric-ending, live-tested). */
  hash: string;
}

export const PAPER_INSIGHT_EXAMPLES: PaperInsightExample[] = [
  {
    id: 'paper-fold-flip-ga98-gb98',
    title: '1-Mutation Fold Flip (GA98 → GB98)',
    citation: 'Alexander et al., PNAS 2009',
    finding:
      'A single L45→Y45 substitution flips a 56-residue protein from a 3-helix bundle into a 4β+α fold (2LHD) — a classic fold-switching test for structure predictors.',
    canvasNote:
      'Alexander et al. (PNAS 2009): Compare ESMFold2 against experimental NMR 2LHD on a 1-mutation fold switch.',
    tools: ['ESMFold2', '3D Compare'],
    walkthrough: [
      {
        phase: 'setup',
        spotlightTarget: 'canvas',
        title: 'What we\'re testing: 1-mutation fold switch',
        body: 'This pipeline sends the GB98 sequence (L45→Y45) into ESMFold2 and aligns its prediction against the experimental NMR structure 2LHD (4β+α) in 3D Compare — testing whether single-sequence folding catches a single-residue fold flip.',
      },
      {
        phase: 'run',
        spotlightTarget: 'run',
        title: 'Click Run to fold & compare',
        body: 'Run ESMFold2 and 3D Compare locally on your GPU to overlay the predicted fold against experimental NMR 2LHD.',
      },
      {
        phase: 'result',
        spotlightTarget: 'viewer',
        title: 'Compare predicted vs. experimental fold',
        body: 'Experimental NMR GB98 (2LHD, cyan) is a 4β+α sheet-and-helix fold, while ESMFold2 (orange) stays trapped in a 3-helix bundle (~9.6 Å away, pLDDT ~51) — illustrating the fold-switching blind spot.',
        nodeId: 'cmp',
        tab: 'peek',
      },
      {
        phase: 'result',
        spotlightTarget: 'inspector',
        title: 'Try flipping residue 45 back (Y→L)',
        body: 'In the sequence box here, change residue 45 from Y back to L (GA98) and click Run — ESMFold2 predicts the same 3-helix fold, but confidence jumps from pLDDT ~51 to ~74.',
        nodeId: 'seq_gb',
        tab: 'params',
      },
    ],
    hash: '#g=ZZBdaoNAFIW3IueppTdgjbapb5aYVMYUSgaLiAQTxxCIxjgGCiGvXUB30r70vQvIIrqSMpofQp9mOBzu9927RQE7iiDFejKfgrAGbbFMpmIJG8PHh94kNC2QKmxEMROwwXnIfM9_9tmLw1zX8ZjrB31nyB2XhQPme86zw3jgDoNXHrK-6zE-4AF3saNmonblm9bv-0doWteIKYLIDBAyRc5XaUMWmdHJEll3urqeg5Anb-ODgYR9S8hlMlqlSkcuivlSgCBknq2WqTGuRalK1o7gDgxtkMha6-r6qIEVedVuWh55sFGm084iVWE69VLYMPynvvJVr_bzrbXe5v7rZv_ZSs_yEoQZ4pggLm8oT6Z02O2cXCjIutrM6k2l7Nt5lchE1dz5fJj_rVMiG3yp8KZOPTOmqHv8mDoZ90ZM0V2bxDvtDw',
  },
  {
    id: 'paper-oncogenic-ras-1crr',
    title: 'Oncogenic Ras Without Unfolding',
    citation: 'Ostrem et al., Nature 2013',
    finding:
      'Oncogenic G12 and Q61 mutations break Ras GTPase signaling while keeping H-Ras (1CRR) folded and dynamic — unlike buried core mutations that destabilize the fold.',
    canvasNote:
      'Ostrem et al. (Nature 2013): Contrast stability, evolution, and switch dynamics on H-Ras (1CRR).',
    tools: ['ESM-C', 'ThermoMPNN', 'Dyna-1'],
    walkthrough: [
      {
        phase: 'setup',
        spotlightTarget: 'canvas',
        title: 'What we\'re testing: cancer drivers vs. fold stability',
        body: 'H-Ras (1CRR) fans out in parallel to ThermoMPNN (stability ΔΔG), ESM-C (evolutionary constraint), and Dyna-1 (backbone dynamics) — testing whether oncogenic G12/Q61 mutations preserve fold stability while breaking evolutionarily conserved switch function.',
      },
      {
        phase: 'run',
        spotlightTarget: 'run',
        title: 'Click Run to score stability, evolution & dynamics',
        body: 'Run all three models locally on WebGPU to compare stability, mutational fitness, and switch-pocket dynamics across H-Ras.',
      },
      {
        phase: 'result',
        spotlightTarget: 'viewer',
        title: 'Stability: cancer drivers stay folded',
        body: 'In ThermoMPNN, buried hydrophobic core sites (V7, L19, F82) paint red (destabilizing, +1.6 to +2.8 kcal/mol), while oncogenic G12C/D/V and Q61H/L near Mg²⁺·GDP are stabilizing (ΔΔG < 0).',
        nodeId: 'thermo',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A12',
        mutationHighlight: {
          residues: ['A7', 'A19', 'A82', 'A12', 'A61'],
          sort: 'worst',
          query: 'V7, L19, F82, G12, Q61',
          limit: 20,
        },
      },
      {
        phase: 'result',
        spotlightTarget: 'inspector',
        title: 'Evolution: G12 and Q61 are strongly forbidden',
        body: 'In ESM-C\'s mutational heatmap and filtered table here, those exact G12 and Q61 hotspots show deep evolutionary intolerance (LLR −8 to −13): they preserve fold stability while breaking GTPase turn-off.',
        nodeId: 'esmc',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A12',
        mutationHighlight: {
          residues: ['A12', 'A61'],
          sort: 'worst',
          query: 'G12, Q61',
          limit: 20,
        },
      },
      {
        phase: 'result',
        spotlightTarget: 'viewer',
        title: 'Dynamics: P-loop & Switch II pocket',
        body: 'Dyna-1 (sequence + 3D structure) highlights conformational exchange in the P-loop and Switch II pocket (Q61) — the dynamic cryptic pocket targeted by covalent K-Ras(G12C) inhibitors.',
        nodeId: 'dyna',
        tab: 'peek',
        outputPort: 'track',
        focusResidue: 'A61',
      },
    ],
    hash: '#g=hVCxDoIwFPwVcvMjASHEsBkXXVmbDoVWdGghbRkM4bvc_TJTRNPEwendu9y9u7wZBjVjsMKBMIJm6EEq1Bhlm95kIGV7lqiRH5sGC-GUNsIlz0eyEpwYlNMd6D3C7q_K6gG0AT0aEw63F9H5wR763qpe-BCilTCplD2W4JN3I0DryME5QUXdnLdT5yerPkkxFdy_sm-Pf8ItOJLx8IyasTKjKuPEihjsyv2GyqLifHkB',
  },
  {
    id: 'paper-ubiquitin-evolution-1d3z',
    title: 'Ubiquitin Conservation & Dynamics',
    citation: 'Roscoe et al., JMB 2013',
    finding:
      'Mutational scanning of ubiquitin (1D3Z) revealed strict constraint at recognition sites (I44, K48, L73) alongside tolerance for natural human-to-yeast substitutions and a flexible C-terminal tail.',
    canvasNote:
      'Roscoe et al. (JMB 2013): Inspect evolutionary tolerance, recognition patches, and C-terminal dynamics on ubiquitin (1D3Z).',
    tools: ['ESM-C', 'Dyna-1'],
    walkthrough: [
      {
        phase: 'setup',
        spotlightTarget: 'canvas',
        title: 'What we\'re testing: conservation vs. tail flexibility',
        body: 'NMR human ubiquitin (1D3Z) feeds into ESM-C (zero-shot scan of all 1,444 single mutations) and Dyna-1 (sequence + 3D structure dynamics) — testing whether they recover natural yeast substitutions, the essential I44/K48 patch, and the flexible conjugation tail.',
      },
      {
        phase: 'run',
        spotlightTarget: 'run',
        title: 'Click Run to scan mutations & dynamics',
        body: 'Execute ESM-C and Dyna-1 locally on WebGPU, then step through the three literature findings in the outputs and 3D viewer.',
      },
      {
        phase: 'result',
        spotlightTarget: 'inspector',
        title: 'Top substitutions rediscover yeast ubiquitin',
        body: 'Look at the highlighted rows in the ESM-C table here: without any MSA, ESM-C places all three natural human-to-yeast substitutions (P19S #1, E24D #5, A28S #18) at the very top of 1,444 mutants.',
        nodeId: 'esmc',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A19',
        mutationHighlight: {
          residues: ['A19', 'A24', 'A28'],
          sort: 'best',
          query: '',
          limit: 20,
        },
      },
      {
        phase: 'result',
        spotlightTarget: 'inspector',
        title: 'Essential recognition & linkage patch (I44 / K48)',
        body: 'Inspect the highlighted I44, K48, and L73 columns and table below — Roscoe et al.\'s indispensable ubiquitin recognition and poly-ubiquitin linkage sites are deeply intolerant across nearly every substitution.',
        nodeId: 'esmc',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A44',
        mutationHighlight: {
          residues: ['A44', 'A48', 'A73'],
          sort: 'worst',
          query: 'I44, K48, L73',
          limit: 20,
        },
      },
      {
        phase: 'result',
        spotlightTarget: 'viewer',
        title: 'Flexible C-terminal conjugation tail (RLRGG)',
        body: 'In Dyna-1 (sequence + 3D structure), ubiquitin\'s folded β-grasp core sits cool and rigid, while the disordered C-terminal RLRGG tail (R74–G76) lights up flexible for E1/E2 conjugation.',
        nodeId: 'dyna',
        tab: 'peek',
        outputPort: 'track',
        focusResidue: 'A74',
      },
    ],
    hash: '#g=ZY8_C8IwFMS_SrhJ4RVSW0SyqoNDJ3ExZGhNhiCt_ZdBSj-Xu59MEkoJON3jx93xbkIDISVc1YHQgibUL20g0OoqsdpDXV00BNJTdsdMuFW2c3a0Dft-WICKJMxQP0CLrCV1OTyN9qnztUiOLOO8YJsi0G3I6XdTgoKkUIpgon-GsXeP0fVmbY6QT__blr7IpvwuIWXOac8VySw-dvlBqfkH',
  },
];
