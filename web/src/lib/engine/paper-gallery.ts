export interface PaperWalkthroughStep {
  title: string;
  body: string;
  nodeId: string;
  tab: 'peek' | 'params';
  outputPort?: string;
  focusResidue?: string;
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
      'Alexander et al. (PNAS 2009): Click ▶ Run to compare ESMFold2 against experimental NMR 2LHD in a 2-step walkthrough.',
    tools: ['ESMFold2', '3D Compare'],
    walkthrough: [
      {
        title: 'Compare predicted vs. experimental fold',
        body: 'Experimental NMR GB98 (2LHD, cyan) is a 4β+α sheet-and-helix fold, while ESMFold2 (orange) stays trapped in a 3-helix bundle (~9.6 Å away, pLDDT ~51) — illustrating the fold-switching blind spot.',
        nodeId: 'cmp',
        tab: 'peek',
      },
      {
        title: 'Try flipping residue 45 back (Y→L)',
        body: 'In the sequence box on the right, change residue 45 from Y back to L (GA98) and click Run — ESMFold2 predicts the same 3-helix fold, but confidence jumps from pLDDT ~51 to ~74.',
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
      'Ostrem et al. (Nature 2013): Click ▶ Run for a 3-step walkthrough contrasting stability, evolution, and switch dynamics on H-Ras (1CRR).',
    tools: ['ESM-C', 'ThermoMPNN', 'Dyna-1'],
    walkthrough: [
      {
        title: 'Stability: cancer drivers stay folded',
        body: 'In ThermoMPNN, buried hydrophobic core sites (V7, L19, F82) paint red (destabilizing, +1.6 to +2.8 kcal/mol), while oncogenic G12C/D/V and Q61H/L near Mg²⁺·GDP are stabilizing (ΔΔG < 0).',
        nodeId: 'thermo',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A12',
      },
      {
        title: 'Evolution: G12 and Q61 are strongly forbidden',
        body: 'In ESM-C\'s mutational heatmap, those exact G12 and Q61 hotspots show deep evolutionary intolerance (LLR −8 to −13): they preserve fold stability while breaking GTPase turn-off.',
        nodeId: 'esmc',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A12',
      },
      {
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
      'Roscoe et al. (JMB 2013): Click ▶ Run for a 3-step walkthrough of evolutionary tolerance, recognition patches, and C-terminal dynamics on ubiquitin (1D3Z).',
    tools: ['ESM-C', 'Dyna-1'],
    walkthrough: [
      {
        title: 'Top substitutions rediscover yeast ubiquitin',
        body: 'Scroll to the top-ranked mutations in the ESM-C table below: without any MSA, ESM-C places natural human-to-yeast substitutions (P19S #1, E24D #5, A28S #18) at the very top of 1,444 mutants.',
        nodeId: 'esmc',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A19',
      },
      {
        title: 'Essential recognition & linkage patch (I44 / K48)',
        body: 'Inspect positions I44, K48, and L73 in the L×20 heatmap — Roscoe et al.\'s indispensable ubiquitin recognition and poly-ubiquitin linkage sites are deeply intolerant across nearly every substitution.',
        nodeId: 'esmc',
        tab: 'peek',
        outputPort: 'mutations',
        focusResidue: 'A44',
      },
      {
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
