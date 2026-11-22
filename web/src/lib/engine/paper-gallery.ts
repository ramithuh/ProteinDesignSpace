export interface PaperInsightExample {
  id: string;
  title: string;
  citation: string;
  finding: string;
  canvasNote: string;
  tools: string[];
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
      'Alexander et al. (PNAS 2009): L45→Y45 flips GA98 (3α) to GB98 (4β+α). Compare ESMFold2 against experimental NMR 2LHD in 3D Compare.',
    tools: ['ESMFold2', '3D Compare'],
    hash: '#g=ZZBdaoNAFIW3IueppTdgjbapb5aYVMYUSgaLiAQTxxCIxjgGCiGvXUB30r70vQvIIrqSMpofQp9mOBzu9927RQE7iiDFejKfgrAGbbFMpmIJG8PHh94kNC2QKmxEMROwwXnIfM9_9tmLw1zX8ZjrB31nyB2XhQPme86zw3jgDoNXHrK-6zE-4AF3saNmonblm9bv-0doWteIKYLIDBAyRc5XaUMWmdHJEll3urqeg5Anb-ODgYR9S8hlMlqlSkcuivlSgCBknq2WqTGuRalK1o7gDgxtkMha6-r6qIEVedVuWh55sFGm084iVWE69VLYMPynvvJVr_bzrbXe5v7rZv_ZSs_yEoQZ4pggLm8oT6Z02O2cXCjIutrM6k2l7Nt5lchE1dz5fJj_rVMiG3yp8KZOPTOmqHv8mDoZ90ZM0V2bxDvtDw',
  },
  {
    id: 'paper-oncogenic-ras-1crr',
    title: 'Oncogenic Ras Without Unfolding',
    citation: 'Ostrem et al., Nature 2013',
    finding:
      'Oncogenic G12 and Q61 mutations break Ras GTPase signaling while keeping H-Ras (1CRR) folded and dynamic — unlike buried core mutations that destabilize the fold.',
    canvasNote:
      'Ostrem et al. (Nature 2013): Contrast oncogenic G12/Q61 hotspots vs buried core mutations across ESM-C, ThermoMPNN, and Dyna-1 on H-Ras (1CRR).',
    tools: ['ESM-C', 'ThermoMPNN', 'Dyna-1'],
    hash: '#g=hVCxDoIwFPwVcvMjASHEsBkXXVmbDoVWdGghbRkM4bvc_TJTRNPEwendu9y9u7wZBjVjsMKBMIJm6EEq1Bhlm95kIGV7lqiRH5sGC-GUNsIlz0eyEpwYlNMd6D3C7q_K6gG0AT0aEw63F9H5wR763qpe-BCilTCplD2W4JN3I0DryME5QUXdnLdT5yerPkkxFdy_sm-Pf8ItOJLx8IyasTKjKuPEihjsyv2GyqLifHkB',
  },
  {
    id: 'paper-ubiquitin-evolution-1d3z',
    title: 'Ubiquitin Conservation & Dynamics',
    citation: 'Roscoe et al., JMB 2013',
    finding:
      'Mutational scanning of ubiquitin (1D3Z) revealed strict constraint at recognition sites (I44, K48, L73) alongside tolerance for natural human-to-yeast substitutions and a flexible C-terminal tail.',
    canvasNote:
      'Roscoe et al. (JMB 2013): Inspect top ESM-C substitutions (P19S, E24D, A28S) and Dyna-1 C-terminal tail flexibility on NMR ubiquitin (1D3Z).',
    tools: ['ESM-C', 'Dyna-1'],
    hash: '#g=ZY8_C8IwFMS_SrhJ4RVSW0SyqoNDJ3ExZGhNhiCt_ZdBSj-Xu59MEkoJON3jx93xbkIDISVc1YHQgibUL20g0OoqsdpDXV00BNJTdsdMuFW2c3a0Dft-WICKJMxQP0CLrCV1OTyN9qnztUiOLOO8YJsi0G3I6XdTgoKkUIpgon-GsXeP0fVmbY6QT__blr7IpvwuIWXOac8VySw-dvlBqfkH',
  },
];
