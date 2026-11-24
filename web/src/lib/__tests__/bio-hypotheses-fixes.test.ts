import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  BUFFER_OR_SOLVENT_RESNAMES,
  MODIFIED_AMINO_ACID_RESNAMES,
  MONOATOMIC_ION_RESNAMES,
  classifyHetatmCompId,
  evaluateProteinOnlyStructurePolicy,
  hasUnprotonatedOrganicLigands,
  inspectUnprotonatedOrganicLigands,
  tokenizeCifLine,
} from '$lib/protein-ops/hetatm-policy';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  buildMotifLineage,
  resolveGenie3CompiledInput,
  resolveLocalFoldEffectiveConfig,
} from '$lib/engine/conditioning';
import {
  compileAntiFoldDesignPlan,
  type NumberedChainInfo,
  type OrderedStructureResidue,
} from '$lib/engine/mpnn-spec';
import {
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import { compileNodeInputPreview } from '$lib/engine/input-preview';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import type { SequenceSet, StructureArtifact, WorkflowGraph } from '$lib/engine/types';

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

const AA1_TO_AA3: Record<string, string> = {
  A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE',
  G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
  M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG',
  S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
};

function padL(s: string, n: number): string {
  return s.length >= n ? s : ' '.repeat(n - s.length) + s;
}

function fmtAtom(
  serial: number,
  atomName: string,
  resName: string,
  chain: string,
  resNum: number,
  x: number,
  y: number,
  z: number,
  rec: 'ATOM  ' | 'HETATM' = 'ATOM  ',
  elem?: string,
): string {
  const atomField = atomName.length >= 4 ? atomName.slice(0, 4) : ` ${atomName.padEnd(3, ' ')}`;
  const el = elem ?? atomName.replace(/[0-9']/g, '').slice(0, 1);
  return (
    `${rec}${padL(String(serial), 5)} ${atomField} ${padL(resName, 3)} ${chain.slice(0, 1)}` +
    `${padL(String(resNum), 4)}    ` +
    `${padL(x.toFixed(3), 8)}${padL(y.toFixed(3), 8)}${padL(z.toFixed(3), 8)}` +
    `  1.00 20.00          ${padL(el, 2)}`
  );
}

function buildChainPdb(
  chains: Array<{ chain: string; seq: string; startNum?: number }>,
  extraHetatmLines: string[] = [],
): string {
  const lines: string[] = [];
  let serial = 1;
  let x = 0;
  for (const ch of chains) {
    const start = ch.startNum ?? 1;
    for (let i = 0; i < ch.seq.length; i++) {
      const resName = AA1_TO_AA3[ch.seq[i]] ?? 'ALA';
      const num = start + i;
      lines.push(fmtAtom(serial++, 'N', resName, ch.chain, num, x - 1.2, 0.5, 0.0));
      lines.push(fmtAtom(serial++, 'CA', resName, ch.chain, num, x, 0.0, 0.0));
      lines.push(fmtAtom(serial++, 'C', resName, ch.chain, num, x + 1.2, -0.5, 0.2));
      lines.push(fmtAtom(serial++, 'O', resName, ch.chain, num, x + 1.5, -1.6, 0.4));
      x += 3.8;
    }
    x += 10.0;
  }
  for (const h of extraHetatmLines) lines.push(h);
  lines.push('END');
  return lines.join('\n') + '\n';
}

function makeArtifact(pdbText: string, overrides: Partial<StructureArtifact> = {}): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id: overrides.id ?? 'struct-test-1',
    label: overrides.label ?? 'test.pdb',
    sourceModelId: overrides.sourceModelId ?? 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    insCodes: parsed.insCodes,
    sourceResidues: parsed.sourceResidues,
    atomPresence: parsed.atomPresence,
    observedBackboneMask: parsed.observedBackboneMask,
    diagnostics: parsed.diagnostics,
    atomGranularity: 'all-atom',
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. Primitive 2 — Unified Chemical Entity & HETATM Policy (#79 & #82)
// ---------------------------------------------------------------------------

describe('Primitive 2 — Unified Chemical Entity & HETATM Policy', () => {
  it('classifies HETATMs into 4 strictly disjoint classes', () => {
    for (const r of MODIFIED_AMINO_ACID_RESNAMES) {
      expect(BUFFER_OR_SOLVENT_RESNAMES.has(r)).toBe(false);
      expect(MONOATOMIC_ION_RESNAMES.has(r)).toBe(false);
    }
    for (const r of BUFFER_OR_SOLVENT_RESNAMES) {
      expect(MONOATOMIC_ION_RESNAMES.has(r)).toBe(false);
    }

    expect(classifyHetatmCompId('MSE')).toBe('modified-amino-acid');
    expect(classifyHetatmCompId('SEP')).toBe('modified-amino-acid');
    expect(classifyHetatmCompId('TPO')).toBe('modified-amino-acid');
    expect(classifyHetatmCompId('ACE')).toBe('modified-amino-acid');

    expect(classifyHetatmCompId('HOH')).toBe('buffer-or-solvent');
    expect(classifyHetatmCompId('SO4')).toBe('buffer-or-solvent');
    expect(classifyHetatmCompId('GOL')).toBe('buffer-or-solvent');
    expect(classifyHetatmCompId('EDO')).toBe('buffer-or-solvent');

    expect(classifyHetatmCompId('ZN')).toBe('monoatomic-ion');
    expect(classifyHetatmCompId('CA')).toBe('monoatomic-ion');
    expect(classifyHetatmCompId('MG')).toBe('monoatomic-ion');
    expect(classifyHetatmCompId('FE2')).toBe('monoatomic-ion');
    expect(classifyHetatmCompId('NA')).toBe('monoatomic-ion');
    expect(classifyHetatmCompId('CL')).toBe('monoatomic-ion');

    expect(classifyHetatmCompId('SAM')).toBe('organic-ligand');
    expect(classifyHetatmCompId('BTN')).toBe('organic-ligand');
    expect(classifyHetatmCompId('DHZ')).toBe('organic-ligand');
    expect(classifyHetatmCompId('AKG')).toBe('organic-ligand');
    expect(classifyHetatmCompId('ATP')).toBe('organic-ligand');
    expect(classifyHetatmCompId('HEM')).toBe('organic-ligand');
  });

  it('hasUnprotonatedOrganicLigands uses quote-aware mmCIF parsing and never false-positives on metalloproteins, buffers, or MSE', () => {
    expect(tokenizeCifLine(`HETATM 10 C "C4'" . SAM A 1 501 ? 1.0 2.0 3.0 1.0 20.0 C 1`)).toContain("C4'");

    // Metalloprotein mmCIF with ZN, CA, MG, FE2, SO4, MSE, HOH -> 0 organic ligands -> false
    const metalloCif = [
      'data_METALLO',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.pdbx_PDB_model_num',
      'ATOM   1 N  N   ALA A 1',
      'HETATM 2 ZN ZN  ZN  B 1',
      'HETATM 3 CA CA  CA  C 1',
      'HETATM 4 MG MG  MG  D 1',
      'HETATM 5 FE FE  FE2 E 1',
      'HETATM 6 S  S   SO4 F 1',
      'HETATM 7 O  O1  SO4 F 1',
      'HETATM 8 SE SE  MSE A 1',
      'HETATM 9 O  O   HOH G 1',
    ].join('\n');
    expect(hasUnprotonatedOrganicLigands(metalloCif)).toBe(false);
    const metalloReport = inspectUnprotonatedOrganicLigands(metalloCif);
    expect(metalloReport.organicLigands).toEqual([]);
    expect(metalloReport.ignoredNonOrganic).toEqual(expect.arrayContaining(['ZN', 'CA', 'MG', 'FE2', 'SO4', 'MSE', 'HOH']));

    // Unprotonated organic ligand (SAM with quoted atom names) alongside ZN -> only SAM flagged!
    const unprotCif = [
      'data_SAM_HOLO',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.pdbx_PDB_model_num',
      `HETATM 1 ZN ZN    ZN  1`,
      `HETATM 2 C  "C4'" SAM 1`,
      `HETATM 3 O  "O4'" SAM 1`,
      `HETATM 4 N  N1    SAM 1`,
    ].join('\n');
    const unprotReport = inspectUnprotonatedOrganicLigands(unprotCif);
    expect(unprotReport.hasUnprotonated).toBe(true);
    expect(unprotReport.unprotonatedLigands).toEqual(['SAM']);
    expect(unprotReport.ignoredNonOrganic).toEqual(['ZN']);

    // Protonated organic ligand (SAM with explicit H atom) -> false!
    const protCif = unprotCif + `\nHETATM 5 H "H4'" SAM 1`;
    expect(hasUnprotonatedOrganicLigands(protCif)).toBe(false);

    // LASErMPNN bio check: silent on ZN metalloprotein, warns on unprotonated SAM
    const znPdb = buildChainPdb(
      [{ chain: 'A', seq: 'MKTAYIAKQRQISFVK' }],
      [fmtAtom(900, 'ZN', 'ZN', 'A', 501, 5, 5, 5, 'HETATM', 'ZN')],
    );
    const samPdb = buildChainPdb(
      [{ chain: 'A', seq: 'MKTAYIAKQRQISFVK' }],
      [
        fmtAtom(900, 'C1', 'SAM', 'A', 501, 5, 5, 5, 'HETATM', 'C'),
        fmtAtom(901, 'N1', 'SAM', 'A', 501, 6, 5, 5, 'HETATM', 'N'),
      ],
    );
    const makeLaserGraph = (pdbText: string): WorkflowGraph => ({
      id: 'laser-h-check',
      name: 'Laser H Check',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText } },
        { id: 'laser', adapterId: 'lasermpnn', params: {} },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'laser', toPort: 'structure' }],
    });
    expect(
      validateBioCapabilities(makeLaserGraph(znPdb)).some(
        (i) => i.code === 'lasermpnn-unprotonated-ligand',
      ),
    ).toBe(false);
    expect(
      validateBioCapabilities(makeLaserGraph(samPdb)).some(
        (i) => i.code === 'lasermpnn-unprotonated-ligand' && i.nodeId === 'laser',
      ),
    ).toBe(true);
  });

  it('applies ONE unified Genie 3 policy across conditioning, bio-capabilities, and worker (raw 1PRW passes silently)', () => {
    // Raw 1PRW (Calmodulin with crystal CA ions, 80 protein residues)
    const raw1prwPdb = buildChainPdb(
      [{ chain: 'A', seq: 'A'.repeat(80), startNum: 1 }],
      [
        fmtAtom(900, 'CA', 'CA', 'A', 201, 10, 5, 5, 'HETATM', 'CA'),
        fmtAtom(901, 'CA', 'CA', 'A', 202, 20, 5, 5, 'HETATM', 'CA'),
      ],
    );
    const raw1prwStruct = makeArtifact(raw1prwPdb, {
      id: 'raw-1prw',
      label: '1PRW.cif',
    });
    expect(raw1prwStruct.diagnostics?.ligandNames).toContain('CA');

    // 1. resolveGenie3CompiledInput MUST succeed cleanly on raw 1PRW (fixing #79 trap!)
    const compiled = resolveGenie3CompiledInput(
      { taskMode: 'motif', motifSpan: 'A16-35, A52-71', length: 80 },
      raw1prwStruct,
    );
    expect(compiled.mode).toBe('motif');
    expect(compiled.compiledSpec?.length).toBeGreaterThanOrEqual(40);

    // 2. Layer-1 & Layer-2 are completely silent on raw 1PRW crystal CA
    const g1prw: WorkflowGraph = {
      id: 'g-1prw',
      name: '1PRW Genie3',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'pdb-id', pdbId: '1PRW', pdbText: raw1prwPdb } },
        { id: 'g3', adapterId: 'genie3', params: { taskMode: 'motif', motifSpan: 'A16-35, A52-71', length: 80 } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' }],
    };
    const issues1prw = validateBioCapabilities(g1prw, { pdb: { structure: raw1prwStruct } });
    expect(issues1prw.some((i) => i.code === 'genie3-unsupported-entities')).toBe(false);
    const rt1prw = validateRuntimeNodeInputs(g1prw.nodes[1], { structure: raw1prwStruct });
    expect(rt1prw.warnings).toEqual([]);

    // 3. Uncropped holo complex with >=3 protein residues + SAM warns in Layer 1 (with 1-click crop fix)
    //    and NEVER throws in resolveGenie3CompiledInput or Layer 2!
    const holoPdb = buildChainPdb(
      [{ chain: 'A', seq: 'A'.repeat(40) }],
      [fmtAtom(900, 'C1', 'SAM', 'A', 501, 5, 5, 5, 'HETATM', 'C')],
    );
    const holoStruct = makeArtifact(holoPdb, { id: 'holo-sam', label: '9SAM.cif' });
    expect(() =>
      resolveGenie3CompiledInput(
        { taskMode: 'motif', motifSpan: 'A5-15', length: 60 },
        holoStruct,
      ),
    ).not.toThrow();

    // 4. Hard-block ONLY incompatible all-atom benchmark OR <3 protein residues
    expect(
      evaluateProteinOnlyStructurePolicy({
        benchmark: { id: 'ame_1ctt_cda', shortLabel: 'AME: CDA', supportsGenie3: false },
        proteinResidueCount: 50,
      }).action,
    ).toBe('hard-block');
    expect(
      evaluateProteinOnlyStructurePolicy({
        proteinResidueCount: 2,
        ligandNames: ['SAM'],
      }).action,
    ).toBe('hard-block');
    expect(() =>
      resolveGenie3CompiledInput(
        { taskMode: 'motif', motifSpan: 'A1-2', length: 60 },
        { ...holoStruct, length: 2 },
      ),
    ).toThrow(/at least 3 protein/i);
  });
});

// ---------------------------------------------------------------------------
// 2. Critical CT-01 & #80 — Shared buildMotifLineage + ProtPardelle
// ---------------------------------------------------------------------------

describe('CT-01 & #80 — buildMotifLineage & ProtPardelle motifGuidance', () => {
  it('emits 0-based motifIndices, 1-based fixedResidueIndices, and aligned nativeMotifCoords', () => {
    const upCa = new Float32Array([
      1, 2, 3,   // src 0
      4, 5, 6,   // src 1
      7, 8, 9,   // src 2
      10, 11, 12, // src 3
    ]);
    const lineage = buildMotifLineage([10, 11, 12], [0, 2, 3], upCa);
    expect(lineage.motifIndices).toEqual([10, 11, 12]);
    expect(lineage.fixedResidueIndices).toEqual([11, 12, 13]);
    expect(lineage.nativeMotifCoords).toEqual([
      [1, 2, 3],
      [7, 8, 9],
      [10, 11, 12],
    ]);
  });

  it('wires motifGuidance in registry.ts and protpardelle.worker.ts (conditioning + runHash) and keeps 0-based motifIndices', () => {
    const ppManifest = ADAPTER_REGISTRY.protpardelle;
    const guidanceField = ppManifest.paramSchema?.find((f) => f.key === 'motifGuidance');
    expect(guidanceField).toBeDefined();
    expect(guidanceField?.defaultValue).toBe('replacement');
    expect(guidanceField?.modes).toEqual(['motif']);
    expect(guidanceField?.options?.map((o) => o.value)).toEqual([
      'replacement',
      'replacement+reconstruction',
      'crop-only',
    ]);

    const workerSrc = readFileSync(
      resolve(process.cwd(), 'src/lib/plugins/protpardelle/protpardelle.worker.ts'),
      'utf8',
    );
    expect(workerSrc).toContain('buildMotifLineage');
    expect(workerSrc).toContain('motifIndices: motifLineage.motifIndices');
    expect(workerSrc).toContain('fixedResidueIndices: motifLineage.fixedResidueIndices');
    expect(workerSrc).not.toContain('motifIndices: motifOneBased');
    expect(workerSrc).toMatch(/runHash[\s\S]*?motifGuidance/);
    expect(workerSrc).toMatch(/conditioning:\s*\{[\s\S]*?replacement:[\s\S]*?reconstruction:/);
  });
});

// ---------------------------------------------------------------------------
// 3. #77 — compileAntiFoldDesignPlan 1:1 Preview <-> Worker Parity (1N8Z & 1MEL)
// ---------------------------------------------------------------------------

describe('#77 — compileAntiFoldDesignPlan 1:1 preview <-> worker parity on 1N8Z & 1MEL', () => {
  // Realistic sequentially numbered 1N8Z-like antibody: A = VL, B = VH, C = HER2 antigen
  // VL hallmarks: C at idx 22 (res 23), WYQ at idx 34 (res 35), YYC at idx 85-87 (res 88), FGQG at idx 97 (res 98)
  const vlSeq =
    'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKR';
  // VH hallmarks: C at idx 21 (res 22), WVR at idx 35 (res 36), YYC at idx 93-95 (res 96, shifted by CDR2 -> non-canonical sequential!), WGQG at idx 109
  const vhSeq =
    'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS';
  // Non-Ig antigen chain C (HER2)
  const her2Seq = 'TQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQ';

  // Realistic 1MEL-like complex: L = Lysozyme (non-Ig antigen with chain ID 'L'!), M = VHH nanobody
  const lysozymeSeq =
    'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL';
  const vhhSeq =
    'DVQLQESGGGSVQAGGSLRLSCAASGYTIGPYCMGWFRQAPGKEREGVAAINMGGGITYYADSVKGRFTISQDNAKNTVYLLMNSLEPEDTAIYYCAADSTIYASYYECGHGLSTGGYGYDSWGQGTQVTVSS';

  it('matches vendor antifold-esmif1 regionMask 1:1 on sequentially numbered 1N8Z (A=VL, B=VH, C=HER2)', async () => {
    const pdb1n8z = buildChainPdb([
      { chain: 'A', seq: vlSeq, startNum: 1 },
      { chain: 'B', seq: vhSeq, startNum: 1 },
      { chain: 'C', seq: her2Seq, startNum: 1 },
    ]);
    const struct = makeArtifact(pdb1n8z, { id: '1n8z-test', label: '1N8Z.cif' });
    const antiShared = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/antifold-esmif1/shared/index.js')).href
    );
    const vendorParsed = antiShared.parseStructure(pdb1n8z);

    // Build chainInfos & orderedResidues identically to worker
    const chainInfos: NumberedChainInfo[] = vendorParsed.chains.map(
      (c: { id: string; residues: Array<{ number: number; insertion: string }>; sequence: string }) => ({
        id: c.id,
        residues: c.residues.map((r, idx) => ({
          number: r.number,
          insertion: r.insertion ?? '',
          aa1: c.sequence[idx] ?? 'X',
        })),
      }),
    );
    const orderedResidues: OrderedStructureResidue[] = vendorParsed.chains.flatMap(
      (c: { residues: Array<{ chain: string; number: number; insertion: string }> }) =>
        c.residues.map((r) => ({ chain: r.chain, number: r.number, insertionCode: r.insertion ?? '' })),
    );

    const plan = compileAntiFoldDesignPlan({
      chainInfos,
      orderedResidues,
      variant: 'antifold',
      samplingMode: 'independent',
      targetChainSpec: '',
      numberingParam: 'auto',
      regionsSpec: 'CDR1 CDR2 CDR3',
      fixedPositionsSpec: '',
      unlockInherited: false,
    });

    expect(plan.heavyChain).toBe('B');
    expect(plan.lightChain).toBe('A');
    expect(plan.remappedChains).toContain('B');
    expect(plan.designedCount).toBeGreaterThan(20);

    // Vendor worker prepareInput + regionMask
    const prepared = antiShared.prepareInput(vendorParsed, plan.combineOpts);
    const vendorMask = antiShared.regionMask(prepared.residues, {
      regions: plan.combineOpts.regions,
      numbering: plan.combineOpts.numbering,
      heavyChain: plan.combineOpts.heavyChain,
      lightChain: plan.combineOpts.lightChain,
    });
    const vendorDesignableKeys = new Set<string>();
    for (let i = 0; i < prepared.residues.length; i++) {
      const r = prepared.residues[i];
      if (r && vendorMask[i] === 1) {
        vendorDesignableKeys.add(`${r.chain}:${r.number}:${r.insertion ?? ''}`);
      }
    }

    // Studio input-preview.ts MUST highlight the EXACT same residues 1:1!
    const graph: WorkflowGraph = {
      id: 'g-1n8z',
      name: '1N8Z AntiFold',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: pdb1n8z } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold', regions: 'CDR1 CDR2 CDR3', numbering: 'auto' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const preview = compileNodeInputPreview(graph, 'af', { pdb: { structure: struct } });
    const previewDesignableKeys = new Set<string>();
    for (const ch of preview.chains) {
      for (const r of ch.residues) {
        if (r.roles.includes('designable')) {
          previewDesignableKeys.add(`${r.authChain}:${r.authSeqId}:${r.insCode}`);
        }
      }
    }
    expect(previewDesignableKeys).toEqual(vendorDesignableKeys);
    // Antigen chain C must have 0 designable residues
    expect([...previewDesignableKeys].some((k) => k.startsWith('C:'))).toBe(false);
  });

  it('matches vendor antifold-esmif1 regionMask 1:1 on 1MEL (L=Lysozyme antigen, M=VHH nanobody)', async () => {
    const pdb1mel = buildChainPdb([
      { chain: 'L', seq: lysozymeSeq, startNum: 1 },
      { chain: 'M', seq: vhhSeq, startNum: 1 },
    ]);
    const struct = makeArtifact(pdb1mel, { id: '1mel-test', label: '1MEL.cif' });
    const antiShared = await import(
      pathToFileURL(resolve(process.cwd(), 'static/vendor/antifold-esmif1/shared/index.js')).href
    );
    const vendorParsed = antiShared.parseStructure(pdb1mel);

    const chainInfos: NumberedChainInfo[] = vendorParsed.chains.map(
      (c: { id: string; residues: Array<{ number: number; insertion: string }>; sequence: string }) => ({
        id: c.id,
        residues: c.residues.map((r, idx) => ({
          number: r.number,
          insertion: r.insertion ?? '',
          aa1: c.sequence[idx] ?? 'X',
        })),
      }),
    );
    const orderedResidues: OrderedStructureResidue[] = vendorParsed.chains.flatMap(
      (c: { residues: Array<{ chain: string; number: number; insertion: string }> }) =>
        c.residues.map((r) => ({ chain: r.chain, number: r.number, insertionCode: r.insertion ?? '' })),
    );

    const plan = compileAntiFoldDesignPlan({
      chainInfos,
      orderedResidues,
      variant: 'antifold',
      samplingMode: 'independent',
      targetChainSpec: '',
      numberingParam: 'auto',
      regionsSpec: 'CDR1 CDR2 CDR3',
      fixedPositionsSpec: '',
      unlockInherited: false,
    });

    expect(plan.heavyChain).toBe('M');
    expect(plan.lightChain).toBeUndefined(); // Lysozyme chain 'L' MUST NOT be mistaken for VL!

    const prepared = antiShared.prepareInput(vendorParsed, plan.combineOpts);
    const vendorMask = antiShared.regionMask(prepared.residues, {
      regions: plan.combineOpts.regions,
      numbering: plan.combineOpts.numbering,
      heavyChain: plan.combineOpts.heavyChain,
      lightChain: plan.combineOpts.lightChain,
    });
    const vendorDesignableKeys = new Set<string>();
    for (let i = 0; i < prepared.residues.length; i++) {
      const r = prepared.residues[i];
      if (r && vendorMask[i] === 1) {
        vendorDesignableKeys.add(`${r.chain}:${r.number}:${r.insertion ?? ''}`);
      }
    }

    const graph: WorkflowGraph = {
      id: 'g-1mel',
      name: '1MEL AntiFold',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'custom', pdbText: pdb1mel } },
        { id: 'af', adapterId: 'antifold-esmif1', params: { variant: 'antifold', regions: 'CDR1 CDR2 CDR3', numbering: 'auto' } },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'af', toPort: 'structure' }],
    };
    const preview = compileNodeInputPreview(graph, 'af', { pdb: { structure: struct } });
    const previewDesignableKeys = new Set<string>();
    for (const ch of preview.chains) {
      for (const r of ch.residues) {
        if (r.roles.includes('designable')) {
          previewDesignableKeys.add(`${r.authChain}:${r.authSeqId}:${r.insCode}`);
        }
      }
    }
    expect(previewDesignableKeys).toEqual(vendorDesignableKeys);
    expect(previewDesignableKeys.size).toBeGreaterThan(15);
    expect([...previewDesignableKeys].every((k) => k.startsWith('M:'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 4. CT-05 — Shared resolveLocalFoldEffectiveConfig
// ---------------------------------------------------------------------------

describe('CT-05 — resolveLocalFoldEffectiveConfig parity across worker & preview', () => {
  it('resolves af2-multimer tolerance=0.5 and ESMFold2 64-step floor on holo or >2048 aa', () => {
    expect(resolveLocalFoldEffectiveConfig({ modelKey: 'monomer', params: {} }).tolerance).toBe(0);
    expect(resolveLocalFoldEffectiveConfig({ modelKey: 'multimer', params: {} }).tolerance).toBe(0.5);
    expect(resolveLocalFoldEffectiveConfig({ modelKey: 'multimer-3', params: {} }).tolerance).toBe(0.5);
    expect(
      resolveLocalFoldEffectiveConfig({ modelKey: 'multimer', params: { tolerance: 0.2 } }).tolerance,
    ).toBe(0.2);

    const apoEf2 = resolveLocalFoldEffectiveConfig({
      modelKey: 'ef2-fast-300m',
      params: { esmfold2Steps: 15 },
      maxResidueLength: 120,
    });
    expect(apoEf2.effectiveEsmfold2Steps).toBe(15);
    expect(apoEf2.esmfold2StepsAdjusted).toBe(false);

    const holoEf2 = resolveLocalFoldEffectiveConfig({
      modelKey: 'ef2-fast-300m',
      params: { esmfold2Steps: 15, ligandCcdCodes: 'ATP' },
      maxResidueLength: 120,
    });
    expect(holoEf2.effectiveEsmfold2Steps).toBe(64);
    expect(holoEf2.esmfold2StepsAdjusted).toBe(true);

    const longEf2 = resolveLocalFoldEffectiveConfig({
      modelKey: 'ef2-fast-600m',
      params: { esmfold2Steps: 32 },
      maxResidueLength: 2100,
    });
    expect(longEf2.effectiveEsmfold2Steps).toBe(64);
    expect(longEf2.esmfold2StepsAdjusted).toBe(true);

    // Input preview surfaces effective 64 steps on holo ESMFold2
    const seqSet: SequenceSet = {
      kind: 'SequenceSet',
      id: 'seq-1',
      parentStructureId: 'none',
      sourceModelId: 'sequence-input',
      candidates: [
        {
          id: 'c1',
          sequence: 'MKTAYIAKQRQISFVK',
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
        },
      ],
    };
    const graph: WorkflowGraph = {
      id: 'g-ef2-holo',
      name: 'EF2 Holo',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { fastaText: '>c1\nMKTAYIAKQRQISFVK' } },
        {
          id: 'fold',
          adapterId: 'localfold-af2',
          params: { model: 'ef2-fast-300m', esmfold2Steps: 15, ligandCcdCodes: 'ATP' },
        },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' }],
    };
    const preview = compileNodeInputPreview(graph, 'fold', { seq: { sequences: seqSet } });
    expect(preview.payload.compiledSummary.esmfold2Steps).toBe(64);

    // Worker encodes effectiveEsmfold2Steps into runHash and FoldedStructure.id
    const lfSrc = readFileSync(
      resolve(process.cwd(), 'src/lib/plugins/localfold-af2/localfold.worker.ts'),
      'utf8',
    );
    expect(lfSrc).toContain('resolveLocalFoldEffectiveConfig');
    expect(lfSrc).toContain('esmfold2Steps: effectiveEsmfold2Steps');
    expect(lfSrc).toContain('d${effectiveEsmfold2Steps}');
  });
});

// ---------------------------------------------------------------------------
// 5. CT-08 — compileSchemaFallbackPreview & 'schema' Error Promotion
// ---------------------------------------------------------------------------

describe('CT-08 — compileSchemaFallbackPreview & schema error promotion', () => {
  it('resolves honest ready / pending-upstream / error availability for unbranched adapters instead of perpetual partial', () => {
    // Unconditional generators should report 'ready'
    for (const adapterId of [
      'protpardelle',
      'rfdiffusion',
      'boltzgen',
      'evodiff',
      'la-proteina',
      'proteina',
      'sid-protein',
      'genie2',
      'frameflow',
      'foldflow2',
    ]) {
      const g: WorkflowGraph = {
        id: `g-${adapterId}`,
        name: adapterId,
        nodes: [{ id: 'n1', adapterId, params: {} }],
        edges: [],
      };
      const prev = compileNodeInputPreview(g, 'n1', {});
      expect(prev.availability, `${adapterId} should be ready, not partial`).toBe('ready');
    }

    // Disconnected dyna1 / esmc / top-k-filter / structure-compare should report 'error' (not 'partial')
    for (const adapterId of ['esmc', 'dyna1', 'top-k-filter', 'structure-compare']) {
      const g: WorkflowGraph = {
        id: `g-${adapterId}-disc`,
        name: adapterId,
        nodes: [{ id: 'n1', adapterId, params: {} }],
        edges: [],
      };
      const prev = compileNodeInputPreview(g, 'n1', {});
      expect(prev.availability, `disconnected ${adapterId} should be error`).toBe('error');
    }

    // Wired-unrun protpardelle motif node reports 'pending-upstream'
    const gPending: WorkflowGraph = {
      id: 'g-pp-pending',
      name: 'PP Pending',
      nodes: [
        { id: 'rfd', adapterId: 'rfd4-proteina', params: { mode: 'unconditional', length: 60 } },
        { id: 'pp', adapterId: 'protpardelle', params: { taskMode: 'motif', length: 80 } },
      ],
      edges: [{ fromNode: 'rfd', fromPort: 'structure', toNode: 'pp', toPort: 'structure' }],
    };
    const pendingPrev = compileNodeInputPreview(gPending, 'pp', {});
    expect(pendingPrev.availability).toBe('pending-upstream');
    expect(pendingPrev.missingUpstreamNodeIds).toEqual(['rfd']);

    // Schema validation error promotes availability from 'ready' -> 'error'
    const gBadSchema: WorkflowGraph = {
      id: 'g-bad-schema',
      name: 'Bad Schema',
      nodes: [{ id: 'pp', adapterId: 'protpardelle', params: { taskMode: 'unconditional', length: 9999 } }],
      edges: [],
    };
    const errPrev = compileNodeInputPreview(gBadSchema, 'pp', {});
    expect(errPrev.issues.some((i) => i.source === 'schema' && i.severity === 'error')).toBe(true);
    expect(errPrev.availability).toBe('error');
  });
});

// ---------------------------------------------------------------------------
// 6. Generic field.warnBelow, Disjoint motifSpan, WF-13, & Dyna-1 seq-only
// ---------------------------------------------------------------------------

describe('Generic field.warnBelow, disjoint motifSpan, WF-13 defaults, and Dyna-1 warnings', () => {
  it('evaluates field.warnBelow generically from ADAPTER_REGISTRY respecting warnBelow.modes', () => {
    const gGenie2: WorkflowGraph = {
      id: 'g-g2',
      name: 'Genie2 Low Steps',
      nodes: [{ id: 'g2', adapterId: 'genie2', params: { steps: 200 } }],
      edges: [],
    };
    const g2Warn = validateBioCapabilities(gGenie2).find(
      (i) => i.code === 'param-warn-below-genie2-steps',
    );
    expect(g2Warn).toBeDefined();
    expect(g2Warn?.fix?.label).toBe('Set 1000');
    expect(g2Warn?.fix?.paramsPatch).toEqual({ steps: 1000 });

    // La-Proteina SDE warns at 100 steps, ODE does NOT warn at 100 steps
    const gLpSde: WorkflowGraph = {
      id: 'g-lp-sde',
      name: 'LP SDE',
      nodes: [{ id: 'lp', adapterId: 'la-proteina', params: { mode: 'sde', steps: 100 } }],
      edges: [],
    };
    const lpSdeWarn = validateBioCapabilities(gLpSde).find(
      (i) => i.code === 'param-warn-below-la-proteina-steps',
    );
    expect(lpSdeWarn).toBeDefined();
    expect(lpSdeWarn?.fix?.paramsPatch).toEqual({ steps: 400 });

    const gLpOde: WorkflowGraph = {
      id: 'g-lp-ode',
      name: 'LP ODE',
      nodes: [{ id: 'lp', adapterId: 'la-proteina', params: { mode: 'ode', steps: 100 } }],
      edges: [],
    };
    expect(
      validateBioCapabilities(gLpOde).some((i) => i.code === 'param-warn-below-la-proteina-steps'),
    ).toBe(false);

    // Proteina sc warns at 100 steps, ode does NOT warn at 100 steps
    const gProtSc: WorkflowGraph = {
      id: 'g-prot-sc',
      name: 'Prot sc',
      nodes: [{ id: 'pr', adapterId: 'proteina', params: { mode: 'sc', steps: 100 } }],
      edges: [],
    };
    expect(
      validateBioCapabilities(gProtSc).some((i) => i.code === 'param-warn-below-proteina-steps'),
    ).toBe(true);

    const gProtOde: WorkflowGraph = {
      id: 'g-prot-ode',
      name: 'Prot ode',
      nodes: [{ id: 'pr', adapterId: 'proteina', params: { mode: 'ode', steps: 100 } }],
      edges: [],
    };
    expect(
      validateBioCapabilities(gProtOde).some((i) => i.code === 'param-warn-below-proteina-steps'),
    ).toBe(false);
  });

  it('warns when a motif node motifSpan is completely disjoint from its own contig or upstream cropSegments', () => {
    const gDisjoint: WorkflowGraph = {
      id: 'g-disjoint-motif',
      name: 'Disjoint Motif',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          params: { mode: 'pdb-id', pdbId: '1PRW', benchmarkId: 'rfd_1prw_efhand', cropSegments: 'A16-35, A52-71' },
        },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          params: {
            mode: 'motif',
            motifSpan: 'A1-10',
            contig: '10-10,A16-35,16-16,A52-71,10-10',
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' }],
    };
    const issues = validateBioCapabilities(gDisjoint);
    expect(issues.some((i) => i.code === 'motif-span-disjoint-from-contig' && i.nodeId === 'rfd')).toBe(true);
    const cropWarn = issues.find(
      (i) => i.code === 'motif-span-disjoint-from-crop' && i.nodeId === 'rfd',
    );
    expect(cropWarn).toBeDefined();
    expect(cropWarn?.fix?.paramsPatch?.motifSpan).toBe('A16-35, A52-71');
  });

  it('emits gentle dyna1-seq-only-calibration warning and keeps dyna1.worker.ts seq-only track tag clean', () => {
    const gSeqOnly: WorkflowGraph = {
      id: 'g-dyna-seq',
      name: 'Dyna1 SeqOnly',
      nodes: [
        { id: 'seq', adapterId: 'sequence-input', params: { fastaText: '>s1\nMKTAYIAKQRQISFVK' } },
        { id: 'dyna', adapterId: 'dyna1', params: { inputMode: 'auto' } },
      ],
      edges: [{ fromNode: 'seq', fromPort: 'sequences', toNode: 'dyna', toPort: 'sequences' }],
    };
    expect(
      validateBioCapabilities(gSeqOnly).some(
        (i) => i.code === 'dyna1-seq-only-calibration' && i.nodeId === 'dyna',
      ),
    ).toBe(true);

    const dynaSrc = readFileSync(
      resolve(process.cwd(), 'src/lib/plugins/dyna1/dyna1.worker.ts'),
      'utf8',
    );
    expect(dynaSrc).toContain("'seq-only'");
  });

  it('uses schema default length (WF-13) instead of hardcoded 100 for BoltzGen token ceilings', () => {
    // BoltzGen schema default length is 48 (not 100); 350 + 48 = 398 (>384 AF3 ceiling) vs 350 + 100 = 450,
    // and standalone 48 aa monomer should NOT exceed ceiling or inflate to 100.
    const gBoltz: WorkflowGraph = {
      id: 'g-boltz-wf13',
      name: 'BoltzGen Default Len',
      nodes: [{ id: 'bg', adapterId: 'boltzgen', params: { taskMode: 'unconditional' } }],
      edges: [],
    };
    const prev = compileNodeInputPreview(gBoltz, 'bg', {});
    expect(prev.budget.estimatedTokens).toBe(48);
  });
});
