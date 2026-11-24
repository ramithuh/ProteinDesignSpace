import { describe, expect, it, vi, beforeAll } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import {
  assertRfd4ConditionSelectorsMatch,
  extractRfd4TemplateContext,
  pdbTextToSplitCifText,
  remapAndValidateRfd4CustomSpec,
  validateRfd4InputsAgainstTemplate,
  validateRfd4TemplateSelectors,
} from '$lib/engine/conditioning';
import {
  isResidueSpecDisjointFromBenchmark,
  isResidueSpecDisjointFromIntervals,
  parseResidueIntervals,
} from '$lib/engine/benchmark-sync';
import { getBenchmarkById } from '$lib/engine/benchmarks';
import {
  validateBioCapabilities,
  validateRuntimeNodeInputs,
} from '$lib/engine/bio-capabilities';
import {
  compileWorkflowPlanToGraph,
  reconcilePlannedWorkflow,
  type BlockGraphWorkflowPlan,
} from '$lib/engine/workflow-planner';
import { parseStructureText } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type { StructureArtifact, WorkflowGraph } from '$lib/engine/types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4Mod: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4CifMod: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4SpecMod: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rfd4PrepareMod: any;

beforeAll(async () => {
  const vendorRoot = path.resolve(process.cwd(), 'static/vendor/rfd4-proteina');
  [rfd4Mod, rfd4CifMod, rfd4SpecMod, rfd4PrepareMod] = await Promise.all([
    import(pathToFileURL(path.join(vendorRoot, 'webgpu/index.js')).href),
    import(pathToFileURL(path.join(vendorRoot, 'shared/rfd4p/pipeline/cif.js')).href),
    import(pathToFileURL(path.join(vendorRoot, 'shared/rfd4p/pipeline/spec.js')).href),
    import(pathToFileURL(path.join(vendorRoot, 'shared/rfd4p/pipeline/prepare.js')).href),
  ]);
});

function buildCropped1CttPdb(): string {
  const resDefs: Array<{ resNum: number; resName: string; atoms: Array<[string, string]> }> = [
    {
      resNum: 102,
      resName: 'HIS',
      atoms: [
        ['N', 'N'],
        ['CA', 'C'],
        ['C', 'C'],
        ['O', 'O'],
        ['CB', 'C'],
        ['CG', 'C'],
        ['ND1', 'N'],
        ['CD2', 'C'],
        ['CE1', 'C'],
        ['NE2', 'N'],
      ],
    },
    {
      resNum: 104,
      resName: 'GLU',
      atoms: [
        ['N', 'N'],
        ['CA', 'C'],
        ['C', 'C'],
        ['O', 'O'],
        ['CB', 'C'],
        ['CG', 'C'],
        ['CD', 'C'],
        ['OE1', 'O'],
        ['OE2', 'O'],
      ],
    },
    {
      resNum: 129,
      resName: 'CYS',
      atoms: [
        ['N', 'N'],
        ['CA', 'C'],
        ['C', 'C'],
        ['O', 'O'],
        ['CB', 'C'],
        ['SG', 'S'],
      ],
    },
    {
      resNum: 132,
      resName: 'CYS',
      atoms: [
        ['N', 'N'],
        ['CA', 'C'],
        ['C', 'C'],
        ['O', 'O'],
        ['CB', 'C'],
        ['SG', 'S'],
      ],
    },
  ];

  const lines: string[] = [];
  let serial = 1;
  for (let i = 0; i < resDefs.length; i++) {
    const r = resDefs[i];
    const x0 = i * 4.2;
    for (let j = 0; j < r.atoms.length; j++) {
      const [nm, el] = r.atoms[j];
      const atomCol = nm.length === 4 ? nm : ` ${nm.padEnd(3, ' ')}`;
      const x = (x0 + (j % 3) * 1.1).toFixed(3).padStart(8);
      const y = (Math.floor(j / 3) * 1.2).toFixed(3).padStart(8);
      const z = ((j % 2) * 0.8).toFixed(3).padStart(8);
      lines.push(
        `ATOM  ${String(serial++).padStart(5)} ${atomCol} ${r.resName} A${String(r.resNum).padStart(4)}    ${x}${y}${z}  1.00 15.00          ${el.padStart(2)}`,
      );
    }
  }
  lines.push(
    `HETATM${String(serial++).padStart(5)} ZN    ZN A 296       2.500   2.500   1.500  1.00 10.00          ZN`,
    `HETATM${String(serial++).padStart(5)}  C1  DHZ A 295       4.000   2.500   1.500  1.00 10.00           C`,
    `HETATM${String(serial++).padStart(5)}  O4  DHZ A 295       5.200   2.500   1.500  1.00 10.00           O`,
    'END',
  );
  return lines.join('\n') + '\n';
}

function buildShifted1CttCif(): string {
  return [
    'data_1ctt_shifted',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
    'ATOM 1 N N . HIS A 1 98 ? 0.0 0.5 0.0 1.0 10.0 102 HIS A N 1',
    'ATOM 2 C CA . HIS A 1 98 ? 1.2 0.0 0.0 1.0 10.0 102 HIS A CA 1',
    'ATOM 3 C C . HIS A 1 98 ? 2.4 0.5 0.0 1.0 10.0 102 HIS A C 1',
    'ATOM 4 O O . HIS A 1 98 ? 2.4 1.7 0.0 1.0 10.0 102 HIS A O 1',
    'ATOM 5 N N . GLU A 1 100 ? 4.0 0.5 0.0 1.0 10.0 104 GLU A N 1',
    'ATOM 6 C CA . GLU A 1 100 ? 5.2 0.0 0.0 1.0 10.0 104 GLU A CA 1',
    'ATOM 7 C C . GLU A 1 100 ? 6.4 0.5 0.0 1.0 10.0 104 GLU A C 1',
    'ATOM 8 O O . GLU A 1 100 ? 6.4 1.7 0.0 1.0 10.0 104 GLU A O 1',
    'ATOM 9 N N . CYS A 1 125 ? 8.0 0.5 0.0 1.0 10.0 129 CYS A N 1',
    'ATOM 10 C CA . CYS A 1 125 ? 9.2 0.0 0.0 1.0 10.0 129 CYS A CA 1',
    'ATOM 11 C C . CYS A 1 125 ? 10.4 0.5 0.0 1.0 10.0 129 CYS A C 1',
    'ATOM 12 O O . CYS A 1 125 ? 10.4 1.7 0.0 1.0 10.0 129 CYS A O 1',
    'ATOM 13 N N . CYS A 1 128 ? 12.0 0.5 0.0 1.0 10.0 132 CYS A N 1',
    'ATOM 14 C CA . CYS A 1 128 ? 13.2 0.0 0.0 1.0 10.0 132 CYS A CA 1',
    'ATOM 15 C C . CYS A 1 128 ? 14.4 0.5 0.0 1.0 10.0 132 CYS A C 1',
    'ATOM 16 O O . CYS A 1 128 ? 14.4 1.7 0.0 1.0 10.0 132 CYS A O 1',
    'HETATM 17 ZN ZN . ZN B 2 . ? 2.5 2.5 1.5 1.0 10.0 296 ZN A ZN 1',
    'HETATM 18 C C1 . DHZ C 3 . ? 4.0 2.5 1.5 1.0 10.0 295 DHZ A C1 1',
    '#',
  ].join('\n');
}

function makeStructureFromText(rawText: string, label: string): StructureArtifact {
  const parsed = parseStructureText(rawText);
  const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(rawText);
  return {
    kind: 'StructureArtifact',
    id: `struct-${label}`,
    label,
    sourceModelId: 'pdb-input',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: isCif ? parsed.pdbText : rawText,
    ...(isCif ? { cifText: rawText } : {}),
    chainIds: parsed.chainIds,
    chainLengths: parsed.diagnostics.chainLengths,
    sourceResidues: parsed.sourceResidues,
    diagnostics: parsed.diagnostics,
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums),
  };
}

function loadSelfhostModule() {
  const serverPath = path.resolve(process.cwd(), '../server/selfhost-server.mjs');
  const req = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return req(serverPath) as any;
}

function rawHttpGetJson(port: number, reqPath: string): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, method: 'GET', path: reqPath }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c) => chunks.push(Buffer.from(c)));
      res.on('end', () =>
        resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf8') }),
      );
    });
    req.on('error', reject);
    req.end();
  });
}

describe('Issue #97: RFD4 fail-fast residue validation, per-token crop coverage, customSpecYaml remapping, & R4 stale-build guard', () => {
  it('1. fails fast with available Chain A residues (A102, A104, A129, A132) before weight download and in Layer 2 preflight when A128 is requested against cropped 1CTT', () => {
    const croppedPdb = buildCropped1CttPdb();
    const shiftedCif = buildShifted1CttCif();
    const pdbTpl = makeStructureFromText(croppedPdb, '1CTT Theozyme');
    const cifTpl = makeStructureFromText(shiftedCif, '1CTT Theozyme');

    for (const tpl of [pdbTpl, cifTpl]) {
      // Direct helper check across motifSpan, contig, motifAtoms, and hotspots
      const ctx = extractRfd4TemplateContext(tpl.cifText ?? tpl.pdbText);
      expect(() =>
        validateRfd4TemplateSelectors(ctx, {
          motifSpan: 'A102, A104, A128, A132',
          motifAtoms: 'A102:ND1,CG,CE1; A104:OE1,CD; A128:SG,CB; A132:SG,CB,CA',
          templateLabel: '1CTT Theozyme',
        }),
      ).toThrow(
        'Residue A128 not found in template "1CTT Theozyme" — available Chain A residues: A102, A104, A129, A132',
      );

      expect(() =>
        validateRfd4InputsAgainstTemplate(
          {
            mode: 'motif',
            contig: '180-180,A102,A104,A128,A132',
            unindexedMotif: true,
          },
          tpl,
        ),
      ).toThrow(/Residue A128 not found in template "1CTT Theozyme".*A102, A104, A129, A132/);

      expect(() =>
        validateRfd4InputsAgainstTemplate(
          {
            mode: 'binder',
            lockChain: 'A',
            hotspots: 'A102, A128',
          },
          tpl,
        ),
      ).toThrow(/Residue A128 not found in template "1CTT Theozyme".*A102, A104, A129, A132/);

      // Layer 2 runtime preflight check in bio-capabilities.ts
      expect(() =>
        validateRuntimeNodeInputs(
          {
            id: 'rfd4_1',
            adapterId: 'rfd4-proteina',
            label: 'RFD4-Proteina',
            params: {
              mode: 'motif',
              length: 180,
              contig: '180-180',
              motifSpan: 'A102, A104, A128, A132',
              unindexedMotif: true,
            },
          },
          { structure: tpl },
        ),
      ).toThrow(
        'Residue A128 not found in template "1CTT Theozyme" — available Chain A residues: A102, A104, A129, A132',
      );

      // Valid canonical 1CTT residues (A102, A104, A129, A132) pass cleanly
      expect(() =>
        validateRuntimeNodeInputs(
          {
            id: 'rfd4_ok',
            adapterId: 'rfd4-proteina',
            label: 'RFD4-Proteina',
            params: {
              mode: 'motif',
              length: 180,
              contig: '180-180',
              motifSpan: 'A102, A104, A129, A132',
              motifAtoms: 'A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA',
              unindexedMotif: true,
            },
          },
          { structure: tpl },
        ),
      ).not.toThrow();
    }

    // Compact range summarization on full 294-residue Chain A (A1-294)
    const full294Pdb =
      Array.from(
        { length: 294 },
        (_, i) =>
          `ATOM  ${String(i + 1).padStart(5)}  CA  ALA A${String(i + 1).padStart(4)}       0.000   0.000   0.000  1.00 10.00           C`,
      ).join('\n') + '\nEND\n';
    const fullCtx = extractRfd4TemplateContext(full294Pdb);
    expect(() =>
      validateRfd4TemplateSelectors(fullCtx, {
        motifSpan: 'A350',
        templateLabel: '1CTT Full',
      }),
    ).toThrow('Residue A350 not found in template "1CTT Full" — available Chain A residues: A1-294');

    // Verify rfd4.worker.ts calls validateRfd4InputsAgainstTemplate BEFORE ensureLoaded(ctx, needEncoder)
    const workerSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/lib/plugins/rfd4-proteina/rfd4.worker.ts'),
      'utf8',
    );
    const valIdx = workerSrc.indexOf('validateRfd4InputsAgainstTemplate(');
    const loadIdx = workerSrc.indexOf('await ensureLoaded(ctx, needEncoder)');
    expect(valIdx).toBeGreaterThan(0);
    expect(loadIdx).toBeGreaterThan(valIdx);
  });

  it('2. detects partial-overlap crop mismatches (A102, A104, A128, A132 vs cropSegments = "A102, A104, A129, A132") across benchmark-sync, bio-capabilities, and workflow-planner', () => {
    const cttBench = getBenchmarkById('ame_1ctt_cda')!;
    const efhandBench = getBenchmarkById('rfd_1prw_efhand')!;
    const pdl1Bench = getBenchmarkById('pdl1_5o45')!;

    // Partial-overlap with 1 missing token (A128) MUST be flagged as uncovered/disjoint
    const cropIntervals = parseResidueIntervals('A102, A104, A129, A132');
    expect(isResidueSpecDisjointFromIntervals('A102, A104, A128, A132', cropIntervals)).toBe(true);
    expect(isResidueSpecDisjointFromBenchmark('A102, A104, A128, A132', cttBench)).toBe(true);

    // Valid subsets & full matches remain accepted (no false positives on sub-motifs, hotspots, or ligands)
    expect(isResidueSpecDisjointFromIntervals('A102, A104, A129, A132', cropIntervals)).toBe(false);
    expect(isResidueSpecDisjointFromIntervals('A102, A132', cropIntervals)).toBe(false);
    expect(isResidueSpecDisjointFromBenchmark('A16-35', efhandBench)).toBe(false);
    expect(isResidueSpecDisjointFromBenchmark('A115', pdl1Bench)).toBe(false);
    expect(
      isResidueSpecDisjointFromIntervals('180-180,A102,A104,A129,A132,/0,B296,/0,C295', cropIntervals),
    ).toBe(false);

    // Layer 1 validateBioCapabilities warns on partial-overlap motifSpan and provides 1-click fix
    const partialMotifGraph: WorkflowGraph = {
      id: 'g-partial-crop-97',
      name: 'Partial Crop Mismatch',
      nodes: [
        {
          id: 'pdb',
          adapterId: 'pdb-input',
          label: '1CTT Theozyme',
          params: {
            mode: 'pdb-id',
            pdbId: '1CTT',
            benchmarkId: 'ame_1ctt_cda',
            cropSegments: 'A102, A104, A129, A132, A:ZN, A:DHZ',
          },
        },
        {
          id: 'rfd4',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: {
            mode: 'motif',
            length: 180,
            contig: '180-180',
            motifSpan: 'A102, A104, A128, A132',
            unindexedMotif: true,
          },
        },
      ],
      edges: [{ fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd4', toPort: 'structure' }],
    };
    const issues = validateBioCapabilities(partialMotifGraph);
    const cropIssue = issues.find(
      (i) => i.code === 'motif-span-disjoint-from-crop' && i.nodeId === 'rfd4',
    );
    expect(cropIssue).toBeDefined();
    expect(cropIssue!.message).toContain('A128');
    expect(cropIssue!.fix?.paramsPatch?.motifSpan).toBe('A102, A104, A129, A132');

    // Layer 1 also catches partial-overlap in preserved contig spans when motifSpan is omitted
    const partialContigGraph: WorkflowGraph = {
      ...partialMotifGraph,
      nodes: [
        partialMotifGraph.nodes[0],
        {
          id: 'rfd4',
          adapterId: 'rfd4-proteina',
          label: 'RFD4-Proteina',
          params: {
            mode: 'motif',
            length: 180,
            contig: '20-20,A102,20-20,A128,20-20',
          },
        },
      ],
    };
    const contigIssues = validateBioCapabilities(partialContigGraph);
    const contigCropIssue = contigIssues.find(
      (i) => i.code === 'motif-span-disjoint-from-crop' && i.nodeId === 'rfd4',
    );
    expect(contigCropIssue).toBeDefined();
    expect(contigCropIssue!.message).toContain('A128');

    // Planner repairs partial-overlap motifSpan + motifAtoms on 1CTT benchmark
    const badPlan: BlockGraphWorkflowPlan = {
      kind: 'block_graph',
      mode: 'new',
      nodes: [
        { id: 'pdb', adapter: 'pdb-input', label: '1CTT', params: { pdbId: '1CTT' } },
        {
          id: 'rfd',
          adapter: 'rfd4-proteina',
          label: 'RFD4',
          params: {
            mode: 'motif',
            motifSpan: 'A102, A104, A128, A132',
            motifAtoms: 'A102:ND1,CG,CE1; A104:OE1,CD; A128:SG,CB; A132:SG,CB,CA',
            unindexedMotif: true,
          },
        },
      ],
      edges: [['pdb', 'rfd']],
      summary: 'Scaffold 1CTT catalytic site',
    };
    const repaired = compileWorkflowPlanToGraph(
      reconcilePlannedWorkflow(badPlan, 'Scaffold 1CTT catalytic site with RFD4'),
    ).graph;
    const rfdNode = repaired.nodes.find((n) => n.id === 'rfd')!;
    expect(rfdNode.params.motifSpan).toBe('A102, A104, A129, A132');
    expect(rfdNode.params.motifAtoms).toBe(
      'A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA',
    );
  });

  it('3. remaps customSpecYaml with author residue numbers against cropped/non-1-indexed PDB via prepared.templateContext with zero "matched no atoms" crashes', () => {
    const croppedPdb = buildCropped1CttPdb();
    // Simulate prepareTemplateForRfd4: pdbTextToSplitCifText renumbers Chain A label_seq_id to 1..4
    // and splits ZN / DHZ onto distinct non-polymer chains B and C!
    const splitCif = pdbTextToSplitCifText(croppedPdb);
    const templateContext = extractRfd4TemplateContext(splitCif);
    const atomArray = rfd4CifMod.parseCifToAtomArray(splitCif, { fixArginines: true });
    rfd4PrepareMod.assignIonCharges(atomArray);

    const authorYaml = [
      'contig: "16-16,A102,A104,A129,A132,/0,A:ZN,/0,A:DHZ"',
      'conditions:',
      '  C_CRDGRP:',
      '    1:',
      '      - select: "sel(\'A/*/102/[ND1,CG,CE1]\') | sel(\'A/*/104/[OE1,CD]\') | sel(\'A/*/129/[SG,CB]\') | sel(\'A/*/132/[SG,CB,CA]\') | sel(\'A/ZN/*/*\') | sel(\'A/DHZ/*/*\')"',
      '  C_SEQ:',
      '    true:',
      '      - select: "sel(\'A/*/[102,104,129,132]/*\')"',
      '  C_HOT:',
      '    true:',
      '      - select: "sel(\'A/*/[102,132]/*\')"',
      '  C_HB:',
      '    ACCEPTOR:',
      '      - select: "sel(\'A/DHZ/*/O4\')"',
      '  C_IDX:',
      '    false:',
      '      - select: "sel(\'A/*/102/*\')"',
      '        group: motif_0',
      '      - select: "sel(\'A/*/104/*\')"',
      '        group: motif_1',
      '      - select: "sel(\'A/*/129/*\')"',
      '        group: motif_2',
      '      - select: "sel(\'A/*/132/*\')"',
      '        group: motif_3',
    ].join('\n');

    // Without remapping, author residue numbers against 1..N-renumbered split-CIF crash with `matched no atoms`
    expect(() =>
      rfd4Mod.buildBatch('contig: "16-16,A102,A104,A129,A132"', {
        inputAtomArray: atomArray,
        seed: 7,
      }),
    ).toThrow(/Preserved segment 'A102-102' matched no atoms/);

    const parsedYaml = rfd4Mod.parseConditionSpecYaml(authorYaml);
    const remapped = remapAndValidateRfd4CustomSpec(parsedYaml, templateContext, {
      isFromPdb: true,
      templateLabel: '1CTT Cropped PDB',
    });

    expect(remapped.contig).toBe('16-16,A1,A2,A3,A4,/0,B296,/0,C295');
    expect(() =>
      assertRfd4ConditionSelectorsMatch(
        atomArray,
        remapped.contig as string,
        remapped.conditions as Record<string, unknown>,
        rfd4SpecMod,
      ),
    ).not.toThrow();

    const built = rfd4Mod.buildBatch(remapped, { inputAtomArray: atomArray, seed: 7 });
    // 16 scaffold + 4 unindexed catalytic residues + 1 ZN atom + 2 DHZ atoms = 23 tokens
    expect(built.n).toBe(23);
    const inCtxCount = Array.from(built.batch.is_in_context_conditioning.data).filter(
      (v) => Number(v) > 0,
    ).length;
    expect(inCtxCount).toBe(4);

    // Invalid residue A128 inside customSpecYaml fails fast with available Chain A residues
    const badYaml = 'contig: "16-16,A102,A104,A128,A132"';
    const badParsed = rfd4Mod.parseConditionSpecYaml(badYaml);
    expect(() =>
      remapAndValidateRfd4CustomSpec(badParsed, templateContext, {
        isFromPdb: true,
        templateLabel: '1CTT Cropped PDB',
      }),
    ).toThrow(
      'Residue A128 not found in template "1CTT Cropped PDB" — available Chain A residues: A102, A104, A129, A132',
    );
    expect(() =>
      validateRfd4InputsAgainstTemplate(
        { mode: 'motif', customSpecYaml: badYaml },
        makeStructureFromText(croppedPdb, '1CTT Cropped PDB'),
      ),
    ).toThrow(/Residue A128 not found in template "1CTT Cropped PDB"/);

    // Native mmCIF label_seq_id YAML (A98,A100,A125,A128 on -4-shifted CIF) stays untouched & builds cleanly
    const shiftedCif = buildShifted1CttCif();
    const shiftedCtx = extractRfd4TemplateContext(shiftedCif);
    const shiftedAtoms = rfd4CifMod.parseCifToAtomArray(shiftedCif, { fixArginines: true });
    rfd4PrepareMod.assignIonCharges(shiftedAtoms);
    const labelSpaceYaml = rfd4Mod.parseConditionSpecYaml(
      'contig: "12-12,A98,A100,A125,A128,/0,B296,/0,C295"',
    );
    const labelKept = remapAndValidateRfd4CustomSpec(labelSpaceYaml, shiftedCtx, {
      isFromPdb: false,
      templateLabel: '1CTT Shifted CIF',
    });
    expect(labelKept.contig).toBe('12-12,A98,A100,A125,A128,/0,B296,/0,C295');
    expect(rfd4Mod.buildBatch(labelKept, { inputAtomArray: shiftedAtoms, seed: 3 }).n).toBe(18);
  });

  it('4. R4: inspectBuildFreshness & /api/health detect stale web/build against git HEAD, log startup warning, and handle non-git dirs gracefully', async () => {
    const { createSelfhostServer, inspectBuildFreshness } = loadSelfhostModule();
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'denovo-r4-test-'));
    const buildDir = path.join(tmpRoot, 'web', 'build');
    fs.mkdirSync(buildDir, { recursive: true });
    fs.writeFileSync(path.join(buildDir, 'index.html'), '<html>ok</html>');

    const shaOld = '6544101aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const shaHead = '361ad44bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

    try {
      // 1. Non-git environment (tmpRoot has no .git): never crashes, headSha=null, staleBuild=false
      fs.writeFileSync(
        path.join(buildDir, '.build-meta.json'),
        JSON.stringify({ gitSha: shaOld }),
      );
      const nonGit = inspectBuildFreshness({ repoRoot: tmpRoot, buildDir });
      expect(nonGit).toEqual({
        buildSha: shaOld,
        headSha: null,
        staleBuild: false,
      });

      // 2. Fresh build matching HEAD -> staleBuild: false
      const fresh = inspectBuildFreshness({
        repoRoot: tmpRoot,
        buildDir,
        headSha: shaHead,
        buildSha: shaHead,
      });
      expect(fresh).toEqual({
        buildSha: shaHead,
        headSha: shaHead,
        staleBuild: false,
      });

      // 3. Stale build lagging HEAD -> staleBuild: true + prominent startup warning + /api/health exposure
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      let inst: ReturnType<typeof createSelfhostServer> | null = null;
      try {
        inst = createSelfhostServer({
          host: '127.0.0.1',
          port: 0,
          repoRoot: tmpRoot,
          buildDir,
          headSha: shaHead,
          hostGpuHw: { name: 'NVIDIA GB10', shortName: 'GB10', vramMB: 128000 },
        });
        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringMatching(/Stale web\/build detected.*6544101.*361ad44/i),
        );

        inst.setRunnerBridge(
          {
            getHealth: async () => ({
              ready: true,
              gpu: { vendor: 'nvidia', architecture: 'blackwell', description: 'GB10', fallback: false },
              activeAdapter: null,
              error: null,
            }),
            runNode: async () => ({ outputs: {}, elapsedMs: 1 }),
            cancel: async () => {},
            releaseGpu: async () => {},
          },
          {
            ready: true,
            gpu: { vendor: 'nvidia', architecture: 'blackwell', description: 'GB10', fallback: false },
            activeAdapter: null,
            error: null,
          },
        );

        await new Promise<void>((r) => inst!.server.listen(0, '127.0.0.1', r));
        const addr = inst.server.address();
        const port = typeof addr === 'object' && addr ? addr.port : 0;

        const healthResp = await rawHttpGetJson(port, '/api/health');
        expect(healthResp.status).toBe(200);
        const health = JSON.parse(healthResp.body);
        expect(health.ok).toBe(true);
        expect(health.buildSha).toBe(shaOld);
        expect(health.headSha).toBe(shaHead);
        expect(health.staleBuild).toBe(true);

        // Dotfile .build-meta.json is not publicly served by static file handler
        const metaStatic = await rawHttpGetJson(port, '/.build-meta.json');
        expect(metaStatic.status).toBe(404);

        // Updating .build-meta.json on disk (simulating `npm --prefix web run build`) clears staleBuild live!
        fs.writeFileSync(
          path.join(buildDir, '.build-meta.json'),
          JSON.stringify({ gitSha: shaHead }),
        );
        const refreshedHealth = JSON.parse((await rawHttpGetJson(port, '/api/health')).body);
        expect(refreshedHealth.buildSha).toBe(shaHead);
        expect(refreshedHealth.headSha).toBe(shaHead);
        expect(refreshedHealth.staleBuild).toBe(false);
      } finally {
        warnSpy.mockRestore();
        if (inst) {
          await new Promise<void>((r) => inst.server.close(() => r()));
        }
      }
    } finally {
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });
});
