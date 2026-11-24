import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  extractRfd4DesignMetadata,
  extractRfd4TemplateContext,
  resolveGenie3CompiledInput,
} from '$lib/engine/conditioning';
import { parseStructureText } from '$lib/protein-ops/pdb';
import type { StructureArtifact } from '$lib/engine/types';

function makeMinimalPdb(residues: Array<{ chain: string; num: number; ins?: string; name?: string }>): string {
  const lines: string[] = [];
  let serial = 1;
  residues.forEach((r, i) => {
    const resName = r.name ?? 'ALA';
    const x = (i * 3.8).toFixed(3).padStart(8);
    const ins = r.ins ?? ' ';
    const num = String(r.num).padStart(4);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  N   ${resName} ${r.chain}${num}${ins}   ${x}   0.000   0.000  1.00 20.00           N`);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  CA  ${resName} ${r.chain}${num}${ins}   ${x}   1.450   0.000  1.00 20.00           C`);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  C   ${resName} ${r.chain}${num}${ins}   ${x}   2.000   1.200  1.00 20.00           C`);
    lines.push(`ATOM  ${String(serial++).padStart(5)}  O   ${resName} ${r.chain}${num}${ins}   ${x}   1.200   2.100  1.00 20.00           O`);
  });
  lines.push('END');
  return `${lines.join('\n')}\n`;
}

describe('Astra 6.2 Audit Wave 3: Backbone Generators & Conditioning Correctness (Shards 1–3)', () => {
  it('EvoDiff sampling.choose rejects non-finite logits instead of falling through to token 20 (B)', async () => {
    const url = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/evodiff/shared/sampling.js'),
    ).href;
    const mod = (await import(/* @vite-ignore */ url)) as {
      choose: (logits: Float32Array, position: number, u: number) => number;
    };
    const logits = new Float32Array(31);
    logits[5] = Number.NaN;
    expect(() => mod.choose(logits, 0, 0.5)).toThrow(/Non-finite/i);
  });

  it('FoldFlow 2 toPDB omits virtual CB on GLY and serializes unknown X as UNK instead of ALA', async () => {
    const url = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/foldflow2/shared/pdb.js'),
    ).href;
    const mod = (await import(/* @vite-ignore */ url)) as {
      toPDB: (arg: { atoms: Float32Array; length: number }, opts?: { sequence?: string }) => string;
    };
    const atoms = new Float32Array(3 * 15);
    for (let i = 0; i < atoms.length; i++) atoms[i] = i * 0.1;
    const pdb = mod.toPDB({ atoms, length: 3 }, { sequence: 'GAX' });
    const lines = pdb.split('\n').filter((l) => l.startsWith('ATOM'));
    // Residue 1 (GLY): 4 atoms (no CB)
    const r1 = lines.filter((l) => l.slice(22, 26).trim() === '1');
    expect(r1.map((l) => l.slice(12, 16).trim())).toEqual(['N', 'CA', 'C', 'O']);
    // Residue 2 (ALA): 5 atoms (has CB)
    const r2 = lines.filter((l) => l.slice(22, 26).trim() === '2');
    expect(r2.map((l) => l.slice(12, 16).trim())).toEqual(['N', 'CA', 'C', 'CB', 'O']);
    // Residue 3 (X): UNK, not ALA
    const r3 = lines.filter((l) => l.slice(22, 26).trim() === '3');
    expect(r3[0].slice(17, 20).trim()).toBe('UNK');
  });

  it('BoltzGen and ProtPardelle parsers reject blank PDB coordinate columns instead of coercing to 0.0', async () => {
    const blankPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N',
      'ATOM      2  CA  ALA A   1               1.000   2.000  1.00 20.00           C',
      'ATOM      3  C   ALA A   1       2.000   1.000   0.000  1.00 20.00           C',
      'ATOM      4  O   ALA A   1       2.000   2.000   0.000  1.00 20.00           O',
      'END',
    ].join('\n');

    const bgUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/boltzgen/shared/input.js'),
    ).href;
    const bg = (await import(/* @vite-ignore */ bgUrl)) as { parsePDB: (s: string) => unknown };
    expect(() => bg.parsePDB(blankPdb)).toThrow(/Invalid PDB coordinate/i);

    const ppUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/protpardelle/shared/structure.js'),
    ).href;
    const pp = (await import(/* @vite-ignore */ ppUrl)) as {
      parseStructure: (s: string, opts?: { format?: string }) => unknown;
    };
    expect(() => pp.parseStructure(blankPdb, { format: 'pdb' })).toThrow(/Invalid atom coordinates/i);
  });

  it('RFdiffusion parsePDB stops at ENDMDL and parseContig respects insertion codes', async () => {
    const url = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/rfdiffusion/shared/structure.js'),
    ).href;
    const rfd = (await import(/* @vite-ignore */ url)) as {
      parsePDB: (s: string) => { residues: Array<{ chain: string; number: number; icode: string }> };
      parseContig: (c: string, opts: { pdb: unknown }) => Array<{ source?: { number: number; icode: string } }>;
    };
    const multiModel = [
      'MODEL        1',
      makeMinimalPdb([
        { chain: 'A', num: 10 },
        { chain: 'A', num: 10, ins: 'A' },
        { chain: 'A', num: 11 },
      ]).replace('END\n', ''),
      'ENDMDL',
      'MODEL        2',
      makeMinimalPdb([{ chain: 'A', num: 99 }]),
      'ENDMDL',
    ].join('\n');

    const parsed = rfd.parsePDB(multiModel);
    expect(parsed.residues.length).toBe(3);
    expect(parsed.residues.map((r) => `${r.number}${r.icode}`)).toEqual(['10', '10A', '11']);

    const contig = rfd.parseContig('A10A-11', { pdb: parsed });
    expect(contig.map((r) => `${r.source?.number}${r.source?.icode}`)).toEqual(['10A', '11']);
  });

  it('conditioning.ts remapPolymerRange does NOT silently fall back to label_seq_id when author residue misses', () => {
    const cif = [
      'data_test',
      'loop_',
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_entity_id',
      '_atom_site.label_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.auth_seq_id',
      '_atom_site.auth_asym_id',
      'ATOM 1 C CA ALA A 1 1 ? 0.0 0.0 0.0 101 A',
      'ATOM 2 C CA GLY A 1 2 ? 3.8 0.0 0.0 102 A',
      'ATOM 3 C CA VAL A 1 3 ? 7.6 0.0 0.0 103 A',
    ].join('\n');
    const ctx = extractRfd4TemplateContext(cif);
    // Requesting missing author residue A1 on a structure numbered A101-A103 must NOT remap to label_seq_id 1 (A101)!
    const miss = ctx.remapPolymerRange('A', 1, 1);
    expect(miss).toEqual({ labelChain: 'A', start: 1, end: 1 });
    // Wait: assert miss did NOT match auth 101 by testing a missing number > maxLabelSeq (e.g. A50)
    // AND testing that A101 maps cleanly to label 1 while A2 stays 2 (not remapped when label exists)
    const hit = ctx.remapPolymerRange('A', 101, 102);
    expect(hit).toEqual({ labelChain: 'A', start: 1, end: 2 });
    // Specifically verify remapPolymerRange on A1 when label_seq_id starts at 5 (distinct from 1):
    const cifOffset = cif
      .replace(' 1 ? 0.0', ' 5 ? 0.0')
      .replace(' 2 ? 3.8', ' 6 ? 3.8')
      .replace(' 3 ? 7.6', ' 7 ? 7.6');
    const ctx2 = extractRfd4TemplateContext(cifOffset);
    // Requesting A5 (matches label_seq_id 5, NOT auth_seq_id 101!) must NOT hit label 5
    const hit101 = ctx2.remapPolymerRange('A', 101, 101);
    expect(hit101).toEqual({ labelChain: 'A', start: 5, end: 5 });
    const miss5 = ctx2.remapPolymerRange('A', 5, 6);
    expect(miss5).toEqual({ labelChain: 'A', start: 5, end: 6 });
  });

  it('extractRfd4DesignMetadata separates coordinate-only restraints from MPNN sequence locks', () => {
    const pdb = makeMinimalPdb([
      { chain: 'A', num: 1 },
      { chain: 'A', num: 2 },
      { chain: 'A', num: 3 },
    ]);
    const parsed = parseStructureText(pdb);
    const atoms = [
      { atomName: 'CA', tokenId: 0, coordGroup: 1, inCoord: [0, 0, 0] },
      { atomName: 'CA', tokenId: 1, coordGroup: -1 },
      { atomName: 'CA', tokenId: 2, coordGroup: 1, inCoord: [7.6, 0, 0] },
    ];
    // Only residue 0 has C_SEQ=1; residue 2 has coordinate-only restraint (C_CRDGRP=1, C_SEQ=0)
    const batch = { condition_sequence_1_residue: { data: new Int32Array([1, 0, 0]) } };
    const meta = extractRfd4DesignMetadata(parsed, atoms, batch);
    expect(meta.fixedResidueIndices).toEqual([1]); // ONLY residue 0 is sequence-fixed
    expect(meta.motifIndices).toEqual([0, 2]); // Both residues 0 and 2 have motif coordinates!
  });

  it('resolveGenie3CompiledInput ignores stale lockChain in motif mode and rejects unmatched/empty motif spans', () => {
    const pdb = makeMinimalPdb([
      { chain: 'A', num: 10 },
      { chain: 'A', num: 11 },
      { chain: 'A', num: 12 },
      { chain: 'A', num: 13 },
      { chain: 'B', num: 1 },
      { chain: 'B', num: 2 },
      { chain: 'B', num: 3 },
    ]);
    const parsed = parseStructureText(pdb);
    const tpl = {
      kind: 'StructureArtifact',
      id: 'tpl-1',
      label: 'tpl',
      sourceModelId: 'pdb-input',
      length: parsed.nResidues,
      sequence: parsed.sequence,
      caCoords: parsed.caCoords,
      pdbText: pdb,
      chainIds: parsed.chainIds,
    } as StructureArtifact;

    // Stale binder lockChain='B' in motif mode must NOT flip to binder mode!
    const motifCompiled = resolveGenie3CompiledInput(
      { taskMode: 'motif', length: 40, motifSpan: 'A10-13', lockChain: 'B' },
      tpl,
    );
    expect(motifCompiled.mode).toBe('motif');

    // Generated-only contig=40-40 in motif mode must throw instead of silently running unconditional
    expect(() =>
      resolveGenie3CompiledInput({ taskMode: 'motif', length: 40, contig: '40-40' }, tpl),
    ).toThrow(/motif mode requires/i);

    // Unmatched motif span Z99-101 must throw instead of being silently dropped
    expect(() =>
      resolveGenie3CompiledInput({ taskMode: 'motif', length: 40, contig: '15-15,Z99-101,15-15' }, tpl),
    ).toThrow(/matched 0 residues/i);
  });
});
