import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';
import { validateNodeParams, validateStructureText, validateTiedPositions } from '$lib/engine/param-validation';
import { formatCandidateSequence, resolveActiveRowChainLengths, resolveFoldIpsae } from '$lib/engine/peek';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';

function readSrc(relPath: string): string {
  return readFileSync(resolve(process.cwd(), relPath), 'utf8');
}

describe('M3 UI & QA regressions (Svelte 5 AST parse, pure validation, and source guards)', () => {
  it('all modified M3 Svelte components parse cleanly into Svelte 5 ASTs', () => {
    const components = [
      'src/lib/ui/WorkflowPlannerBar.svelte',
      'src/lib/ui/WorkflowCanvas.svelte',
      'src/lib/ui/RecipeDrawer.svelte',
      'src/lib/ui/NumericInput.svelte',
      'src/lib/ui/SchemaParamFields.svelte',
      'src/lib/ui/NodeInspector.svelte',
      'src/lib/ui/ProteinNode.svelte',
      'src/lib/ui/ArtifactPeekPanel.svelte',
      'src/lib/ui/ComparePeekPanel.svelte',
      'src/lib/viewer/MolSuperposeViewer.svelte',
      'src/routes/+page.svelte',
    ];
    for (const rel of components) {
      const src = readSrc(rel);
      const ast = parse(src, { modern: true });
      expect(ast.fragment).toBeDefined();
    }
  });

  it('QA #3: RecipeDrawer clears shareStatus via clearStatus() in handlers/errors/input and NOT in external graph $effect', () => {
    const src = readSrc('src/lib/ui/RecipeDrawer.svelte');
    expect(src).toContain('function clearStatus()');
    expect(src).toMatch(/function switchFormat[\s\S]*?clearStatus\(\);/);
    expect(src).toMatch(/function confirmDiscardAndSwitch[\s\S]*?clearStatus\(\);/);
    expect(src).toMatch(/function applyEdits[\s\S]*?clearStatus\(\);[\s\S]*?catch \(err\)[\s\S]*?clearStatus\(\);/);
    expect(src).toMatch(/async function copyShareUrl[\s\S]*?clearStatus\(\);[\s\S]*?catch \(err\)[\s\S]*?clearStatus\(\);/);
    expect(src).toMatch(/async function copyText[\s\S]*?clearStatus\(\);/);
    expect(src).toMatch(/oninput=\{[\s\S]*?clearStatus\(\);[\s\S]*?\}/);
    const effectBlock = src.match(/\$effect\(\(\) => \{\s*const gid = graph\.id;[\s\S]*?\}\);/);
    expect(effectBlock?.[0]).toBeTruthy();
    expect(effectBlock![0]).not.toContain('clearStatus');
  });

  it('QA #4: empty custom PDB text fails validation cleanly and UI components strip stale demo_3helix_53aa.pdb', () => {
    const errs = validateNodeParams({
      id: 'pdb1',
      adapterId: 'pdb-input',
      label: 'PDB Input',
      params: { mode: 'custom', pdbText: '', filename: '' },
    });
    expect(errs).toEqual(['"PDB Input": Paste a PDB / mmCIF structure or upload a file.']);
    expect(validateStructureText('').status).toBe('error');

    const inspectorSrc = readSrc('src/lib/ui/NodeInspector.svelte');
    expect(inspectorSrc).toContain("key === 'sequence' && value.length <= 8000");
    expect(inspectorSrc).toContain("key === 'pdbText' && value.length <= 8000");
    expect(inspectorSrc).toContain('buildTextParamUpdate');
    expect(inspectorSrc).toContain("patchPdbStatus(node.id, { error: null, summary: null })");
    expect(inspectorSrc).not.toContain("filename || 'custom.pdb'");

    const nodeSrc = readSrc('src/lib/ui/ProteinNode.svelte');
    expect(nodeSrc).toContain("'No PDB loaded'");
    expect(nodeSrc).toContain('demo_3helix_53aa.pdb');
    expect(nodeSrc).toContain('selectPrimaryOutputPort');

    const pageSrc = readSrc('src/routes/+page.svelte');
    expect(pageSrc).toContain('isCustomEmptyPdb');
    expect(pageSrc).toContain('keepStateIds: new Set([node.id])');
    expect(pageSrc).toContain('inputPreviewErrors');
  });

  it('QA #6: SchemaParamFields wires validateTiedPositions inline with .input-error and alert hint', () => {
    expect(validateTiedPositions('INVALID_TIED')).toBeTruthy();
    expect(validateTiedPositions('A10=B10; A11=B11')).toBeNull();
    expect(validateTiedPositions('')).toBeNull();

    const mpnnErrs = validateNodeParams({
      id: 'mpnn1',
      adapterId: 'web-protein-mpnn',
      label: 'SolubleMPNN',
      params: { ...defaultParamsForAdapter('web-protein-mpnn'), tiedPositions: 'INVALID_TIED' },
    });
    expect(mpnnErrs.some((e) => /tied/i.test(e))).toBe(true);

    const schemaSrc = readSrc('src/lib/ui/SchemaParamFields.svelte');
    expect(schemaSrc).toContain('validateTiedPositions');
    expect(schemaSrc).toContain("field.key === 'tiedPositions'");
    expect(schemaSrc).toContain('class:input-error={Boolean(fieldErr)}');
    expect(schemaSrc).toContain('role="alert"');
  });

  it('QA #7 & #8: ArtifactPeekPanel uses resolveFoldIpsae, resolveActiveRowChainLengths, and formatCandidateSequence', () => {
    const seq40 = 'NLYIQWLKDGGPSSGRPPPSNLYIQWLKDGGPSSGRPPPS';
    const pae = new Float32Array(40 * 40).fill(17.5);
    expect(resolveFoldIpsae({ pae, chainLengths: [20, 20] })).toEqual({ ipsae: 0, ipsaeMin: 0 });
    const rowLens = resolveActiveRowChainLengths({ chainLengths: [20, 20], alignedFoldPdb: '' });
    expect(formatCandidateSequence(seq40, rowLens)).toBe(
      'NLYIQWLKDGGPSSGRPPPS:NLYIQWLKDGGPSSGRPPPS',
    );

    const peekSrc = readSrc('src/lib/ui/ArtifactPeekPanel.svelte');
    expect(peekSrc).toContain('resolveFoldIpsae');
    expect(peekSrc).toContain('resolveActiveRowChainLengths');
    expect(peekSrc).toContain('formatCandidateSequence');
    expect(peekSrc).toContain('input-preview-banner');
    expect(peekSrc).toContain('...(row.chainLengths ? { chainLengths: row.chainLengths } : {})');
  });

  it('R1, R5, QA #1-#2, QA #5, QA #11, Compare #1-#4: enforces UI integration guards', () => {
    const plannerBarSrc = readSrc('src/lib/ui/WorkflowPlannerBar.svelte');
    expect(plannerBarSrc).toContain('$state.snapshot(graph)');
    expect(plannerBarSrc).toContain('$state.snapshot(draftPreview.graph)');

    const canvasSrc = readSrc('src/lib/ui/WorkflowCanvas.svelte');
    expect(canvasSrc).toContain('onDraftPreviewChange');
    expect(canvasSrc).toContain('notifyDraftPreview');

    const numSrc = readSrc('src/lib/ui/NumericInput.svelte');
    expect(numSrc).toContain('lastCommittedBySelf');
    expect(numSrc).toContain('shouldSyncNumericDraft');

    const cmpPanelSrc = readSrc('src/lib/ui/ComparePeekPanel.svelte');
    expect(cmpPanelSrc).toContain('computePairwiseSequenceDiff');
    expect(cmpPanelSrc).toContain('compact-cols');
    expect(cmpPanelSrc).toContain('Seq score (NLL)');
    expect(cmpPanelSrc).toContain('AA comp (H/P/C)');
    expect(cmpPanelSrc).toContain('Seq identity');

    const molSrc = readSrc('src/lib/viewer/MolSuperposeViewer.svelte');
    expect(molSrc).toContain('canonicalCamera');
    expect(molSrc).toContain('getInvariantFocus');
    expect(molSrc).toContain('effectiveSpin');
    expect(molSrc).toContain('hashStructureSample');
    expect(molSrc).toContain('const step = 1;');
    expect(molSrc).not.toContain('Math.floor(n / 8192)');

    // Extract hashStructureSample and verify full-text hashing detects every single-atom shift in >65 KB PDB and >120 KB mmCIF
    const hashFnMatch = molSrc.match(/function hashStructureSample\(text: string\): string \{([\s\S]*?)\n  \}/);
    expect(hashFnMatch?.[1]).toBeTruthy();
    const hashFn = new Function('text', hashFnMatch![1]) as (text: string) => string;
    const basePdb = 'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n'.repeat(900);
    expect(basePdb.length).toBeGreaterThan(65000);
    const basePdbHash = hashFn(basePdb);
    for (let k = 0; k < 50; k++) {
      const idx = Math.floor((basePdb.length / 50) * k) + 33;
      const altDigit = basePdb[idx] === '9' ? '8' : '9';
      const mutated = basePdb.slice(0, idx) + altDigit + basePdb.slice(idx + 1);
      expect(mutated.length).toBe(basePdb.length);
      expect(hashFn(mutated)).not.toBe(basePdbHash);
    }
    const baseCif = '_atom_site.id 1 CA ALA A 1 1.000 2.000 3.000\n'.repeat(2800);
    expect(baseCif.length).toBeGreaterThan(120000);
    const baseCifHash = hashFn(baseCif);
    for (let k = 0; k < 50; k++) {
      const idx = Math.floor((baseCif.length / 50) * k) + 27;
      const altDigit = baseCif[idx] === '7' ? '6' : '7';
      const mutated = baseCif.slice(0, idx) + altDigit + baseCif.slice(idx + 1);
      expect(hashFn(mutated)).not.toBe(baseCifHash);
    }

    const inspectorSrc = readSrc('src/lib/ui/NodeInspector.svelte');
    expect(inspectorSrc).toContain('userEditedLength');

    const pageSrc = readSrc('src/routes/+page.svelte');
    expect(pageSrc).toContain('onhashchange');
    expect(pageSrc).toContain('applySharedHash');
    expect(pageSrc).toContain('hashError');
    expect(pageSrc).toContain('activeDraftPreview');
    expect(pageSrc).toContain('isCanonicalViewerLayout');
    expect(pageSrc).toContain('compareEmptyMessage');
    expect(pageSrc).toContain('r-tab-group');
    expect(pageSrc).toContain('r-tab-actions');
    expect(pageSrc).toContain('importFileInput');
    expect(pageSrc).toContain('hasNoTaskParams');
    expect(pageSrc).toMatch(/<button[^>]*role="menuitem"[^>]*>[\s\S]*?Import JSON \/ YAML/);
    expect(pageSrc).toMatch(/onChangePairingMode=\{\(mode\) => \{\s*comparePairingMode = mode;\s*branchCandidateIdx = \{\};/);
  });
});
