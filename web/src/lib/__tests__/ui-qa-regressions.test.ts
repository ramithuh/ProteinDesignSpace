import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';
import { resolveAutoModelLabelOnChange } from '$lib/ui/node-display-helpers';
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
      'src/lib/ui/InspectorHeader.svelte',
      'src/lib/ui/inspector/LocalFoldWidget.svelte',
      'src/lib/ui/NodeStudioPopover.svelte',
      'src/lib/ui/SequenceEditor.svelte',
      'src/lib/ui/ProteinNode.svelte',
      'src/lib/ui/ArtifactPeekPanel.svelte',
      'src/lib/ui/DockingScoreCard.svelte',
      'src/lib/ui/peek/EmbeddingPeekView.svelte',
      'src/lib/ui/peek/TrajectoryPeekView.svelte',
      'src/lib/ui/ComparePeekPanel.svelte',
      'src/lib/viewer/MolSuperposeViewer.svelte',
      'src/lib/viewer/MutationHeatmap.svelte',
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
    const nodeHelpersSrc = readSrc('src/lib/ui/node-display-helpers.ts');
    expect(nodeHelpersSrc).toContain("'No PDB loaded'");
    expect(nodeHelpersSrc).toContain('demo_3helix_53aa.pdb');
    expect(nodeSrc).toContain('formatNodeParamSummary');
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

  it('Wave 1D (WF-01/WF-03/WF-04/WF-10): unambiguous 1-to-1 auto-commit, synchronous benchmark seeding, preset defaults & inspector chips', () => {
    const canvasSrc = readSrc('src/lib/ui/WorkflowCanvas.svelte');
    expect(canvasSrc).toContain('!autoCommitProposed && nextProposals.length === 1');
    expect(canvasSrc).toContain('committedGraph = { ...nextGraph, edges: [...filteredEdges, singleEdge] }');

    const pageSrc = readSrc('src/routes/+page.svelte');
    expect(pageSrc).toContain('reconcileBenchmarkSeeding');
    expect(pageSrc).toContain('function mergeAdapterDefaults');
    expect(pageSrc).toMatch(/function handleGraphChange[\s\S]*?reconcileBenchmarkSeeding\(next\)/);
    expect(pageSrc).toMatch(/function updateNodeParams[\s\S]*?reconcileBenchmarkSeeding\(rawNextGraph\)/);
    expect(pageSrc).toMatch(/function selectPreset[\s\S]*?mergeAdapterDefaults/);
    expect(pageSrc).toContain('upstreamBenchmarkId={selectedUpstreamBenchmarkId}');

    const inspectorSrc = readSrc('src/lib/ui/NodeInspector.svelte');
    expect(inspectorSrc).toContain('detectQuickPdbBenchmarkMatch');
    expect(inspectorSrc).toContain('buildBenchmarkPdbParams');
    expect(inspectorSrc).toContain('Apply {quickBenchMatch.shortLabel} benchmark');
    expect(inspectorSrc).toContain('Synced from ${benchmarkSyncInfo.benchLabel}');
    expect(inspectorSrc).toContain('↺ Reset to benchmark');
    expect(inspectorSrc).toContain('handleResetToBenchmark');
  });

  it('Wave 1D (UX-01/UX-02/UX-03/UX-08): top-most Escape router, Space interactive guard, stale URL hash cleanup & 2-click clear confirms', () => {
    const pageSrc = readSrc('src/routes/+page.svelte');
    expect(pageSrc).toContain('function isInteractiveOrEditableElement');
    expect(pageSrc).toContain('!isInteractiveOrEditableElement(ev.target)');
    expect(pageSrc).toContain("document.querySelector('.mc-backdrop')");
    expect(pageSrc).toMatch(/if \(pendingRun\) \{\s*ev\.preventDefault\(\);\s*cancelPendingRun\(\);\s*return;\s*\}/);
    expect(pageSrc).toMatch(/if \(showExamplesGallery\) \{\s*ev\.preventDefault\(\);\s*showExamplesGallery = false;\s*return;\s*\}/);
    expect(pageSrc).toContain('function clearStaleUrlHash()');
    expect(pageSrc).toMatch(/function selectPreset[\s\S]*?clearStaleUrlHash\(\)/);
    expect(pageSrc).toMatch(/async function selectPaperInsight[\s\S]*?clearStaleUrlHash\(\)/);
    expect(pageSrc).toMatch(/function loadFullWorkflowGraph[\s\S]*?if \(!opts\?\.preserveHash\)/);
    expect(pageSrc).toContain('preserveHash: true');
    expect(pageSrc).toContain('brandClearConfirmPending');
    expect(pageSrc).toContain('handleBrandLogoClick');
    expect(pageSrc).toContain('cacheClearConfirmPending');
    expect(pageSrc).toContain('handleClearCacheClick');

    const canvasSrc = readSrc('src/lib/ui/WorkflowCanvas.svelte');
    expect(canvasSrc).toContain('.mc-backdrop');
    expect(canvasSrc).toContain('.gal-backdrop');
    expect(canvasSrc).toContain('.pw-coach');
    expect(canvasSrc).toMatch(/if \(catalogOpen\) \{\s*ev\.preventDefault\(\);\s*catalogOpen = false;\s*return;\s*\}/);
    expect(canvasSrc).toMatch(/if \(wireLegendOpen\) \{\s*ev\.preventDefault\(\);\s*wireLegendOpen = false;\s*return;\s*\}/);
  });

  it('Wave 1D (UX-05/UX-06/UX-07/UX-09/UX-10): unified toast/banner spine, honest action feedback, split Mol* style/PNG, Studio click isolation & 6-block compare cap', () => {
    const pageSrc = readSrc('src/routes/+page.svelte');
    expect(pageSrc).toContain('canvas-top-spine');
    expect(pageSrc).toContain('appliedDraftNotice');
    expect(pageSrc).toContain('handleStudioInputResidueClick');
    expect(pageSrc).toContain('studioInputViewerActive\n                ? handleStudioInputResidueClick\n                : handleResidueClick');
    expect(pageSrc).toContain('onActiveBrushChange={(brush) => (studioActiveBrush = brush)}');
    expect(pageSrc).toMatch(/async function freeVram[\s\S]*?catch \(err\)[\s\S]*?showToastMessage/);
    expect(pageSrc).toMatch(/async function handleClearCache[\s\S]*?catch \(err\)[\s\S]*?showToastMessage/);

    const canvasSrc = readSrc('src/lib/ui/WorkflowCanvas.svelte');
    expect(canvasSrc).not.toContain('class="toast"');
    expect(canvasSrc).not.toContain('class="ai-draft-warning');
    expect(canvasSrc).toContain('cur.length >= 6');
    expect(canvasSrc).toContain('Compare mode supports up to 6 blocks at once');

    const cmpSrc = readSrc('src/lib/ui/ComparePeekPanel.svelte');
    expect(cmpSrc).toContain('up to 6 blocks.');
    expect(cmpSrc).not.toContain('up to 4 blocks.');

    const molSrc = readSrc('src/lib/viewer/MolSuperposeViewer.svelte');
    expect(molSrc).not.toContain('{#if expanded && hasAnyStructure}');
    expect(molSrc).toContain('{#if hasAnyStructure}');
    expect(molSrc).toContain('PNG failed');
    expect(molSrc).toContain('onNotify?.(');

    const seqSrc = readSrc('src/lib/ui/SequenceEditor.svelte');
    expect(seqSrc).toContain('Copy failed');
    expect(seqSrc).toContain('onNotify?.(');

    const heatmapSrc = readSrc('src/lib/viewer/MutationHeatmap.svelte');
    expect(heatmapSrc).toContain('Copy failed');
    expect(heatmapSrc).toContain('onNotify?.(');

    const studioSrc = readSrc('src/lib/ui/NodeStudioPopover.svelte');
    expect(studioSrc).toContain('Copy failed');
    expect(studioSrc).toContain('onActiveBrushChange?.(activeBrush)');
  });

  it('auto-updates card title when switching multi-model dropdowns while preserving suffixes and custom labels', () => {
    // LocalFold aliases & exact labels
    expect(resolveAutoModelLabelOnChange('localfold-af2', 'ESMFold2 300M', 'ef2-fast-600m')).toBe(
      'EF2 Fast 600M',
    );
    expect(resolveAutoModelLabelOnChange('localfold-af2', 'EF2 Fast 300M', 'boltz2')).toBe('Boltz-2');
    expect(resolveAutoModelLabelOnChange('localfold-af2', 'LocalFold (AF2)', 'protenix2')).toBe(
      'Protenix-v2',
    );
    expect(resolveAutoModelLabelOnChange('localfold-af2', 'AF2 Model 1 (MPNN)', 'boltz2')).toBe(
      'Boltz-2 (MPNN)',
    );
    expect(resolveAutoModelLabelOnChange('localfold-af2', 'ESMFold2 300M #2', 'ef2-fast-600m')).toBe(
      'EF2 Fast 600M #2',
    );
    expect(resolveAutoModelLabelOnChange('localfold-af2', 'Catalytic Pocket Refold', 'boltz2')).toBeNull();

    // ProteinMPNN variants
    expect(resolveAutoModelLabelOnChange('web-protein-mpnn', 'SolubleMPNN', 'ca')).toBe(
      'Cα-ProteinMPNN',
    );
    expect(resolveAutoModelLabelOnChange('web-protein-mpnn', 'Cα-MPNN (Branch 1)', 'proteinmpnn')).toBe(
      'ProteinMPNN (Branch 1)',
    );
    expect(resolveAutoModelLabelOnChange('web-protein-mpnn', 'Core Packer', 'ca')).toBeNull();

    // AntiFold / ESM-IF1 variants
    expect(resolveAutoModelLabelOnChange('antifold-esmif1', 'AntiFold / ESM-IF1', 'esmif1')).toBe(
      'ESM-IF1',
    );
    expect(resolveAutoModelLabelOnChange('antifold-esmif1', 'ESM-IF1', 'antifold')).toBe('AntiFold');
    expect(resolveAutoModelLabelOnChange('antifold-esmif1', 'CDR Loop Designer', 'esmif1')).toBeNull();
  });
});
