import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Color } from 'molstar/lib/mol-util/color/color.js';
import { ColorScale } from 'molstar/lib/mol-util/color/scale.js';
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition.js';
import { UncertaintyColorThemeProvider } from 'molstar/lib/mol-theme/color/uncertainty.js';
import { resolveUncertaintyColorArgFromValues } from '$lib/viewer/bfactor-domain';
import { validateBioCapabilities } from '$lib/engine/bio-capabilities';
import { reconcileBenchmarkSeeding } from '$lib/engine/benchmark-sync';
import { decodeHashToGraph, encodeGraphToHash } from '$lib/engine/share';
import type { WorkflowGraph } from '$lib/engine/types';

function readWebFile(relPath: string): string {
  return readFileSync(resolve(process.cwd(), relPath), 'utf8');
}

describe('Overnight Pass C — #60: ESM-C 3D Mol* llr-fitness domain & heatmap polarity parity', () => {
  it('computes adaptive non-clamped domains for predominantly negative natural protein LLR values and filters crystal outliers', () => {
    // Realistic 76-aa natural protein mean LLR (~ -14 to -1) + 2 crystal B-factor outliers
    const naturalLlr: number[] = [];
    for (let i = 0; i < 76; i++) {
      naturalLlr.push(-14.0 + (i / 75) * 13.0);
    }
    naturalLlr.push(58.0, 64.0);

    const arg = resolveUncertaintyColorArgFromValues(naturalLlr, 'llr-fitness');
    expect(arg.color).toBe('uncertainty');
    expect(arg.colorParams).toBeDefined();
    const [hi, lo] = arg.colorParams!.domain;
    // Reversed polarity: hi > lo (high/beneficial -> blue, low/deleterious -> red)
    expect(hi).toBeGreaterThan(lo);
    expect(hi).toBe(0);
    // Preserves deep negative conservation range instead of clamping to -6
    expect(lo).toBeLessThan(-12);
    expect(lo).toBeGreaterThanOrEqual(-14);
    expect(hi - lo).toBeGreaterThanOrEqual(2.0);
  });

  it('enforces a minimum span of 2.0 for narrow or constant LLR distributions', () => {
    const narrowNeg = resolveUncertaintyColorArgFromValues([-0.8, -0.5, -0.2], 'llr-fitness');
    expect(narrowNeg.colorParams!.domain).toEqual([0, -2]);

    const narrowMixed = resolveUncertaintyColorArgFromValues([-0.4, 0.0, 0.2], 'llr-fitness');
    const [hiMixed, loMixed] = narrowMixed.colorParams!.domain;
    expect(hiMixed).toBeGreaterThan(loMixed);
    expect(hiMixed - loMixed).toBeCloseTo(2.0, 6);

    const allZero = resolveUncertaintyColorArgFromValues([0, 0, 0], 'llr-fitness');
    expect(allZero.colorParams!.domain).toEqual([2, 0]);
  });

  it('maps deleterious LLR -> red and beneficial LLR -> blue in real Mol* UncertaintyColorTheme (matching MutationHeatmap)', () => {
    const arg = resolveUncertaintyColorArgFromValues([-14.0, -7.0, -2.0, 0.0], 'llr-fitness');
    const domain = arg.colorParams!.domain;
    expect(domain).toEqual([0, -14]);

    const params = UncertaintyColorThemeProvider.getParams({} as never);
    const merged = PD.merge(params, UncertaintyColorThemeProvider.defaultValues, { domain });
    const normalized = PD.normalizeParams(params, merged, 'all');
    expect(normalized.domain).toEqual([0, -14]);

    const scale = ColorScale.create({
      reverse: true,
      domain: normalized.domain,
      listOrName: normalized.list.colors,
    });

    // Low / deleterious LLR (-14, -20) => red (0xbf2222)
    expect(Color.toHexString(scale.color(-14))).toBe('0xbf2222');
    expect(Color.toHexString(scale.color(-20))).toBe('0xbf2222');
    // Midpoint (-7) => white (0xffffff)
    expect(Color.toHexString(scale.color(-7))).toBe('0xffffff');
    // High / neutral-beneficial LLR (0, +2) => blue (0x3361e1)
    expect(Color.toHexString(scale.color(0))).toBe('0x3361e1');
    expect(Color.toHexString(scale.color(2))).toBe('0x3361e1');
  });

  it('uses reversed red->white->blue gradient on both llr-fitness legend pills in MolSuperposeViewer', () => {
    const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
    const revMatches = viewerSrc.match(
      /\{:else if \(bfactorSemantics as string\) === 'llr-fitness'\}\s*<span>Deleterious<\/span>\s*<span class="bfactor-grad is-reversed" aria-hidden="true"><\/span>\s*<span>Beneficial \(LLR\)<\/span>/g,
    );
    expect(revMatches).toHaveLength(2);
    expect(viewerSrc).toMatch(
      /\.bfactor-grad\.is-reversed\s*\{[\s\S]*?linear-gradient\(90deg,\s*#dc2626\s+0%,\s*#f8fafc\s+50%,\s*#2563eb\s+100%\)/,
    );
  });
});

describe('Overnight Pass C — #66: Mutation Explorer row click pins heatmap column', () => {
  it('maintains userPinnedPosIdx state, resets on matrix.id change, merges pins without duplicates, and scrolls into view', () => {
    const heatmapSrc = readWebFile('src/lib/viewer/MutationHeatmap.svelte');
    expect(heatmapSrc).toContain('let userPinnedPosIdx = $state<number | null>(null);');
    expect(heatmapSrc).toMatch(
      /if \(matrix\.id !== lastMatrixId\)\s*\{\s*lastMatrixId = matrix\.id;\s*userPinnedPosIdx = null;/,
    );
    // Merges highlightSpec pins + userPinnedPosIdx cleanly without duplicates
    expect(heatmapSrc).toContain('!merged.some((h) => h.col === userPinnedPosIdx)');
    // Both heatmap cell click and table row click route through pinAndSelectResidue
    expect(heatmapSrc).toContain('function pinAndSelectResidue(');
    expect(heatmapSrc).toContain('pinAndSelectResidue(hoverCell.col, hoverCell.residueId, false);');
    expect(heatmapSrc).toContain('pinAndSelectResidue(m.positionIndex, m.residueId, true);');
    // Auto-scrolls horizontally when pinned column is out of view
    expect(heatmapSrc).toContain('function scrollColumnIntoView(');
    expect(heatmapSrc).toContain("sc.scrollTo({ left: targetScroll, behavior: 'smooth' })");
    // Renders .mh-col-pin for merged highlightedCols
    expect(heatmapSrc).toContain('{#each highlightedCols as pin (pin.col)}');
    expect(heatmapSrc).toContain('class="mh-col-pin"');
  });
});

describe('Overnight Pass C — #86: Node Studio virtualizer scroll collapse & default-closed Representation menu', () => {
  it('prevents flex-column collapse on virtualizer spacers, chain headers, and sequence rows in AnnotatedFastaView', () => {
    const fastaSrc = readWebFile('src/lib/ui/AnnotatedFastaView.svelte');
    expect(fastaSrc).toContain(
      '<div class="seq-spacer" style="height: {visibleWindow.topPad}px; flex-shrink: 0;" aria-hidden="true"></div>',
    );
    expect(fastaSrc).toContain(
      '<div class="seq-spacer" style="height: {visibleWindow.bottomPad}px; flex-shrink: 0;" aria-hidden="true"></div>',
    );
    expect(fastaSrc).toMatch(/\.seq-spacer\s*\{[\s\S]*?flex-shrink:\s*0;/);
    expect(fastaSrc).toMatch(/\.chain-head\s*\{[\s\S]*?flex-shrink:\s*0;/);
    expect(fastaSrc).toMatch(/\.seq-row\s*\{[\s\S]*?flex-shrink:\s*0;/);
  });

  it('defaults Representation menu closed in Node Studio while keeping standalone expanded viewer behavior', () => {
    const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
    const pageSrc = readWebFile('src/routes/+page.svelte');

    expect(viewerSrc).toContain('defaultStyleMenuOpen?: boolean;');
    expect(viewerSrc).toContain('defaultStyleMenuOpen = undefined');
    expect(viewerSrc).toContain('const shouldAutoOpen = exp && (defOpen ?? true);');
    expect(pageSrc).toContain(
      "defaultStyleMenuOpen={expandedPane === 'viewer' && !studioOpen}",
    );
  });
});

describe('Overnight Pass C — #88: Workbench shell & guard polish', () => {
  it('closes MolSuperposeViewer styleMenuOpen on Escape in both split and expanded views', () => {
    const viewerSrc = readWebFile('src/lib/viewer/MolSuperposeViewer.svelte');
    expect(viewerSrc).toMatch(
      /if \(!styleMenuOpen \|\| typeof window === 'undefined'\) return;[\s\S]*?function handleStyleMenuEscape\(ev: KeyboardEvent\)[\s\S]*?ev\.preventDefault\(\);[\s\S]*?ev\.stopPropagation\(\);[\s\S]*?styleMenuOpen = false;/,
    );
    expect(viewerSrc).toContain("window.addEventListener('keydown', handleStyleMenuEscape, true);");
  });

  it('gates Run CTA, shortcuts, and runWorkflow against severity===error bio compatibility issues (e.g. SiD-Protein -> SolubleMPNN)', () => {
    const incompatibleGraph: WorkflowGraph = {
      id: 'sid-solmpnn-incompatible',
      name: 'SiD-Protein to SolubleMPNN',
      description: '',
      nodes: [
        {
          id: 'sid_1',
          adapterId: 'sid-protein',
          label: 'SiD-Protein',
          params: { length: 80, numSteps: 25 },
        },
        {
          id: 'mpnn_1',
          adapterId: 'web-protein-mpnn',
          label: 'SolubleMPNN',
          params: { family: 'solublempnn', numSequences: 2 },
        },
      ],
      edges: [
        {
          fromNode: 'sid_1',
          fromPort: 'structure',
          toNode: 'mpnn_1',
          toPort: 'structure',
        },
      ],
    };

    const issues = validateBioCapabilities(incompatibleGraph, {});
    const errors = issues.filter((i) => i.severity === 'error');
    expect(errors.length).toBeGreaterThan(0);

    const pageSrc = readWebFile('src/routes/+page.svelte');
    expect(pageSrc).toContain("let bioErrorIssues = $derived(bioIssues.filter((i) => i.severity === 'error'));");
    expect(pageSrc).toContain('let hasBioErrors = $derived(bioErrorIssues.length > 0);');
    // Run CTA disabled on hasBioErrors with informative tooltip
    expect(pageSrc).toMatch(
      /data-tour="run-btn"[\s\S]*?disabled=\{isRunning \|\| validationErrors\.length > 0 \|\| hasInvalidNumeric \|\| hasBioErrors\}[\s\S]*?Biological compatibility error:/,
    );
    // runWorkflow itself also guards against severity === 'error' bio issues on targetGraph
    expect(pageSrc).toMatch(
      /const targetBioErrors = validateBioCapabilities\(targetGraph, outputsByNode\)\.filter\([\s\S]*?severity === 'error'[\s\S]*?\);[\s\S]*?if \(targetBioErrors\.length > 0\)/,
    );
  });

  it('reconciles benchmark seeding without teardown on #g= decode/loadFullWorkflowGraph and closes Node Studio', async () => {
    const unseededBenchmarkGraph: WorkflowGraph = {
      id: 'shared-bench-5o45',
      name: 'Shared PD-L1 Binder',
      description: '',
      nodes: [
        {
          id: 'pdb_1',
          adapterId: 'pdb-input',
          label: 'PD-L1 Target',
          params: {
            mode: 'demo-bundle',
            benchmarkId: 'pdl1_5o45',
            pdbId: '5O45',
            cropSegments: 'A18-132',
          },
        },
        {
          id: 'rfd4_1',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: {
            mode: 'unconditional',
            contig: '',
            hotspots: '',
          },
        },
      ],
      edges: [
        {
          fromNode: 'pdb_1',
          fromPort: 'structure',
          toNode: 'rfd4_1',
          toPort: 'structure',
        },
      ],
    };

    const { hash } = await encodeGraphToHash(unseededBenchmarkGraph);
    const decoded = await decodeHashToGraph(hash);
    expect(decoded).not.toBeNull();

    const { graph: reconciled, changedNodeIds } = reconcileBenchmarkSeeding(decoded!, {
      teardown: false,
    });
    expect(changedNodeIds).toContain('rfd4_1');
    const rfd4Node = reconciled.nodes.find((n) => n.id === 'rfd4_1');
    expect(rfd4Node?.params.mode).toBe('binder');
    expect(String(rfd4Node?.params.hotspots ?? '').length).toBeGreaterThan(0);

    const pageSrc = readWebFile('src/routes/+page.svelte');
    expect(pageSrc).toMatch(
      /function loadFullWorkflowGraph[\s\S]*?if \(expandedPane === 'studio'\)\s*\{\s*expandedPane = 'none';[\s\S]*?reconcileBenchmarkSeeding\(withDefaults, \{ teardown: false \}\)/,
    );
  });

  it('offsets .canvas-top-spine below the top-left Quick Tour hint pill so toasts/banners never overlap at 1440x920', () => {
    const pageSrc = readWebFile('src/routes/+page.svelte');
    const canvasSrc = readWebFile('src/lib/ui/WorkflowCanvas.svelte');

    const tipsMatch = canvasSrc.match(/\.canvas-tips\s*\{[\s\S]*?top:\s*(\d+)px;/);
    const spineMatch = pageSrc.match(/\.canvas-top-spine\s*\{[\s\S]*?top:\s*(\d+)px;/);
    expect(tipsMatch?.[1]).toBeTruthy();
    expect(spineMatch?.[1]).toBeTruthy();

    const tipsTop = Number(tipsMatch![1]);
    const spineTop = Number(spineMatch![1]);
    // .canvas-tips starts at ~10px and is ~28-30px tall; spine must sit clearly below (~>=44px)
    expect(spineTop - tipsTop).toBeGreaterThanOrEqual(34);
  });
});
