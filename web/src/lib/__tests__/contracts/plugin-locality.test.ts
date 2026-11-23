import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  buildNodePreviewSignature,
  formatGeomStripLabel,
  getSiblingGeneratorIds,
  syncTaskToSiblingGenerators,
} from '$lib/ui/workbench-controller';
import { pdbInputPlugin } from '$lib/plugins/pdb-input';
import { computeUploadDigest } from '$lib/engine/param-validation';
import type { WorkflowGraph } from '$lib/engine/types';

const SRC_LIB_ROOT = resolve(process.cwd(), 'src/lib');

function collectTsFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...collectTsFiles(full));
    } else if (entry.isFile() && /\.(ts|mts|js|mjs)$/.test(entry.name)) {
      out.push(full);
    }
  }
  return out;
}

describe('Phase 6: Architecture Layer Purity & Plugin Locality Certification Gates', () => {
  it('Layer 1 Core Kernel (src/lib/kernel/) has ZERO imports from domains, plugins, protein-ops, or engine', () => {
    const kernelDir = join(SRC_LIB_ROOT, 'kernel');
    const files = collectTsFiles(kernelDir);
    expect(files.length).toBeGreaterThanOrEqual(8);

    const forbiddenImportRe =
      /(?:from|import)\s+['"][^'"]*(?:\$lib\/(?:domains|plugins|protein-ops|engine)|\.\.\/(?:domains|plugins|protein-ops|engine))/;

    const violations: string[] = [];
    for (const file of files) {
      const src = readFileSync(file, 'utf8');
      const rel = relative(SRC_LIB_ROOT, file);
      if (forbiddenImportRe.test(src)) {
        violations.push(`${rel}: forbidden cross-layer import`);
      }
      if (/['"]protein-biopolymer['"]/.test(src)) {
        violations.push(`${rel}: hardcoded protein-biopolymer literal in Layer 1`);
      }
      if (/adapterId\s*===/.test(src)) {
        violations.push(`${rel}: hardcoded adapterId === check in Layer 1`);
      }
    }
    expect(violations).toEqual([]);
  });

  it('Layer 2 Domain Packs (src/lib/domains/) have ZERO imports from Layer 3 plugins', () => {
    const domainsDir = join(SRC_LIB_ROOT, 'domains');
    const files = collectTsFiles(domainsDir);
    expect(files.length).toBeGreaterThanOrEqual(1);

    const forbiddenPluginImportRe = /(?:from|import)\s+['"][^'"]*(?:\$lib\/plugins|\.\.\/plugins)/;
    const violations: string[] = [];
    for (const file of files) {
      const src = readFileSync(file, 'utf8');
      const rel = relative(SRC_LIB_ROOT, file);
      if (forbiddenPluginImportRe.test(src)) {
        violations.push(`${rel}: forbidden Layer 2 -> Layer 3 plugin import`);
      }
    }
    expect(violations).toEqual([]);
  });

  it('Legacy src/lib/workers/ directory and legacy-bridge.ts are permanently deleted', () => {
    expect(existsSync(join(SRC_LIB_ROOT, 'workers'))).toBe(false);
    expect(existsSync(join(SRC_LIB_ROOT, 'plugins/legacy-bridge.ts'))).toBe(false);
  });

  it('Extracted UI shell components contain ZERO hardcoded adapterId === checks', () => {
    const shellComponents = [
      'ui/InspectorHeader.svelte',
      'ui/DockingScoreCard.svelte',
      'ui/RecipeDrawer.svelte',
      'ui/OnboardingModal.svelte',
      'ui/peek/EmbeddingPeekView.svelte',
      'ui/peek/TrajectoryPeekView.svelte',
    ];
    const violations: string[] = [];
    for (const rel of shellComponents) {
      const full = join(SRC_LIB_ROOT, rel);
      expect(existsSync(full)).toBe(true);
      expect(statSync(full).size).toBeGreaterThan(100);
      const src = readFileSync(full, 'utf8');
      if (/adapterId\s*===\s*['"]/.test(src)) {
        violations.push(`${rel}: contains hardcoded adapterId === '<id>'`);
      }
    }
    expect(violations).toEqual([]);
  });

  it('tools/check-deploy.mjs dynamically scans vendor/*/weights instead of hardcoding model lists', () => {
    const checkDeploySrc = readFileSync(resolve(process.cwd(), '../tools/check-deploy.mjs'), 'utf8');
    expect(checkDeploySrc).toContain("resolve(vendorDir, entry.name, 'weights')");
    expect(checkDeploySrc).not.toContain("'vendor/genie3/weights'");
  });

  it('workbench-controller helpers handle viewportRole labels, preview signatures, and sibling generator sync', () => {
    // Astra P0: GNINA receptor never mislabeled as "Designed:"
    expect(formatGeomStripLabel('receptor', 'gnina', false)).toBe('Receptor:');
    expect(formatGeomStripLabel('input', 'pdb-input', false)).toBe('Input:');
    expect(formatGeomStripLabel('predicted', 'localfold-af2', false)).toBe('Predicted:');
    expect(formatGeomStripLabel(undefined, 'pdb-input', false)).toBe('Input:');
    expect(formatGeomStripLabel(undefined, 'pdb-input', true)).toBe('Designed:');
    expect(formatGeomStripLabel(undefined, 'genie3', false)).toBe('Designed:');

    // Preview signatures (full-content digest detects same-length point mutations past char 64!)
    expect(
      buildNodePreviewSignature({
        adapterId: 'pdb-input',
        params: { mode: 'pdb-id', pdbId: '1ubq' },
      }),
    ).toBe('pdb-id|1UBQ|0||');
    expect(
      buildNodePreviewSignature({
        adapterId: 'pdb-input',
        params: { mode: 'custom', pdbText: '' },
      }),
    ).toBeNull();
    const seqBase = 'A'.repeat(70) + 'M';
    const seqMut = 'A'.repeat(70) + 'K';
    const sigBase = buildNodePreviewSignature({
      adapterId: 'sequence-input',
      params: { sequence: seqBase, label: 'test' },
    });
    const sigMut = buildNodePreviewSignature({
      adapterId: 'sequence-input',
      params: { sequence: seqMut, label: 'test' },
    });
    expect(sigBase).not.toBeNull();
    expect(sigBase).not.toBe(sigMut);

    // Sibling generator sync handles empty/NaN length gracefully without propagating NaN
    const nanGraph: WorkflowGraph = {
      id: 'nan-len-test',
      name: 'NaN Guard Test',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: { mode: 'unconditional', length: '' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: { taskMode: 'unconditional', length: 80 },
        },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' },
      ],
    };
    const nanRes = syncTaskToSiblingGenerators(nanGraph, 'rfd');
    expect(nanRes).not.toBeNull();
    expect(nanRes!.syncLen).toBe(70);
    expect(nanRes!.nextGraph.nodes.find((n) => n.id === 'g3')?.params.length).toBe(70);

    // pdb-input restoreOmittedInlineAsset restores missingUpload markers even in JSON mode (rawRecipeText=undefined)
    const largePdb = 'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n'.repeat(30);
    const sha = computeUploadDigest(largePdb);
    const restored = pdbInputPlugin.share.restoreOmittedInlineAsset?.(
      { mode: 'custom', pdbText: largePdb, filename: 'design.pdb' },
      { mode: 'custom', pdbText: '', missingUpload: true, expectedPdbSha256: sha },
      undefined,
      'pdb1',
    );
    expect(restored?.pdbText).toBe(largePdb);
    expect(restored?.missingUpload).toBeUndefined();

    // Sibling generator sync across parallel branches from same pdb-input
    const graph: WorkflowGraph = {
      id: 'sib-test',
      name: 'Sibling Sync Test',
      nodes: [
        { id: 'pdb', adapterId: 'pdb-input', params: { mode: 'demo-bundle' } },
        {
          id: 'rfd',
          adapterId: 'rfd4-proteina',
          label: 'RFD4',
          params: { mode: 'binder', length: 65, hotspots: 'A10,A12', lockChain: 'A' },
        },
        {
          id: 'g3',
          adapterId: 'genie3',
          label: 'Genie 3',
          params: { taskMode: 'unconditional', length: 80 },
        },
        {
          id: 'bg',
          adapterId: 'boltzgen',
          label: 'BoltzGen-1',
          params: { taskMode: 'unconditional', length: 48 },
        },
      ],
      edges: [
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'rfd', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'g3', toPort: 'structure' },
        { fromNode: 'pdb', fromPort: 'structure', toNode: 'bg', toPort: 'structure' },
      ],
    };

    expect(getSiblingGeneratorIds(graph, 'rfd')).toEqual(['g3', 'bg']);
    const synced = syncTaskToSiblingGenerators(graph, 'rfd');
    expect(synced).not.toBeNull();
    expect(synced!.unsupportedSiblingLabels).toEqual([]);
    const g3After = synced!.nextGraph.nodes.find((n) => n.id === 'g3')!;
    const bgAfter = synced!.nextGraph.nodes.find((n) => n.id === 'bg')!;
    expect(g3After.params.taskMode).toBe('binder');
    expect(g3After.params.length).toBe(65);
    expect(g3After.params.hotspots).toBe('A10,A12');
    expect(bgAfter.params.taskMode).toBe('binder');
    expect(bgAfter.params.length).toBe(65);
    expect(bgAfter.params.hotspots).toBe('A10,A12');
  });
});
