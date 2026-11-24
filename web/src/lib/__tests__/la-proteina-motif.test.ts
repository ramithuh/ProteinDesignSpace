import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { laProteinaPlugin } from '$lib/plugins/la-proteina';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';

describe('La-Proteina conditional atomistic motif scaffolding (LD4–LD7 + AE3)', () => {
  it('exposes optional structure input and ld1/ld4/ld5/ld6/ld7 variant + contig schema', () => {
    const manifest = ADAPTER_REGISTRY['la-proteina'];
    expect(manifest.inputs.structure).toEqual({ type: 'StructureArtifact', required: false });
    const variantField = manifest.paramSchema?.find((f) => f.key === 'variant');
    expect(variantField?.defaultValue).toBe('ld1');
    const optionValues = variantField?.options?.map((o) => o.value) ?? [];
    expect(optionValues).toEqual(['ld1', 'ld4', 'ld5', 'ld6', 'ld7']);
    const contigField = manifest.paramSchema?.find((f) => f.key === 'contig');
    expect(contigField?.kind).toBe('text');
    expect(contigField?.defaultValue).toBe('');
  });

  it('computes variant-aware preflight weights and VRAM estimates', () => {
    const baseCtx = { upstreamProfiles: {}, connectedInputs: new Set<string>() };
    const ld1Int8 = laProteinaPlugin.preflight!({
      ...baseCtx,
      params: { variant: 'ld1', precision: 'int8', numDesigns: 1 },
    });
    expect(ld1Int8.estimatedDownloadMB).toBe(277);
    expect(ld1Int8.estimatedVramMB).toBe(900);

    const ld1Fp16 = laProteinaPlugin.preflight!({
      ...baseCtx,
      params: { variant: 'ld1', precision: 'fp16', numDesigns: 1 },
    });
    expect(ld1Fp16.estimatedDownloadMB).toBe(548);
    expect(ld1Fp16.estimatedVramMB).toBe(1200);

    for (const variant of ['ld4', 'ld5', 'ld6', 'ld7']) {
      const mInt8 = laProteinaPlugin.preflight!({
        ...baseCtx,
        params: { variant, precision: 'int8', numDesigns: 2 },
      });
      expect(mInt8.estimatedDownloadMB).toBe(402);
      expect(mInt8.estimatedVramMB).toBe(1100);
      expect(mInt8.workItems).toBe(2);

      const mFp16 = laProteinaPlugin.preflight!({
        ...baseCtx,
        params: { variant, precision: 'fp16', numDesigns: 1 },
      });
      expect(mFp16.estimatedDownloadMB).toBe(794);
      expect(mFp16.estimatedVramMB).toBe(1500);
    }
  });

  it('formats concise canvas summaries for both unconditional LD1 and motif LD4–LD7', () => {
    const ld1Summary = formatNodeParamSummary('la-proteina', {
      variant: 'ld1',
      length: 80,
      steps: 400,
      mode: 'sde',
    });
    expect(ld1Summary).toBe('80 aa · 400 steps · SDE');
    expect(laProteinaPlugin.display?.formatCanvasSummary?.({ variant: 'ld1', length: 80, steps: 400, mode: 'sde' })).toBe(
      ld1Summary,
    );

    const ld4Summary = formatNodeParamSummary('la-proteina', {
      variant: 'ld4',
      contig: '20/A16-35/20',
      steps: 400,
      mode: 'sde',
      numDesigns: 2,
    });
    expect(ld4Summary).toBe('LD4 motif (20/A16-35/20) · 400 steps · SDE · ×2');
    expect(
      laProteinaPlugin.display?.formatCanvasSummary?.({
        variant: 'ld4',
        contig: '20/A16-35/20',
        steps: 400,
        mode: 'sde',
        numDesigns: 2,
      }),
    ).toBe(ld4Summary);

    const ld7AutoSummary = formatNodeParamSummary('la-proteina', {
      variant: 'ld7',
      contig: '',
      length: 100,
      steps: 400,
      mode: 'ode',
    });
    expect(ld7AutoSummary).toBe('LD7 motif 100 aa · 400 steps · ODE');
  });

  it('vendored motif.js parses PDB, resolves contig, and extracts atom37 motif conditioning', async () => {
    const motifUrl = pathToFileURL(
      resolve(process.cwd(), 'static/vendor/la-proteina/shared/motif.js'),
    ).href;
    const motifMod = await import(/* @vite-ignore */ motifUrl);
    const pdb = [
      'ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 20.00           N',
      'ATOM      2  CA  ALA A  10       1.458   0.000   0.000  1.00 20.00           C',
      'ATOM      3  C   ALA A  10       2.009   1.420   0.000  1.00 20.00           C',
      'ATOM      4  O   ALA A  10       1.251   2.390   0.000  1.00 20.00           O',
      'ATOM      5  CB  ALA A  10       1.986  -0.760  -1.210  1.00 20.00           C',
      'ATOM      6  N   GLY A  11       3.320   1.510   0.000  1.00 20.00           N',
      'ATOM      7  CA  GLY A  11       4.010   2.800   0.000  1.00 20.00           C',
      'ATOM      8  C   GLY A  11       5.520   2.620   0.000  1.00 20.00           C',
      'ATOM      9  O   GLY A  11       6.100   1.540   0.000  1.00 20.00           O',
      'END',
    ].join('\n');

    const condAllAtom = motifMod.extractMotif({
      pdb,
      contig: '5/A10-11/5',
      atomSelectionMode: 'all_atom',
      seed: 42,
    });
    expect(condAllAtom.length).toBe(12);
    expect(condAllAtom.motifIndices).toEqual([5, 6]);
    expect(condAllAtom.resolvedContig).toBe('5/A6-7/5');

    const condTipAtom = motifMod.extractMotif({
      pdb,
      contig: '5/A10-11/5',
      atomSelectionMode: 'tip_atoms',
      seed: 42,
    });
    expect(condTipAtom.length).toBe(12);
    expect(condTipAtom.motifIndices).toEqual([5, 6]);
  });
});
