import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ADAPTER_REGISTRY, LOCALFOLD_MODEL_OPTIONS, MPNN_VARIANT_OPTIONS } from '$lib/engine/registry';
import {
  HOSTED_PORT_CARD_IDS,
  MODEL_CARDS,
  formatModelCardPillLabel,
  getActivePrecision,
  getModelCardForAdapter,
} from '$lib/engine/model-cards';

const EXPECTED_HOSTED_REVISIONS: Record<(typeof HOSTED_PORT_CARD_IDS)[number], { repo: string; rev: string }> = {
  'thermompnn-webgpu': {
    repo: 'https://huggingface.co/ramithuh/thermompnn-webgpu',
    rev: '9adab6c70209',
  },
  'ligandmpnn-webgpu': {
    repo: 'https://huggingface.co/ramithuh/ligandmpnn-webgpu',
    rev: 'cecd9ee340dc',
  },
  'lasermpnn-webgpu': {
    repo: 'https://huggingface.co/ramithuh/lasermpnn-webgpu',
    rev: 'a9ae906fc718',
  },
  'antifold-esmif1-webgpu': {
    repo: 'https://huggingface.co/ramithuh/antifold-esmif1-webgpu',
    rev: 'a85f8071d594',
  },
  'esmc-webgpu': {
    repo: 'https://huggingface.co/ramithuh/esmc-webgpu',
    rev: '773a9473acb5',
  },
  'protpardelle-webgpu': {
    repo: 'https://huggingface.co/ramithuh/protpardelle-webgpu',
    rev: 'dc2e820def02',
  },
  'dyna1-webgpu': {
    repo: 'https://huggingface.co/ramithuh/dyna1-webgpu',
    rev: 'eec62e7debef',
  },
  'rfdiffusion-webgpu': {
    repo: 'https://huggingface.co/ramithuh/rfdiffusion-webgpu',
    rev: 'd98f06bf8256',
  },
  'boltzgen-webgpu': {
    repo: 'https://huggingface.co/ramithuh/boltzgen-webgpu',
    rev: '6193f31710d0',
  },
  'evodiff-webgpu': {
    repo: 'https://huggingface.co/ramithuh/evodiff-webgpu',
    rev: '682460d4262d',
  },
};

describe('Scientific Model Cards & Precision/Parity Disclosure', () => {
  it('resolves a non-null ModelCardSpec for every GPU/WebGPU/ONNX adapter in ADAPTER_REGISTRY', () => {
    const mlAdapters = Object.values(ADAPTER_REGISTRY).filter((a) => a.runtime !== 'cpu-ts');
    expect(mlAdapters.length).toBeGreaterThanOrEqual(8);

    for (const adapter of mlAdapters) {
      const card = getModelCardForAdapter(adapter.id);
      expect(card, `Missing ModelCardSpec for adapter "${adapter.id}"`).not.toBeNull();
      expect(card!.status).toBe('live');
      expect(card!.adapterId).toBe(adapter.id);
      expect(card!.precisions.length).toBeGreaterThan(0);
      expect(card!.precisions.some((p) => p.id === card!.activePrecisionId)).toBe(true);
      expect(card!.limitations.length).toBeGreaterThan(0);
      expect(card!.upstreamRepo).toMatch(/^https:\/\//);
      expect(card!.license.trim().length).toBeGreaterThan(0);
      expect(card!.citation.trim().length).toBeGreaterThan(0);
    }

    // Built-in pure CPU utilities have no ML model card
    for (const adapter of Object.values(ADAPTER_REGISTRY).filter((a) => a.runtime === 'cpu-ts')) {
      expect(getModelCardForAdapter(adapter.id)).toBeNull();
    }
  });

  it('ensures every card in MODEL_CARDS has a valid activePrecisionId and complete provenance', () => {
    const cards = Object.values(MODEL_CARDS);
    expect(cards.length).toBe(14); // All 14 GPU/ONNX adapters are live!

    for (const card of cards) {
      expect(card.precisions.length).toBeGreaterThan(0);
      const activeIds = card.precisions.map((p) => p.id);
      expect(activeIds, `activePrecisionId "${card.activePrecisionId}" missing on "${card.id}"`).toContain(
        card.activePrecisionId,
      );
      const active = getActivePrecision(card);
      expect(active.id).toBe(card.activePrecisionId);
      expect(active.approxSizeMB).toBeGreaterThan(0);
      expect(active.quantizationDetail.trim().length).toBeGreaterThan(10);
      expect(card.limitations.length).toBeGreaterThan(0);
      expect(formatModelCardPillLabel(card, false).length).toBeGreaterThan(0);
      expect(formatModelCardPillLabel(card, true)).toBe(active.label);
    }
  });

  it('includes all 10 HF-hosted WebGPU ports with exact measured parity metrics, pinned HF URLs, live promotion, and honest limitations', () => {
    expect(HOSTED_PORT_CARD_IDS).toHaveLength(10);
    const liveWave7Cards: Record<string, { adapterId: string; defaultPrecision: string }> = {
      'thermompnn-webgpu': { adapterId: 'thermompnn', defaultPrecision: 'fp16' },
      'ligandmpnn-webgpu': { adapterId: 'ligandmpnn', defaultPrecision: 'fp16' },
      'lasermpnn-webgpu': { adapterId: 'lasermpnn', defaultPrecision: 'fp16' },
      'antifold-esmif1-webgpu': { adapterId: 'antifold-esmif1', defaultPrecision: 'fp16' },
      'esmc-webgpu': { adapterId: 'esmc', defaultPrecision: 'int8' },
      'protpardelle-webgpu': { adapterId: 'protpardelle', defaultPrecision: 'int8' },
      'rfdiffusion-webgpu': { adapterId: 'rfdiffusion', defaultPrecision: 'int8' },
      'boltzgen-webgpu': { adapterId: 'boltzgen', defaultPrecision: 'int8' },
      'evodiff-webgpu': { adapterId: 'evodiff', defaultPrecision: 'int8' },
      'dyna1-webgpu': { adapterId: 'dyna1', defaultPrecision: 'int8' },
    };

    for (const id of HOSTED_PORT_CARD_IDS) {
      const card = MODEL_CARDS[id];
      expect(card, `Missing hosted port card "${id}"`).toBeDefined();
      if (id in liveWave7Cards) {
        expect(card.status).toBe('live');
        expect(card.adapterId).toBe(liveWave7Cards[id].adapterId);
        expect(card.activePrecisionId).toBe(liveWave7Cards[id].defaultPrecision);
      } else {
        expect(card.status).toBe('weights-hosted');
      }
      expect(card.weightsRepo).toBe(EXPECTED_HOSTED_REVISIONS[id].repo);
      expect(card.weightsRevision).toBe(EXPECTED_HOSTED_REVISIONS[id].rev);
      expect(card.upstreamCommit).toMatch(/^[0-9a-f]{40}$/);
      expect(card.license.trim().length).toBeGreaterThan(0);
      expect(card.limitations.length).toBeGreaterThanOrEqual(1);

      // All 6 hosted ports ship fp32, fp16, and int8 precision tiers with measured parity
      const tierIds = card.precisions.map((p) => p.id);
      expect(tierIds).toEqual(expect.arrayContaining(['fp32', 'fp16', 'int8']));

      for (const tier of card.precisions) {
        expect(tier.parityVsPytorchFp32, `Missing parity on ${id}/${tier.id}`).not.toBeNull();
        expect(tier.parityVsPytorchFp32!.metric.trim().length).toBeGreaterThan(0);
        // Must contain real scientific-notation measured numbers
        expect(tier.parityVsPytorchFp32!.value).toMatch(/\d+\.\d+e[+-]\d+/i);
        expect(tier.parityVsPytorchFp32!.scope.trim().length).toBeGreaterThan(0);
      }

      // Precision override works cleanly without mutating MODEL_CARDS
      const origActive = card.activePrecisionId;
      const alt = origActive === 'fp32' ? 'int8' : 'fp32';
      const overridden = getModelCardForAdapter(id, { precision: alt });
      expect(overridden?.activePrecisionId).toBe(alt);
      expect(MODEL_CARDS[id].activePrecisionId).toBe(origActive);
    }

    // Dynamic variant & precision resolution for Wave-7A adapters
    const esmIf1Int8 = getModelCardForAdapter('antifold-esmif1', { variant: 'esmif1', precision: 'int8' });
    expect(esmIf1Int8?.displayName).toBe('ESM-IF1');
    expect(esmIf1Int8?.activePrecisionId).toBe('int8');
    expect(esmIf1Int8?.license).toContain('MIT');

    const antiFoldFp16 = getModelCardForAdapter('antifold-esmif1', { variant: 'antifold', precision: 'fp16' });
    expect(antiFoldFp16?.displayName).toBe('AntiFold (Antibody CDRs)');
    expect(antiFoldFp16?.activePrecisionId).toBe('fp16');
    expect(antiFoldFp16?.license).toContain('BSD-3-Clause');

    const thermoInt8 = getModelCardForAdapter('thermompnn', { precision: 'int8' });
    expect(thermoInt8?.activePrecisionId).toBe('int8');
    expect(getActivePrecision(thermoInt8!).approxSizeMB).toBe(1.9);

    const lig010 = getModelCardForAdapter('ligandmpnn', {
      checkpoint: 'ligandmpnn_v_32_010_25',
      precision: 'fp32',
    });
    expect(lig010?.displayName).toBe('LigandMPNN (v_32_010_25)');
    expect(lig010?.activePrecisionId).toBe('fp32');
  });

  it('specializes LocalFold, ProteinMPNN, and RFD4 cards from adapter params and covers all LocalFold families', () => {
    const baseLf = MODEL_CARDS['localfold-af2'];
    const joinedLimits = baseLf.limitations.join(' ');
    expect(joinedLimits).toContain('AF2 Monomer');
    expect(joinedLimits).toContain('AF2-Multimer');
    expect(joinedLimits).toContain('RoseTTAFold3');
    expect(joinedLimits).toContain('Boltz-2');
    expect(joinedLimits).toContain('OpenBind-0');

    for (const opt of LOCALFOLD_MODEL_OPTIONS) {
      const card = getModelCardForAdapter('localfold-af2', { model: opt.id });
      expect(card).not.toBeNull();
      expect(card!.displayName).toContain(opt.shortLabel);
      expect(card!.precisions[0].approxSizeMB).toBe(opt.approxMB);
      expect(card!.license).toContain(opt.license);
      expect(card!.weightsRevision).toMatch(/^[0-9a-f]{12}$/);
    }

    for (const tower of ['300', '600']) {
      const card = getModelCardForAdapter('localfold-af2', { model: `ef2-fast-${tower}m` })!;
      expect(card.weightsRevision).toBe('7f265c212f30');
      expect(card.precisions[0].label).toBe('INT5 trunk + INT3 ESM-C');
      expect(card.precisions[0].quantizationDetail).toContain(`ESM-C ${tower}M`);
      expect(card.limitations.join(' ')).toContain('Synthyra');
      expect(card.precisions[0].parityVsPytorchFp32).toBeNull();
    }

    for (const v of MPNN_VARIANT_OPTIONS) {
      const card = getModelCardForAdapter('web-protein-mpnn', { family: v.family });
      expect(card).not.toBeNull();
      expect(card!.displayName).toBe(v.label);
    }

    const rfd4Partial = getModelCardForAdapter('rfd4-proteina', { mode: 'partial' });
    expect(rfd4Partial?.precisions[0].approxSizeMB).toBe(469);
    expect(MODEL_CARDS['rfd4-proteina'].precisions[0].approxSizeMB).toBe(330);
  });

  it('gives honest upstream-port disclosures rather than fabricated relRMS for vendored models', () => {
    expect(MODEL_CARDS['rfd4-proteina'].precisions[0].parityVsPytorchFp32?.value).toContain(
      'no custom relRMS benchmark recorded in-repo',
    );
    expect(MODEL_CARDS['localfold-af2'].precisions[0].parityVsPytorchFp32?.value).toContain(
      'no custom relRMS benchmark recorded in-repo',
    );
    expect(MODEL_CARDS['web-protein-mpnn'].precisions[0].parityVsPytorchFp32?.value).toContain(
      'no hardware WebGPU relRMS recorded upstream',
    );
    // Genie 3 has real measured L=16 FP16 relRMS from genie3-webgpu/README_WEBGPU.md
    expect(MODEL_CARDS.genie3.precisions[0].parityVsPytorchFp32?.value).toContain('2.243e-4');
    expect(MODEL_CARDS.genie3.weightsRepo).toBe('https://huggingface.co/ramithuh/genie3-webgpu');
  });

  it('contains zero legacy brand references across all model card source files', () => {
    const forbidden = ['xy', 'mol'].join('');
    const files = [
      resolve(process.cwd(), 'src/lib/engine/model-cards.ts'),
      resolve(process.cwd(), 'src/lib/ui/ModelCardPopover.svelte'),
      resolve(process.cwd(), 'src/lib/__tests__/model-cards.test.ts'),
    ];
    for (const filePath of files) {
      const text = readFileSync(filePath, 'utf8').toLowerCase();
      expect(text, `Forbidden reference found in ${filePath}`).not.toContain(forbidden);
    }
  });
});
