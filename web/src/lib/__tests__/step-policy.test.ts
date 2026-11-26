import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
  MODEL_STEP_POLICIES,
  assertSafeStepCount,
  formatLowStepGeometryWarning,
  getModelStepPolicy,
  resolveStepRule,
  validateStepCount,
} from '$lib/engine/step-policy';
import { MODEL_CARDS } from '$lib/engine/model-cards';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { decodeFlowQuery, parseFlowDsl } from '$lib/engine/flow-dsl';

const GENERATOR_WORKER_FILES: Readonly<Record<string, string>> = {
  'la-proteina': 'src/lib/plugins/la-proteina/laproteina.worker.ts',
  protpardelle: 'src/lib/plugins/protpardelle/protpardelle.worker.ts',
  boltzgen: 'src/lib/plugins/boltzgen/boltzgen.worker.ts',
  'rfd4-proteina': 'src/lib/plugins/rfd4-proteina/rfd4.worker.ts',
  proteina: 'src/lib/plugins/proteina/proteina.worker.ts',
  genie2: 'src/lib/plugins/genie2/genie2.worker.ts',
  rfd3: 'src/lib/plugins/rfd3/rfd3.worker.ts',
  rfdiffusion: 'src/lib/plugins/rfdiffusion/rfdiffusion.worker.ts',
  frameflow: 'src/lib/plugins/frameflow/frameflow.worker.ts',
  foldflow2: 'src/lib/plugins/foldflow2/foldflow2.worker.ts',
  'sid-protein': 'src/lib/plugins/sid-protein/sidprotein.worker.ts',
};

async function loadSelfhostModule() {
  const serverPath = path.resolve(process.cwd(), '../server/selfhost-server.mjs');
  const req = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return req(serverPath) as any;
}

function rawHttpRequest(opts: {
  port: number;
  method?: string;
  path: string;
  headers?: Record<string, string>;
  body?: string;
}): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port: opts.port,
        method: opts.method ?? 'GET',
        path: opts.path,
        headers: opts.headers,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(Buffer.from(c)));
        res.on('end', () =>
          resolve({
            status: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString('utf8'),
          }),
        );
      },
    );
    req.on('error', reject);
    if (opts.body) req.write(opts.body);
    req.end();
  });
}

describe('Model-Card-backed per-model sampling step policy', () => {
  it('maintains single-source-of-truth parity across MODEL_STEP_POLICIES, MODEL_CARDS, and ADAPTER_REGISTRY', () => {
    const policyIds = Object.keys(MODEL_STEP_POLICIES).sort();
    expect(policyIds).toHaveLength(11);
    expect(policyIds).toEqual(Object.keys(GENERATOR_WORKER_FILES).sort());

    for (const adapterId of policyIds) {
      const envelope = MODEL_STEP_POLICIES[adapterId];
      expect(getModelStepPolicy(adapterId)).toBe(envelope);

      // 1. Model Card samplingEnvelope parity
      const card =
        MODEL_CARDS[adapterId] ??
        Object.values(MODEL_CARDS).find((c) => c.adapterId === adapterId);
      expect(card, `Missing MODEL_CARDS entry for ${adapterId}`).toBeDefined();
      expect(card!.samplingEnvelope, `Missing samplingEnvelope on MODEL_CARDS for ${adapterId}`).toEqual(
        envelope,
      );

      // 2. Registry paramSchema parity (paramKey, min/max, warnBelow)
      const manifest = ADAPTER_REGISTRY[adapterId];
      expect(manifest, `Missing ADAPTER_REGISTRY entry for ${adapterId}`).toBeDefined();
      const stepField = (manifest.paramSchema ?? []).find((f) => f.key === envelope.paramKey);
      expect(
        stepField,
        `ADAPTER_REGISTRY['${adapterId}'].paramSchema missing step field "${envelope.paramKey}"`,
      ).toBeDefined();

      const minAcrossRules = Math.min(...envelope.rules.map((r) => r.minSafeSteps));
      const maxAcrossRules = Math.max(...envelope.rules.map((r) => r.maxSteps));
      const defaultResolved = resolveStepRule(adapterId, {});
      expect(defaultResolved).not.toBeNull();
      const defaultRule = defaultResolved!.rule;

      expect(stepField!.min, `${adapterId}.${envelope.paramKey}.min`).toBe(minAcrossRules);
      expect(stepField!.max, `${adapterId}.${envelope.paramKey}.max`).toBe(maxAcrossRules);

      if (defaultRule.recommendedMinSteps > minAcrossRules) {
        expect(
          stepField!.warnBelow?.threshold,
          `${adapterId}.${envelope.paramKey}.warnBelow.threshold`,
        ).toBe(defaultRule.recommendedMinSteps);
      } else if (stepField!.warnBelow) {
        expect(stepField!.warnBelow.threshold).toBeLessThanOrEqual(defaultRule.recommendedMinSteps);
      }
    }
  });

  it('enforces mode-aware step envelopes across all 11 generators', () => {
    // la-proteina: SDE vs ODE
    expect(validateStepCount('la-proteina', { steps: 100 })?.ok).toBe(false);
    expect(validateStepCount('la-proteina', { mode: 'sde', steps: 199 })?.ok).toBe(false);
    expect(validateStepCount('la-proteina', { mode: 'sde', steps: 200 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('la-proteina', { mode: 'sde', steps: 349 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('la-proteina', { mode: 'sde', steps: 350 })).toMatchObject({
      ok: true,
      level: 'ok',
    });
    expect(validateStepCount('la-proteina', { mode: 'sde', steps: 400 })).toMatchObject({
      ok: true,
      level: 'ok',
    });
    expect(validateStepCount('la-proteina', { mode: 'ode', steps: 49 })?.ok).toBe(false);
    expect(validateStepCount('la-proteina', { mode: 'ode', steps: 50 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('la-proteina', { mode: 'ode', steps: 99 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('la-proteina', { mode: 'ode', steps: 100 })).toMatchObject({
      ok: true,
      level: 'ok',
    });
    expect(validateStepCount('la-proteina', { mode: 'ode', steps: 400 })).toMatchObject({
      ok: true,
      level: 'ok',
    });

    // protpardelle: unconditional / motif / binder vs partial
    for (const taskMode of ['unconditional', 'motif', 'binder']) {
      expect(validateStepCount('protpardelle', { taskMode, numSteps: 100 })?.ok).toBe(false);
      expect(validateStepCount('protpardelle', { taskMode, numSteps: 199 })?.ok).toBe(false);
      expect(validateStepCount('protpardelle', { taskMode, numSteps: 200 })).toMatchObject({
        ok: true,
        level: 'ok',
      });
    }
    expect(validateStepCount('protpardelle', { taskMode: 'partial', numSteps: 49 })?.ok).toBe(false);
    expect(validateStepCount('protpardelle', { taskMode: 'partial', numSteps: 50 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('protpardelle', { taskMode: 'partial', numSteps: 200 })).toMatchObject({
      ok: true,
      level: 'ok',
    });

    // boltzgen
    expect(validateStepCount('boltzgen', { numSteps: 3 })?.ok).toBe(false);
    expect(validateStepCount('boltzgen', { numSteps: 100 })?.ok).toBe(false);
    expect(validateStepCount('boltzgen', { numSteps: 199 })?.ok).toBe(false);
    expect(validateStepCount('boltzgen', { numSteps: 200 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('boltzgen', { numSteps: 500 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('boltzgen', { numSteps: 501 })?.ok).toBe(false);

    // genie2
    expect(validateStepCount('genie2', { steps: 10 })?.ok).toBe(false);
    expect(validateStepCount('genie2', { steps: 500 })?.ok).toBe(false);
    expect(validateStepCount('genie2', { steps: 999 })?.ok).toBe(false);
    expect(validateStepCount('genie2', { steps: 1000 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('genie2', { steps: 1001 })?.ok).toBe(false);

    // proteina: sc vs vf
    expect(validateStepCount('proteina', { mode: 'sc', steps: 49 })?.ok).toBe(false);
    expect(validateStepCount('proteina', { mode: 'sc', steps: 100 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('proteina', { mode: 'sc', steps: 350 })).toMatchObject({
      ok: true,
      level: 'ok',
    });
    expect(validateStepCount('proteina', { mode: 'vf', steps: 49 })?.ok).toBe(false);
    expect(validateStepCount('proteina', { mode: 'vf', steps: 50 })).toMatchObject({
      ok: true,
      level: 'warn',
    });
    expect(validateStepCount('proteina', { mode: 'vf', steps: 100 })).toMatchObject({
      ok: true,
      level: 'ok',
    });

    // rfd4-proteina
    expect(validateStepCount('rfd4-proteina', { nsteps: 49 })?.ok).toBe(false);
    expect(validateStepCount('rfd4-proteina', { nsteps: 50 })).toMatchObject({ ok: true, level: 'warn' });
    expect(validateStepCount('rfd4-proteina', { nsteps: 57 })).toMatchObject({ ok: true, level: 'warn' });
    expect(validateStepCount('rfd4-proteina', { nsteps: 58 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('rfd4-proteina', { nsteps: 120 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('rfd4-proteina', { nsteps: 121 })?.ok).toBe(false);

    // rfd3, rfdiffusion, frameflow, foldflow2
    const min20Models: Array<{ id: string; key: 'steps' | 'numSteps'; max: number }> = [
      { id: 'rfd3', key: 'steps', max: 200 },
      { id: 'rfdiffusion', key: 'numSteps', max: 50 },
      { id: 'frameflow', key: 'steps', max: 200 },
      { id: 'foldflow2', key: 'steps', max: 100 },
    ];
    for (const m of min20Models) {
      expect(validateStepCount(m.id, { [m.key]: 19 })?.ok, `${m.id} < 20`).toBe(false);
      expect(validateStepCount(m.id, { [m.key]: 20 }), `${m.id} == 20`).toMatchObject({
        ok: true,
        level: 'warn',
      });
      expect(validateStepCount(m.id, { [m.key]: 49 }), `${m.id} == 49`).toMatchObject({
        ok: true,
        level: 'warn',
      });
      expect(validateStepCount(m.id, { [m.key]: 50 }), `${m.id} == 50`).toMatchObject({
        ok: true,
        level: 'ok',
      });
      expect(validateStepCount(m.id, { [m.key]: m.max }), `${m.id} == max`).toMatchObject({
        ok: true,
        level: 'ok',
      });
      expect(validateStepCount(m.id, { [m.key]: m.max + 1 })?.ok, `${m.id} > max`).toBe(false);
    }

    // sid-protein
    expect(validateStepCount('sid-protein', { steps: 0 })?.ok).toBe(false);
    expect(validateStepCount('sid-protein', { steps: 1 })).toMatchObject({ ok: true, level: 'warn' });
    expect(validateStepCount('sid-protein', { steps: 3 })).toMatchObject({ ok: true, level: 'warn' });
    expect(validateStepCount('sid-protein', { steps: 4 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('sid-protein', { steps: 32 })).toMatchObject({ ok: true, level: 'ok' });
    expect(validateStepCount('sid-protein', { steps: 33 })?.ok).toBe(false);

    // assertSafeStepCount throws on invalid and resolves defaults/valid numbers
    expect(() => assertSafeStepCount('la-proteina', { mode: 'sde', steps: 100 }, 400)).toThrow(
      /below the safe minimum \(200\)/,
    );
    expect(assertSafeStepCount('la-proteina', { mode: 'ode', steps: 80 }, 400)).toBe(80);
    expect(assertSafeStepCount('rfd3', {}, 50)).toBe(50);
    expect(() => assertSafeStepCount('boltzgen', { numSteps: 3 }, 200)).toThrow(
      /below the safe minimum \(200\)/,
    );
  });

  it('formats low-step geometry warnings only when defects occur below recommended/upstream steps', () => {
    const breakWarn = formatLowStepGeometryWarning(
      'la-proteina',
      { mode: 'sde', steps: 220 },
      { caBreaks: 3, clashes: 0, minCaDist: 3.7 },
    );
    expect(breakWarn).toMatch(/La-Proteina \(sde mode\)/);
    expect(breakWarn).toMatch(/3 Cα breaks/);
    expect(breakWarn).toMatch(/steps=220/);
    expect(breakWarn).toMatch(/recommended ≥350/);

    const clashWarn = formatLowStepGeometryWarning(
      'rfd4-proteina',
      { nsteps: 52 },
      { caBreaks: 0, clashes: 4, minCaDist: 1.8 },
    );
    expect(clashWarn).toMatch(/RFD4-Proteina/);
    expect(clashWarn).toMatch(/4 steric clashes/);
    expect(clashWarn).toMatch(/nsteps=52/);

    // Clean geometry returns null even at low step counts
    expect(
      formatLowStepGeometryWarning(
        'la-proteina',
        { mode: 'sde', steps: 220 },
        { caBreaks: 0, clashes: 0, minCaDist: 3.8 },
      ),
    ).toBeNull();

    // At or above recommended & upstream default returns null even if defects occur
    expect(
      formatLowStepGeometryWarning(
        'la-proteina',
        { mode: 'sde', steps: 400 },
        { caBreaks: 2, clashes: 1, minCaDist: 1.9 },
      ),
    ).toBeNull();

    expect(
      formatLowStepGeometryWarning(
        'rfd4-proteina',
        { nsteps: 60 },
        { caBreaks: 1, clashes: 1, minCaDist: 1.9 },
      ),
    ).toBeNull();
  });

  it('audits all 11 generator workers: assertSafeStepCount runs before engine/weight loading and low-step warnings are wired', () => {
    for (const [adapterId, relPath] of Object.entries(GENERATOR_WORKER_FILES)) {
      const fullPath = path.resolve(process.cwd(), relPath);
      expect(fs.existsSync(fullPath), `Missing worker file ${relPath}`).toBe(true);
      const src = fs.readFileSync(fullPath, 'utf8');

      expect(src, `${relPath} must import from $lib/engine/step-policy`).toMatch(
        /from\s+['"]\$lib\/engine\/step-policy['"]/,
      );
      expect(src, `${relPath} must call assertSafeStepCount('${adapterId}', ...)`).toContain(
        `assertSafeStepCount('${adapterId}'`,
      );
      expect(src, `${relPath} must call formatLowStepGeometryWarning('${adapterId}', ...)`).toContain(
        `formatLowStepGeometryWarning('${adapterId}'`,
      );

      // Verify assertSafeStepCount executes inside run(ctx) BEFORE ensureEngine/ensureLoaded
      const runIdx = src.search(/\basync\s+run\s*\(/);
      expect(runIdx, `${relPath} missing async run(`).toBeGreaterThanOrEqual(0);
      const runBody = src.slice(runIdx);

      const assertIdx = runBody.indexOf(`assertSafeStepCount('${adapterId}'`);
      const loadIdx = runBody.search(/\bawait\s+(?:ensureEngine|ensureLoaded)\s*\(/);
      expect(assertIdx, `${relPath}: assertSafeStepCount not found inside run()`).toBeGreaterThanOrEqual(0);
      expect(loadIdx, `${relPath}: ensureEngine/ensureLoaded not found inside run()`).toBeGreaterThanOrEqual(0);
      expect(
        assertIdx,
        `${relPath}: assertSafeStepCount must run BEFORE downloading weights / initializing WebGPU`,
      ).toBeLessThan(loadIdx);
    }
  });

  it('keeps server/selfhost-server.mjs validateNodeStepPolicy in 100% parity with MODEL_STEP_POLICIES and rejects unsafe /api/run-node requests with HTTP 400', async () => {
    const { SERVER_STEP_POLICIES, validateNodeStepPolicy, createSelfhostServer } =
      await loadSelfhostModule();

    // 1. Structural parity of policy tables
    expect(Object.keys(SERVER_STEP_POLICIES).sort()).toEqual(
      Object.keys(MODEL_STEP_POLICIES).sort(),
    );

    for (const [adapterId, tsEnvelope] of Object.entries(MODEL_STEP_POLICIES)) {
      const srvEnvelope = SERVER_STEP_POLICIES[adapterId];
      expect(srvEnvelope, `Missing SERVER_STEP_POLICIES['${adapterId}']`).toBeDefined();
      expect(srvEnvelope.paramKey).toBe(tsEnvelope.paramKey);
      expect(srvEnvelope.modeParamKey).toBe(tsEnvelope.modeParamKey);
      expect(srvEnvelope.defaultMode).toBe(tsEnvelope.defaultMode);
      expect(srvEnvelope.rules).toHaveLength(tsEnvelope.rules.length);

      for (let i = 0; i < tsEnvelope.rules.length; i++) {
        const tsRule = tsEnvelope.rules[i];
        const srvRule = srvEnvelope.rules[i];
        expect(srvRule.modes).toEqual(tsRule.modes);
        expect(srvRule.minSafeSteps).toBe(tsRule.minSafeSteps);
        expect(srvRule.recommendedMinSteps).toBe(tsRule.recommendedMinSteps);
        expect(srvRule.upstreamDefault).toBe(tsRule.upstreamDefault);
        expect(srvRule.maxSteps).toBe(tsRule.maxSteps);
      }

      // 2. Behavioral parity across all modes & boundary values
      const modesToTest: Array<string | undefined> = [undefined];
      if (tsEnvelope.modeParamKey) {
        if (tsEnvelope.defaultMode) modesToTest.push(tsEnvelope.defaultMode);
        for (const r of tsEnvelope.rules) {
          for (const m of r.modes ?? []) modesToTest.push(m);
        }
        // Also test fallback mode label (e.g. 'ode', 'vf', 'unconditional', 'motif')
        modesToTest.push('ode', 'vf', 'unconditional', 'motif', 'partial');
      }

      for (const modeVal of new Set(modesToTest)) {
        const baseParams: Record<string, unknown> =
          tsEnvelope.modeParamKey && modeVal !== undefined
            ? { [tsEnvelope.modeParamKey]: modeVal }
            : {};
        const resolved = resolveStepRule(adapterId, baseParams)!;
        const { rule } = resolved;
        const testValues: unknown[] = [
          undefined,
          '',
          'not-a-number',
          rule.minSafeSteps - 1,
          rule.minSafeSteps,
          rule.recommendedMinSteps,
          rule.upstreamDefault,
          rule.maxSteps,
          rule.maxSteps + 1,
        ];
        for (const val of testValues) {
          const params = val === undefined ? baseParams : { ...baseParams, [tsEnvelope.paramKey]: val };
          const tsRes = validateStepCount(adapterId, params)!;
          const srvRes = validateNodeStepPolicy(adapterId, params);
          expect(
            srvRes.ok,
            `Parity mismatch for ${adapterId} (${JSON.stringify(params)}): server=${JSON.stringify(srvRes)}, ts=${JSON.stringify(tsRes)}`,
          ).toBe(tsRes.ok);
        }
      }
    }

    // Non-generator adapters pass through without error
    expect(validateNodeStepPolicy('web-protein-mpnn', { numSequences: 8 })).toEqual({ ok: true });

    // 3. End-to-end HTTP 400 rejection on POST /api/run-node before queueing
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'denovo-step-policy-'));
    const buildDir = path.join(tmpRoot, 'build');
    fs.mkdirSync(buildDir, { recursive: true });
    fs.writeFileSync(path.join(buildDir, 'index.html'), '<html>ok</html>');

    let runNodeCalled = false;
    const inst = createSelfhostServer({
      host: '127.0.0.1',
      port: 0,
      buildDir,
      apiToken: '',
    });
    inst.setRunnerBridge(
      {
        getHealth: async () => ({ ready: true, gpu: { fallback: false }, activeAdapter: null, error: null }),
        runNode: async () => {
          runNodeCalled = true;
          return { outputs: {}, elapsedMs: 1 };
        },
        cancelJob: async () => {},
        releaseGpu: async () => {},
      },
      { ready: true, gpu: { fallback: false }, activeAdapter: null, error: null },
    );

    await new Promise<void>((resolve) => inst.server.listen(0, '127.0.0.1', resolve));
    const addr = inst.server.address() as { port: number };
    try {
      const res = await rawHttpRequest({
        port: addr.port,
        method: 'POST',
        path: '/api/run-node',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node: { id: 'n1', adapterId: 'la-proteina', params: { mode: 'sde', steps: 100 } },
          inputs: {},
        }),
      });
      expect(res.status).toBe(400);
      const parsed = JSON.parse(res.body);
      expect(parsed.ok).toBe(false);
      expect(parsed.error).toMatch(/below the safe minimum \(200\)/);
      expect(runNodeCalled).toBe(false);

      const resBoltz = await rawHttpRequest({
        port: addr.port,
        method: 'POST',
        path: '/api/run-node',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node: { id: 'n2', adapterId: 'boltzgen', params: { numSteps: 50 } },
          inputs: {},
        }),
      });
      expect(resBoltz.status).toBe(400);
      expect(JSON.parse(resBoltz.body).error).toMatch(/below the safe minimum \(200\)/);
      expect(runNodeCalled).toBe(false);
    } finally {
      await new Promise<void>((resolve) => inst.server.close(() => resolve()));
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });

  it('rejects unsafe steps and accepts valid mode-aware steps in ?flow= DSL', () => {
    // Reject unsafe step counts
    const badFlows = [
      'https://denovo.run/?flow=laproteina:steps=100',
      'https://denovo.run/?flow=boltzgen:numSteps=50',
      'https://denovo.run/?flow=protpardelle:numSteps=100',
      'https://denovo.run/?flow=genie2:steps=500',
    ];
    for (const url of badFlows) {
      const decoded = decodeFlowQuery(url);
      expect(decoded.ok, `Expected ${url} to be rejected`).toBe(false);
      expect(decoded.errors.map((e) => e.message).join(' ')).toMatch(
        /minimum|must be between|must be >=|safe minimum/i,
      );
    }

    // Accept valid mode-aware step counts
    const odeOk = decodeFlowQuery('https://denovo.run/?flow=laproteina:mode=ode,steps=80');
    expect(odeOk.ok, JSON.stringify(odeOk.errors)).toBe(true);
    expect(odeOk.graph?.nodes[0].params.mode).toBe('ode');
    expect(odeOk.graph?.nodes[0].params.steps).toBe(80);

    const partialOk = parseFlowDsl('pdb:pdbId=1UBQ~protpardelle:taskMode=partial,numSteps=80');
    expect(partialOk.ok, JSON.stringify(partialOk.errors)).toBe(true);
    expect(partialOk.graph?.nodes[1].params.taskMode).toBe('partial');
    expect(partialOk.graph?.nodes[1].params.numSteps).toBe(80);
  });
});
