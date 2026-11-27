// Regression tests for the ten CodeQL code-scanning alerts fixed on `fix-codeql-alerts`:
// ReDoS (#1–#5), cleartext token storage (#6), stack-trace exposure (#7), token hashing (#8),
// incomplete backslash escaping (#9), and the identity `rfdiv(` replacement (#10).
import { afterEach, describe, expect, it, vi } from 'vitest';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { parseFlowDsl } from '$lib/engine/flow-dsl';
import { reconcilePlannedWorkflow, type WorkflowPlan } from '$lib/engine/workflow-planner';
import { buildCanonicalConditionIntent, compileConditionToRfd4 } from '$lib/engine/conditioning';
import {
  COMPUTE_BACKEND_STORAGE_KEY,
  defaultComputeBackendConfig,
  loadComputeBackendConfig,
  saveComputeBackendConfig,
} from '$lib/engine/compute-backend';

const REDOS_BUDGET_MS = 20;

/** Run `fn` once to warm up, then time a second run with the adversarial input. */
function timed<T>(warmup: () => unknown, fn: () => T): { value: T; ms: number } {
  warmup();
  const t0 = performance.now();
  const value = fn();
  return { value, ms: performance.now() - t0 };
}

function vendorModule(rel: string) {
  return import(pathToFileURL(path.resolve(process.cwd(), 'static/vendor', rel)).href);
}

describe('CodeQL #1/#2 (js/redos): flow-dsl semicolon continuation regexes', () => {
  // Outcome of parsing `<param>=<first>;<rest>`: the param value when the flow parses, else the
  // error codes. A continued `;` keeps `rest` inside the param; otherwise `rest` is a new statement.
  // Expected values were recorded from the pre-fix regexes.
  const outcome = (dsl: string, nodeId: string, key: string) => {
    const r = parseFlowDsl(dsl);
    return r.ok ? r.graph?.nodes.find((n) => n.id === nodeId)?.params[key] : r.errors.map((e) => e.code).join(',');
  };
  const tied = (rest: string) => outcome(`src:pdb:pdbId=1PRW~m1:mpnn:tiedPositions=A1=B25;${rest}`, 'm1', 'tiedPositions');
  const fixedAtoms = (rest: string) =>
    outcome(`src:pdb:pdbId=1CTT~r3:rfd3:unindex=A102,A104,fixedAtoms=A102:ND1,CG;${rest}`, 'r3', 'fixedAtoms');

  it('keeps tiedPositions continuation decisions identical on valid and near-miss inputs', () => {
    expect(tied('A2=B26')).toBe('A1=B25;A2=B26');
    expect(tied('A2=B26=C27')).toBe('A1=B25;A2=B26=C27');
    expect(tied('A-2=B-26')).toBe('A1=B25;A-2=B-26');
    expect(tied('12=34')).toBe('INVALID_PARAM'); // continued, then rejected by the param validator
    expect(tied('A=B')).toBe('UNKNOWN_MODEL');
    expect(tied('1=2')).toBe('UNKNOWN_MODEL');
    expect(tied('A2')).toBe('UNKNOWN_MODEL');
    expect(tied('A2=B26x')).toBe('UNKNOWN_MODEL');
    expect(tied('m2:mpnn')).toBe('MISSING_REQUIRED_INPUT');
  });

  it('keeps atom-map continuation decisions identical on valid and near-miss inputs', () => {
    expect(fixedAtoms('A104:OE1,CD')).toBe('A102:ND1,CG;A104:OE1,CD');
    expect(fixedAtoms('A104:OE1, CD')).toBe('A102:ND1,CG;A104:OE1, CD');
    expect(fixedAtoms('A104 : OE1 , CD')).toBe('A102:ND1,CG;A104 : OE1 , CD');
    expect(fixedAtoms('A104-106:N,CA')).toBe('A102:ND1,CG;A104-106:N,CA');
    expect(fixedAtoms("A104:O2',C*")).toBe("A102:ND1,CG;A104:O2',C*");
    expect(fixedAtoms('A104:OE1,,CD')).toBe('A102:ND1,CG;A104:OE1,,CD');
    expect(fixedAtoms('A104:,OE1')).toBe('A102:ND1,CG;A104:,OE1');
    expect(fixedAtoms('A104:OE1 , ,CD')).toBe('A102:ND1,CG;A104:OE1 , ,CD');
    // Whitespace not attached to a separator comma, no residue number, or a model alias: new statement.
    expect(fixedAtoms('A104:OE1 CD')).toBe('UNKNOWN_MODEL');
    expect(fixedAtoms('A104:OE1 , , CD')).toBe('INVALID_PARAM_SYNTAX');
    expect(fixedAtoms('A:OE1')).toBe('UNKNOWN_MODEL');
    expect(fixedAtoms('A104:mpnn')).toBe('MISSING_REQUIRED_INPUT');
  });

  it('rejects exponential-backtracking payloads in linear time', () => {
    const warm = () => tied('A2=B26');
    // `0000=` repeated: the old `[A-Za-z0-9]+-?\d+` token needed ~6 s at 18 repeats.
    const tiedPayload = '00=' + '0000='.repeat(40) + '!';
    const tiedAttack = timed(warm, () => tied(tiedPayload));
    expect(tiedAttack.value).not.toBe(`A1=B25;${tiedPayload}`);
    expect(tiedAttack.ms).toBeLessThan(REDOS_BUDGET_MS);

    // `','` repeated: the old atom list needed ~250 ms at 22 repeats, doubling with each one.
    const atomPayload = "00:'," + "','".repeat(40) + ' !';
    const atomAttack = timed(warm, () => fixedAtoms(atomPayload));
    expect(atomAttack.value).not.toBe(`A102:ND1,CG;${atomPayload}`);
    expect(atomAttack.ms).toBeLessThan(REDOS_BUDGET_MS);
  });
});

describe('CodeQL #3 (js/redos): workflow-planner "omit amino acids" separators', () => {
  const plan = {
    kind: 'conditioned_design',
    designTask: 'binder',
    benchmarkId: 'pdl1_5o45',
    pdbId: '5O45',
    crop: null,
    generators: ['rfd4-proteina'],
    length: 65,
    numDesigns: 1,
    sequenceDesigners: ['antifold'],
    numSequences: 2,
    topK: null,
    refold: true,
    foldModels: ['multimer'],
    score: true,
    stabilityScan: false,
    summary: 'omit',
  } as unknown as WorkflowPlan;
  const omitFor = (clause: string) =>
    (reconcilePlannedWorkflow(plan, `Design a 65aa binder to 5O45 with RFD4 and SolubleMPNN ${clause}`) as {
      omitAminoAcids?: string | null;
    }).omitAminoAcids;

  it('extracts the same residues for every separator form', () => {
    expect(omitFor('omitting C')).toBe('C');
    expect(omitFor('omit C, W')).toBe('CW');
    expect(omitFor('omit C,W and M')).toBe('CWM');
    expect(omitFor('omit amino acids C or W')).toBe('CW');
    expect(omitFor('omit C  W\tM')).toBe('CWM');
    expect(omitFor('omit c and w')).toBe('CW');
    expect(omitFor('omit C W X')).toBe('CW');
    expect(omitFor('omit C , and W')).toBe('C');
    expect(omitFor('omit Cand W')).toBe('W');
  });

  it('handles long whitespace-separated runs in linear time', () => {
    const attack = timed(
      () => omitFor('omit C'),
      () => omitFor('omit a' + '  a'.repeat(2000) + 'b'),
    );
    expect(attack.value).toBe('A');
    expect(attack.ms).toBeLessThan(REDOS_BUDGET_MS * 5);
  });
});

describe('CodeQL #4 (js/polynomial-redos): rfd3 safetensors header tokenizer', () => {
  function safetensors(header: string, dataBytes: number): ArrayBuffer {
    const h = new TextEncoder().encode(header);
    const buf = new ArrayBuffer(8 + h.length + dataBytes);
    new DataView(buf).setBigUint64(0, BigInt(h.length), true);
    new Uint8Array(buf, 8, h.length).set(h);
    return buf;
  }
  const tensor = '"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}';

  it('still parses valid headers and detects duplicate keys behind escapes', async () => {
    const { parseSafetensors } = await vendorModule('rfd3/shared/shard.js');
    const meta = String.raw`"__metadata__":{"path":"C:\\dir\\","q\"{[,:]}":"x\u0022y","p":"\u2028"}`;
    const ok = parseSafetensors(safetensors(`{${meta}, ${tensor}}`, 8));
    expect([...ok.keys()]).toEqual(['w']);
    expect(ok.get('w').shape).toEqual([2]);

    expect(() => parseSafetensors(safetensors(`{${tensor},"\\u0077":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 8))).toThrow(
      /Duplicate safetensors key: w/,
    );
    // A value ending in an escaped backslash must not hide the duplicate key after it.
    expect(() =>
      parseSafetensors(safetensors(String.raw`{"__metadata__":{"p":"x\\","p":"y"},` + tensor + '}', 8)),
    ).toThrow(/Duplicate safetensors key: p/);
    expect(() =>
      parseSafetensors(safetensors(String.raw`{"__metadata__":{"p\"":"x","p":"y"},` + tensor + '}', 8)),
    ).not.toThrow();
  });

  it('tokenizes escape-heavy headers in linear time', async () => {
    const { parseSafetensors } = await vendorModule('rfd3/shared/shard.js');
    const header = `{"__metadata__":{"k":"${'\\"'.repeat(200_000)}"},${tensor}}`;
    const attack = timed(
      () => parseSafetensors(safetensors(`{${tensor}}`, 8)),
      () => parseSafetensors(safetensors(header, 8)),
    );
    expect(attack.value.has('w')).toBe(true);
    expect(attack.ms).toBeLessThan(REDOS_BUDGET_MS);
  });
});

describe('CodeQL #5 (js/polynomial-redos): rfd3 compact residue selections', () => {
  const PDB = [
    'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  ',
    'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  ',
    'ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  ',
    'ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O  ',
    'ATOM      5  CB  ALA A   1       1.986  -0.773  -1.209  1.00 20.00           C  ',
    'ATOM      6  N   GLY A   2       3.320   1.510   0.000  1.00 20.00           N  ',
    'ATOM      7  CA  GLY A   2       3.960   2.810   0.000  1.00 20.00           C  ',
    'ATOM      8  C   GLY A   2       5.470   2.700   0.000  1.00 20.00           C  ',
    'ATOM      9  O   GLY A   2       6.050   1.620   0.000  1.00 20.00           O  ',
    'ATOM     10  N   SER A   3       6.100   3.870   0.000  1.00 20.00           N  ',
    'ATOM     11  CA  SER A   3       7.550   3.960   0.000  1.00 20.00           C  ',
    'ATOM     12  C   SER A   3       8.100   5.380   0.000  1.00 20.00           C  ',
    'ATOM     13  O   SER A   3       7.340   6.350   0.000  1.00 20.00           O  ',
    'ATOM     14  CB  SER A   3       8.080   3.190  -1.210  1.00 20.00           C  ',
    'ATOM     15  OG  SER A   3       9.480   3.280  -1.310  1.00 20.00           O  ',
    'END',
    '',
  ].join('\n');

  // Expected values were recorded from the pre-fix lazy `([A-Za-z0-9]+?)(-?\d+)` regexes.
  it('selects the same residues for compact and colon residue ids', async () => {
    const { buildInput } = await vendorModule('rfd3/shared/input.js');
    const tokens = (contig: string) => buildInput({ mode: 'motif', structure: PDB, contig }).tokens.length;
    expect(tokens('A1-3')).toBe(3);
    expect(tokens('A2-3')).toBe(2);
    expect(tokens('A3')).toBe(1);
    expect(tokens('A:1-2')).toBe(2);
    expect(tokens('A1-1,A3')).toBe(2);
    expect(tokens('A-1-3')).toBe(3);
    expect(tokens('10,A2,5')).toBe(16);
    expect(() => tokens('A3-2')).toThrow(/Unresolved contig component A3-2/);
    expect(() => tokens('A1--3')).toThrow(/Unresolved contig component A1--3/);
    expect(() => tokens('B1-3')).toThrow(/Unresolved contig component B1-3/);

    const fixed = (fixedAtoms: unknown) =>
      buildInput({ mode: 'motif', structure: PDB, contig: 'A1-3', fixedAtoms }).atoms.filter(
        (a: { fixed?: boolean }) => a.fixed,
      ).length;
    expect(fixed('A2-3')).toBe(10);
    expect(fixed('A1')).toBe(5);
    expect(fixed('A1-2,A3')).toBe(15);
    expect(fixed('A:2')).toBe(4);
    expect(fixed({ A1: 'N,CA', 'A:3': 'BKBN' })).toBe(6);
  });

  it('rejects long digit-run selections in linear time', async () => {
    const { buildInput } = await vendorModule('rfd3/shared/input.js');
    const attack = '0'.repeat(50_000) + 'x';
    const run = timed(
      () => buildInput({ mode: 'motif', structure: PDB, contig: 'A1-3' }),
      () => {
        try {
          buildInput({ mode: 'motif', structure: PDB, contig: attack });
          return 'resolved';
        } catch (err) {
          return (err as Error).message.slice(0, 29);
        }
      },
    );
    expect(run.value).toBe('Unresolved contig component 0');
    expect(run.ms).toBeLessThan(REDOS_BUDGET_MS);
  });
});

describe('CodeQL #6 (js/clear-text-storage-of-sensitive-data): compute backend API token', () => {
  const LEGACY_SESSION_TOKEN_KEY = 'denovo.computeBackend.sessionToken.v1';
  const restorers: Array<() => void> = [];

  function fakeStorage(name: 'localStorage' | 'sessionStorage') {
    const mem = new Map<string, string>();
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    const setItem = vi.fn((k: string, v: string) => void mem.set(k, v));
    Object.defineProperty(globalThis, name, {
      configurable: true,
      value: { getItem: (k: string) => mem.get(k) ?? null, setItem, removeItem: (k: string) => mem.delete(k) },
    });
    restorers.push(() => {
      if (original) Object.defineProperty(globalThis, name, original);
      else delete (globalThis as Record<string, unknown>)[name];
    });
    return { mem, setItem };
  }

  afterEach(() => {
    while (restorers.length) restorers.pop()!();
  });

  it('never writes the Bearer token to localStorage or sessionStorage', () => {
    const local = fakeStorage('localStorage');
    const session = fakeStorage('sessionStorage');
    for (const serverUrl of ['https://gpu-box.example.ts.net:8790', '']) {
      saveComputeBackendConfig({ ...defaultComputeBackendConfig(), mode: 'remote', serverUrl, apiToken: 'dnv_secret_123' });
    }
    const written = [...local.setItem.mock.calls, ...session.setItem.mock.calls].map((c) => c.join('='));
    expect(written.join('\n')).not.toContain('dnv_secret_123');
    expect(session.mem.size).toBe(0);
    expect(JSON.parse(local.mem.get(COMPUTE_BACKEND_STORAGE_KEY)!)).toEqual({
      mode: 'remote',
      serverUrl: '',
      hybridThresholdMB: 40,
      maxFoldResidues: null,
    });
  });

  it('keeps the token in memory only and purges tokens persisted by earlier builds', () => {
    const local = fakeStorage('localStorage');
    const session = fakeStorage('sessionStorage');
    const serverUrl = 'https://gpu-box.example.ts.net:8790';
    local.mem.set(COMPUTE_BACKEND_STORAGE_KEY, JSON.stringify({ mode: 'remote', serverUrl, apiToken: 'legacy_local' }));
    session.mem.set(
      LEGACY_SESSION_TOKEN_KEY,
      JSON.stringify({ origin: 'https://gpu-box.example.ts.net:8790', token: 'legacy_session' }),
    );

    const loaded = loadComputeBackendConfig();
    expect(loaded).toMatchObject({ mode: 'remote', serverUrl, apiToken: '' });
    expect(session.mem.has(LEGACY_SESSION_TOKEN_KEY)).toBe(false);
    expect(local.mem.get(COMPUTE_BACKEND_STORAGE_KEY)).not.toContain('legacy_local');
  });
});

describe('CodeQL #7/#8 (js/stack-trace-exposure, js/insufficient-password-hash): self-host server', () => {
  const TOKEN = 'dnv_codeql_regression_token_42';

  async function startServer(bridge: Record<string, unknown>) {
    const serverPath = path.resolve(process.cwd(), '../server/selfhost-server.mjs');
    const { createSelfhostServer } = createRequire(import.meta.url)(serverPath);
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'denovo-codeql-test-'));
    const buildDir = path.join(root, 'build');
    fs.mkdirSync(buildDir);
    fs.writeFileSync(path.join(buildDir, 'index.html'), '<html>ok</html>');
    const inst = createSelfhostServer({ host: '127.0.0.1', port: 0, buildDir, apiToken: TOKEN });
    const health = { ready: true, gpu: { vendor: 'nvidia', architecture: 'blackwell', description: 'nvidia', fallback: false }, activeAdapter: null, error: null };
    inst.setRunnerBridge({ getHealth: async () => health, runNode: async () => ({}), cancel: async () => {}, releaseGpu: async () => {}, ...bridge }, health);
    await new Promise<void>((r) => inst.server.listen(0, '127.0.0.1', r));
    const port = (inst.server.address() as { port: number }).port;
    const request = (method: string, urlPath: string, headers: Record<string, string> = {}, body?: string) =>
      new Promise<{ status: number; json: { ok?: boolean; error?: string } }>((resolve, reject) => {
        const req = http.request({ host: '127.0.0.1', port, method, path: urlPath, headers }, (res) => {
          const chunks: Buffer[] = [];
          res.on('data', (c) => chunks.push(Buffer.from(c)));
          res.on('end', () => resolve({ status: res.statusCode ?? 0, json: JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}') }));
        });
        req.on('error', reject);
        if (body !== undefined) req.write(body);
        req.end();
      });
    const close = async () => {
      await new Promise<void>((r) => inst.server.close(() => r()));
      fs.rmSync(root, { recursive: true, force: true });
    };
    return { request, close };
  }

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('#8: compares Bearer tokens in constant time over the secret length without hashing them', async () => {
    const srv = await startServer({});
    const createHash = vi.spyOn(crypto, 'createHash');
    const timingSafeEqual = vi.spyOn(crypto, 'timingSafeEqual');
    try {
      const health = (token: string) => srv.request('GET', '/api/health', { Authorization: `Bearer ${token}` });
      expect((await health(TOKEN)).status).toBe(200);
      for (const wrong of ['x', TOKEN.slice(0, -1), TOKEN.slice(0, -1) + 'X', TOKEN + 'X', TOKEN.repeat(4)]) {
        expect((await health(wrong)).status).toBe(401);
      }
      // A NUL-padded token must not alias the real one (as it would with a zero-padded HMAC key).
      const nulPadded = await srv.request(
        'POST',
        '/api/auth',
        { 'Content-Type': 'application/json' },
        JSON.stringify({ token: `${TOKEN}\u0000` }),
      );
      expect(nulPadded.status).toBe(401);

      const secretBytes = Buffer.byteLength(TOKEN);
      expect(timingSafeEqual.mock.calls.length).toBeGreaterThanOrEqual(7);
      for (const [a, b] of timingSafeEqual.mock.calls) {
        expect([(a as Buffer).length, (b as Buffer).length]).toEqual([secretBytes, secretBytes]);
      }
      expect(createHash).not.toHaveBeenCalled();
    } finally {
      await srv.close();
    }
  });

  it('#7: answers internal failures with a generic message and logs the details server-side', async () => {
    const internal = new Error('EACCES: permission denied, open /home/alice/.cache/denovo/profile/lock');
    internal.stack = `${internal.message}\n    at Object.openSync (node:fs:573:18)\n    at releaseGpu (/srv/runner.mjs:42:7)`;
    let fail: unknown = internal;
    const srv = await startServer({
      releaseGpu: async () => {
        throw fail;
      },
    });
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    const auth = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' };
    try {
      const res = await srv.request('POST', '/api/release-gpu', auth, '{}');
      expect(res).toEqual({ status: 500, json: { ok: false, error: 'Failed to release host GPU.' } });
      expect(logged).toHaveBeenCalledWith('[denovo-selfhost] API request failed:', internal);

      fail = 'raw string with /etc/denovo/secret path';
      const nonError = await srv.request('POST', '/api/release-gpu', auth, '{}');
      expect(nonError.json.error).toBe('Failed to release host GPU.');

      // Errors the server raises on purpose (4xx with statusCode) keep their actionable message.
      const badJson = await srv.request('POST', '/api/cancel', auth, '{"jobId":');
      expect(badJson.status).toBe(400);
      expect(badJson.json.error).toMatch(/^Invalid JSON body: /);
      const tooBig = await srv.request('POST', '/api/auth', { ...auth, 'Content-Length': String(10 * 1024 * 1024) }, '{}');
      expect(tooBig.status).toBe(413);
      expect(tooBig.json.error).toMatch(/^Payload exceeds limit/);
    } finally {
      await srv.close();
    }
  });
});

describe('CodeQL #9 (js/incomplete-sanitization): RFD4 sel() quoting', () => {
  // Mirrors the quoted-string rule of RFD4's query tokenizer (rfd4p/pipeline/spec.js
  // evalSelection): inside either quote style, `\x` reads as `x`.
  function readSelArg(expr: string): string {
    const m = /^sel\((["'])/.exec(expr);
    if (!m) throw new Error(`not a sel() expression: ${expr}`);
    let out = '';
    let i = m[0].length;
    for (; i < expr.length && expr[i] !== m[1]; i++) {
      if (expr[i] === '\\' && i + 1 < expr.length) i++;
      out += expr[i];
    }
    expect(expr.slice(i)).toBe(`${m[1]})`);
    return out;
  }
  const donorSelect = (hbondDonorSelection: string) =>
    (compileConditionToRfd4(buildCanonicalConditionIntent({ contig: '110-110', hbondDonorSelection })).conditions
      .C_HB as { DONOR: Array<{ select: string }> }).DONOR[0].select;

  it('leaves ordinary selections byte-identical', () => {
    expect(donorSelect('A/SAM/*/[N,N6]')).toBe("sel('A/SAM/*/[N,N6]')");
    expect(donorSelect("A/SAM/*/[N,N6,O2',O3']")).toBe(`sel("A/SAM/*/[N,N6,O2',O3']")`);
  });

  it('escapes backslashes before quotes so the selection round-trips through the RFD4 tokenizer', () => {
    for (const raw of ["A/X/*/O2'\\", 'A/X/*/O\\N', "A/X/*/[O2',N\\\"]", 'A/X/*/\\', "A/X/*/O2'\\\\\""]) {
      expect(readSelArg(donorSelect(raw))).toBe(raw);
    }
    expect(donorSelect("A/X/*/O2'\\")).toBe(`sel("A/X/*/O2'\\\\")`);
  });
});

describe('CodeQL #10 (js/identity-replacement): genie2 scalar JS bridge', () => {
  it('passes rfdiv() through unchanged and still runs on the CPU kernel prelude', async () => {
    const { scalarJS } = await vendorModule('genie2/shared/portable.js');
    const js = scalarJS('var x: f32 = rfdiv(a0[i], 2.0); o[i] = fma(x, 2.0, 1.0);');
    expect(js).toBe('let x = rfdiv(a0[i], 2.0); o[i] = rfFma(x, 2.0, 1.0);');
    const { CPUAlgebra } = await vendorModule('genie2/cpu/algebra.js');
    const ops = new CPUAlgebra();
    const input = ops.upload([3], new Float32Array([2, 4, 6]));
    expect([...ops.kernel([3], [input], js).data]).toEqual([3, 5, 7]);
  });
});
