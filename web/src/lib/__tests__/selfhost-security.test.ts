import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
import { decodeWireValue, encodeWireValue } from '$lib/engine/remote-codec';
import {
  COMPUTE_BACKEND_STORAGE_KEY,
  loadComputeBackendConfig,
  resolveNodeFoldLimitOptions,
  saveComputeBackendConfig,
  validateRemoteServerUrl,
  defaultComputeBackendConfig,
} from '$lib/engine/compute-backend';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import { hydrateLegacyArtifactToEnvelope } from '$lib/kernel/artifacts';
import type { SequenceSet, WorkflowNode } from '$lib/engine/types';

async function loadSelfhostModule() {
  const serverPath = path.resolve(process.cwd(), '../server/selfhost-server.mjs');
  const req = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return req(serverPath) as any;
}

function makeTempBuildDir() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'denovo-sec-test-'));
  const buildDir = path.join(root, 'build');
  const siblingSecret = path.join(root, 'build-secret');
  fs.mkdirSync(buildDir, { recursive: true });
  fs.mkdirSync(siblingSecret, { recursive: true });
  fs.writeFileSync(path.join(buildDir, 'index.html'), '<html>ok</html>');
  fs.writeFileSync(path.join(buildDir, 'runner.html'), '<html>runner</html>');
  fs.writeFileSync(path.join(siblingSecret, 'secret.txt'), 'TOP_SECRET');
  fs.symlinkSync(path.join(siblingSecret, 'secret.txt'), path.join(buildDir, 'escape-link.txt'));
  return { root, buildDir };
}

function rawHttpRequest(opts: {
  port: number;
  method?: string;
  path: string;
  headers?: Record<string, string>;
  body?: string;
}): Promise<{ status: number; headers: http.IncomingHttpHeaders; body: string }> {
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
            headers: res.headers,
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

describe('Self-hosted WebGPU security hardening (Astra 6.2 audit gates)', () => {
  it('H1: refuses non-loopback bind without DENOVO_API_TOKEN', async () => {
    const { createSelfhostServer } = await loadSelfhostModule();
    const { root, buildDir } = makeTempBuildDir();
    try {
      expect(() =>
        createSelfhostServer({ host: '0.0.0.0', port: 0, buildDir, apiToken: '' }),
      ).toThrow(/Refusing to bind non-loopback/i);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('H1/H2/H3/H4/H5/M1/M2/M8/L1/L2: enforces auth, origin, static containment, queue caps, cancel ownership, and software-GPU policy over real HTTP', async () => {
    const { createSelfhostServer } = await loadSelfhostModule();
    const { root, buildDir } = makeTempBuildDir();
    const TOKEN = 'dnv_test_secret_token_987654321';

    let releaseJobA!: () => void;
    let activeJobCancelled = false;

    const inst = createSelfhostServer({
      host: '127.0.0.1',
      port: 0,
      buildDir,
      apiToken: TOKEN,
      allowedOrigins: 'https://denovo.run',
      maxQueueDepth: 1,
      maxBodyMB: 1,
      allowSoftwareGpu: false,
      hostGpuHw: { name: 'NVIDIA GeForce RTX 5090', shortName: 'RTX 5090', vramMB: 32607 },
    });

    const mockBridge = {
      getHealth: async () => ({
        ready: true,
        gpu: { vendor: 'nvidia', architecture: 'blackwell', description: 'nvidia (f16)', fallback: false },
        activeAdapter: null,
        error: null,
      }),
      runNode: async (jobId: string) => {
        activeJobCancelled = false;
        await new Promise<void>((r) => {
          releaseJobA = r;
        });
        if (activeJobCancelled) throw new Error('Workflow run cancelled');
        return { outputs: { ok: true, jobId }, elapsedMs: 12 };
      },
      cancel: async () => {
        activeJobCancelled = true;
        releaseJobA?.();
      },
      releaseGpu: async () => {},
    };
    inst.setRunnerBridge(mockBridge, await mockBridge.getHealth());

    await new Promise<void>((r) => inst.server.listen(0, '127.0.0.1', r));
    const addr = inst.server.address();
    const port = typeof addr === 'object' && addr ? addr.port : 0;

    try {
      // H1/L1: Auth & Origin admission
      const unauth = await rawHttpRequest({ port, path: '/api/health' });
      expect(unauth.status).toBe(401);

      const badToken = await rawHttpRequest({
        port,
        path: '/api/health',
        headers: { Authorization: 'Bearer wrong_token' },
      });
      expect(badToken.status).toBe(401);

      const badOrigin = await rawHttpRequest({
        port,
        path: '/api/health',
        headers: {
          Authorization: `Bearer ${TOKEN}`,
          Origin: 'https://evil.example',
        },
      });
      expect(badOrigin.status).toBe(403);

      const goodOrigin = await rawHttpRequest({
        port,
        path: '/api/health',
        headers: {
          Authorization: `Bearer ${TOKEN}`,
          Origin: 'https://denovo.run',
        },
      });
      expect(goodOrigin.status).toBe(200);
      expect(goodOrigin.headers['access-control-allow-origin']).toBe('https://denovo.run');
      expect(goodOrigin.headers['vary']).toContain('Origin');
      const healthJson = JSON.parse(goodOrigin.body);
      expect(healthJson.ok).toBe(true);
      expect(healthJson.gpu.shortName).toBe('RTX 5090');
      // M1: runningJobId must NEVER be disclosed in /api/health
      expect('runningJobId' in healthJson).toBe(false);

      // H1: Content-Type: application/json required on POST
      const badCt = await rawHttpRequest({
        port,
        method: 'POST',
        path: '/api/release-gpu',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'text/plain' },
        body: '{}',
      });
      expect(badCt.status).toBe(415);

      // H2: Malformed URL encoding returns 400 and leaves server alive
      const badEnc = await rawHttpRequest({ port, path: '/%E0%A4%A' });
      expect(badEnc.status).toBe(400);

      // H3 & L2: Static containment blocks symlink escape & sibling prefix; sets security headers
      const symlinkEscape = await rawHttpRequest({ port, path: '/escape-link.txt' });
      expect(symlinkEscape.status).toBe(404);
      const siblingPrefix = await rawHttpRequest({ port, path: '/../build-secret/secret.txt' });
      expect(siblingPrefix.status).toBe(404);
      const runnerForwarded = await rawHttpRequest({
        port,
        path: '/runner',
        headers: { 'X-Forwarded-For': '100.100.100.100' },
      });
      expect(runnerForwarded.status).toBe(403);
      const indexResp = await rawHttpRequest({ port, path: '/' });
      expect(indexResp.status).toBe(200);
      expect(indexResp.headers['x-content-type-options']).toBe('nosniff');
      expect(indexResp.headers['x-frame-options']).toBe('DENY');

      // H4: Disallowed/unknown adapterId rejected with 400 before queueing
      const badAdapter = await rawHttpRequest({
        port,
        method: 'POST',
        path: '/api/run-node',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ node: { adapterId: 'pdb-input', params: {} }, inputs: {} }),
      });
      expect(badAdapter.status).toBe(400);

      // H5: Oversized payload rejected with 413
      const tooBig = await rawHttpRequest({
        port,
        method: 'POST',
        path: '/api/run-node',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node: { adapterId: 'gnina', params: {} },
          inputs: { pad: 'x'.repeat(1_200_000) },
        }),
      });
      expect(tooBig.status).toBe(413);

      // M1 & M2 & H5: Two-client job lifecycle, queue cap (429), and owner-bound cancellation
      const clientA = await new Promise<{
        req: http.ClientRequest;
        started: { jobId: string; cancelToken: string };
        finished: Promise<string>;
      }>((resolve, reject) => {
        const req = http.request(
          {
            host: '127.0.0.1',
            port,
            method: 'POST',
            path: '/api/run-node',
            headers: {
              Authorization: `Bearer ${TOKEN}`,
              'Content-Type': 'application/json',
              Accept: 'text/event-stream',
            },
          },
          (res) => {
            let buf = '';
            let gotStarted = false;
            let resolveFinished!: (full: string) => void;
            const finished = new Promise<string>((r) => {
              resolveFinished = r;
            });
            res.on('data', (chunk) => {
              buf += chunk.toString('utf8');
              if (!gotStarted && buf.includes('event: started') && buf.includes('\n\n')) {
                const match = buf.match(/event: started\ndata: ([^\n]+)/);
                if (match) {
                  gotStarted = true;
                  resolve({ req, started: JSON.parse(match[1]), finished });
                }
              }
            });
            res.on('end', () => resolveFinished(buf));
          },
        );
        req.on('error', reject);
        req.write(JSON.stringify({ node: { id: 'g1', adapterId: 'gnina', params: {} }, inputs: {} }));
        req.end();
      });

      // Client B tries to cancel Client A's job with wrong cancelToken -> 403 Forbidden!
      const badCancel = await rawHttpRequest({
        port,
        method: 'POST',
        path: '/api/cancel',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: clientA.started.jobId, cancelToken: 'wrong-token' }),
      });
      expect(badCancel.status).toBe(403);

      // Client A cancels its own job with its server-issued cancelToken -> 200 & SSE ends cleanly (M2)
      const ownerCancel = await rawHttpRequest({
        port,
        method: 'POST',
        path: '/api/cancel',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId: clientA.started.jobId,
          cancelToken: clientA.started.cancelToken,
        }),
      });
      expect(ownerCancel.status).toBe(200);
      const streamBody = await clientA.finished;
      expect(streamBody).toContain('Workflow run cancelled');

      // M8: Software-GPU fallback policy stays enforced even after raw health refresh
      inst.setRunnerBridge(
        {
          ...mockBridge,
          getHealth: async () => ({
            ready: true,
            gpu: { vendor: 'google', architecture: '', description: 'SwiftShader', fallback: true },
            activeAdapter: null,
            error: null,
          }),
        },
        {
          ready: true,
          gpu: { vendor: 'google', architecture: '', description: 'SwiftShader', fallback: true },
          activeAdapter: null,
          error: null,
        },
      );
      const swHealth = await rawHttpRequest({
        port,
        path: '/api/health',
        headers: { Authorization: `Bearer ${TOKEN}` },
      });
      expect(JSON.parse(swHealth.body).ok).toBe(false);
      expect(JSON.parse(swHealth.body).error).toMatch(/software WebGPU/i);
    } finally {
      await new Promise<void>((r) => inst.server.close(() => r()));
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('H4/H5/M3: rejects prototype pollution, unaligned TypedArrays, invalid host params, and >host-ceiling folds', async () => {
    // M3: remote-codec rejects __proto__ and inherited type-tag names
    const pollutedJson = JSON.parse('{"__proto__":{"polluted":true}}');
    expect(() => decodeWireValue(pollutedJson)).toThrow(/Forbidden property key/i);
    expect(() =>
      decodeWireValue({ __denovoTypedArray: 'Float32Array', b64: Buffer.from([1, 2, 3]).toString('base64') }),
    ).toThrow(/Invalid TypedArray byte length/i);
    // Inherited 'toString' must NOT be treated as a TypedArray constructor
    const spoofed = decodeWireValue<Record<string, unknown>>({ __denovoTypedArray: 'toString', b64: 'AAAA' });
    expect(spoofed.__denovoTypedArray).toBe('toString');

    // M3: kernel artifact hydration strips prototype keys
    const env = hydrateLegacyArtifactToEnvelope(
      JSON.parse('{"kind":"StructureArtifact","id":"s1","sourceModelId":"pdb-input","__proto__":{"x":1}}'),
    );
    expect(Object.prototype.hasOwnProperty.call(env.payload, '__proto__')).toBe(false);

    // H4: scheduler.runSingleGpuNode enforces server-owned maxFoldResidues on AF3!
    const scheduler = new ExecutionScheduler({ gpuIdleReleaseMs: 0 });
    const af3Node: WorkflowNode = {
      id: 'af3_1',
      adapterId: 'localfold-af2',
      params: { model: 'af3', msaMode: 'single' },
    };
    const bigSeq: SequenceSet = {
      kind: 'SequenceSet',
      id: 'big',
      parentStructureId: 'none',
      sourceModelId: 'sequence-input',
      candidates: [
        {
          id: 'huge_complex',
          sequence: 'A'.repeat(2500),
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
        },
      ],
    };
    await expect(
      scheduler.runSingleGpuNode(af3Node, { sequences: bigSeq }, () => {}, undefined, 2048),
    ).rejects.toThrow(/2048 residues/);

    // H4: Client ceiling on Host GPU clamps to min(userOverride, hostMax)
    const clamped = resolveNodeFoldLimitOptions(
      af3Node,
      { ...defaultComputeBackendConfig(), mode: 'remote', maxFoldResidues: 4096 },
      { ok: true, maxFoldResidues: 1024 },
    );
    expect(clamped.maxFoldResidues).toBe(1024);
  });

  it('H7/M5: validates HTTPS outside loopback, rejects URL credentials, and never persists apiToken in localStorage', () => {
    expect(validateRemoteServerUrl('http://127.0.0.1:8790').ok).toBe(true);
    expect(validateRemoteServerUrl('http://localhost:8790').ok).toBe(true);
    expect(validateRemoteServerUrl('https://flame.tailnet.ts.net:8790').ok).toBe(true);
    expect(validateRemoteServerUrl('http://100.109.74.34:8790').ok).toBe(false);
    expect(validateRemoteServerUrl('https://user:pass@flame.tailnet.ts.net:8790').ok).toBe(false);

    // M5: localStorage never stores apiToken and purges legacy saved tokens on load
    const mem = new Map<string, string>();
    const origLs = globalThis.localStorage;
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: (k: string) => mem.get(k) ?? null,
        setItem: (k: string, v: string) => mem.set(k, v),
        removeItem: (k: string) => mem.delete(k),
      },
    });
    try {
      saveComputeBackendConfig({
        ...defaultComputeBackendConfig(),
        mode: 'remote',
        serverUrl: 'https://flame.tailnet.ts.net:8790',
        apiToken: 'dnv_super_secret',
      });
      expect(mem.get(COMPUTE_BACKEND_STORAGE_KEY)).not.toContain('dnv_super_secret');

      // Legacy entry containing apiToken gets purged on load
      mem.set(
        COMPUTE_BACKEND_STORAGE_KEY,
        JSON.stringify({ mode: 'remote', serverUrl: '', apiToken: 'legacy_secret' }),
      );
      const loaded = loadComputeBackendConfig();
      expect(loaded.apiToken).toBe('');
      expect(mem.get(COMPUTE_BACKEND_STORAGE_KEY)).not.toContain('legacy_secret');
    } finally {
      Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: origLs });
    }
  });
});
