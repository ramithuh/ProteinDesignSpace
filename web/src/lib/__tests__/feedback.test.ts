import { beforeEach, describe, expect, it } from 'vitest';
import cfWorker from '$lib/og/cf-worker';
import {
  buildFeedbackKvKey,
  createMemoryFeedbackKv,
  MAX_FEEDBACK_BODY_BYTES,
  resetFeedbackRateLimiterForTests,
  sanitizeFeedbackSubmission,
  type FeedbackRecord,
} from '$lib/og/feedback-api';

describe('Cloudflare KV feedback endpoint & security invariants', () => {
  beforeEach(() => {
    resetFeedbackRateLimiterForTests();
  });

  it('orders KV keys newest-first lexicographically while keeping readable UTC timestamps', () => {
    const older = buildFeedbackKvKey(Date.UTC(2026, 10, 24, 1, 0, 0), 'aaaa1111');
    const newer = buildFeedbackKvKey(Date.UTC(2026, 10, 24, 2, 0, 0), 'bbbb2222');
    expect([older, newer].sort()[0]).toBe(newer);
    expect(newer).toContain('2026-11-24T02:00:00Z');
  });

  it('stores valid feedback in FEEDBACK_KV with rich list metadata and never persists raw IPs or ATOM lines', async () => {
    const kv = createMemoryFeedbackKv();
    const env = {
      ASSETS: { fetch: async () => new Response('nf', { status: 404 }) },
      FEEDBACK_KV: kv,
      FEEDBACK_ADMIN_TOKEN: 'test-only-dummy-token',
    };

    const postRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: 'https://denovo.run',
          'CF-Connecting-IP': '198.51.100.42',
        },
        body: JSON.stringify({
          category: 'idea',
          message: 'Would love covalent ligand support in Boltz-2!',
          contact: '@biodev',
          workflowUrl: 'https://denovo.run/?p=genie3-mpnn-af2',
          context: {
            blocks: 'PDB Input (pdb-input) -> LocalFold (localfold-af2)\nATOM      1  CA  ALA A   1       0.0 0.0 0.0',
            webgpu: 'Apple M3 Max (fp16)',
            computeMode: 'local',
            theme: 'dark',
          },
        }),
      }),
      env,
    );

    expect(postRes.status).toBe(201);
    const postJson = (await postRes.json()) as { ok: boolean; id: string };
    expect(postJson.ok).toBe(true);
    expect(kv.entries.size).toBe(1);

    const [[key, stored]] = [...kv.entries.entries()];
    expect(key.startsWith('fb:')).toBe(true);
    expect(stored.value).not.toContain('198.51.100.42');
    expect(stored.value).not.toContain('ATOM      1');

    const record = JSON.parse(stored.value) as FeedbackRecord;
    expect(record.category).toBe('idea');
    expect(record.message).toBe('Would love covalent ligand support in Boltz-2!');
    expect(record.contact).toBe('@biodev');
    expect(record.workflowUrl).toBe('https://denovo.run/?p=genie3-mpnn-af2');
    expect(record.context?.blocks).toBe('PDB Input (pdb-input) -> LocalFold (localfold-af2)');

    // Admin GET without token -> 401; with wrong token -> 401; with valid token -> 200 JSON & Markdown
    const unauth = await cfWorker.fetch(new Request('https://denovo.run/api/feedback'), env);
    expect(unauth.status).toBe(401);

    const wrongAuth = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        headers: { Authorization: 'Bearer wrong_token' },
      }),
      env,
    );
    expect(wrongAuth.status).toBe(401);

    const listRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback?full=1', {
        headers: { Authorization: 'Bearer test-only-dummy-token' },
      }),
      env,
    );
    expect(listRes.status).toBe(200);
    const listJson = (await listRes.json()) as { ok: boolean; count: number; items: FeedbackRecord[] };
    expect(listJson.count).toBe(1);
    expect(listJson.items[0].message).toContain('Boltz-2');

    const mdRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback?format=md', {
        headers: { Authorization: 'Bearer test-only-dummy-token' },
      }),
      env,
    );
    expect(mdRes.status).toBe(200);
    expect(await mdRes.text()).toContain('[IDEA]');
  });

  it('drops honeypot bot submissions with 0 KV writes and enforces per-IP rate limits with 0 extra KV writes', async () => {
    const kv = createMemoryFeedbackKv();
    const env = {
      ASSETS: { fetch: async () => new Response('nf', { status: 404 }) },
      FEEDBACK_KV: kv,
    };

    // Bot fills hidden `website` field -> 200 decoy response, 0 KV writes
    const botRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Origin: 'https://denovo.run' },
        body: JSON.stringify({
          category: 'general',
          message: 'Buy cheap peptides now!',
          website: 'https://spam.example',
        }),
      }),
      env,
    );
    expect(botRes.status).toBe(200);
    expect(kv.entries.size).toBe(0);

    // 5 legitimate submissions succeed; 6th from same IP returns 429 without extra KV writes
    for (let i = 0; i < 5; i++) {
      const res = await cfWorker.fetch(
        new Request('https://denovo.run/api/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Origin: 'https://denovo.run',
            'CF-Connecting-IP': '203.0.113.9',
          },
          body: JSON.stringify({ message: `Valid note #${i + 1}` }),
        }),
        env,
      );
      expect(res.status).toBe(201);
    }
    expect(kv.entries.size).toBe(5);

    const sixth = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: 'https://denovo.run',
          'CF-Connecting-IP': '203.0.113.9',
        },
        body: JSON.stringify({ message: 'Valid note #6' }),
      }),
      env,
    );
    expect(sixth.status).toBe(429);
    expect(kv.entries.size).toBe(5);
  });

  it('enforces same-origin, Content-Type, payload size caps, workflowUrl allowlist, and crisp 503 when KV is unbound', async () => {
    const unboundEnv = {
      ASSETS: { fetch: async () => new Response('nf', { status: 404 }) },
    };

    // GET disabled when FEEDBACK_ADMIN_TOKEN is unset
    const getDisabled = await cfWorker.fetch(new Request('https://denovo.run/api/feedback'), unboundEnv);
    expect(getDisabled.status).toBe(405);

    // Cross-origin rejected
    const crossOrigin = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Origin: 'https://evil.example' },
        body: JSON.stringify({ message: 'cross origin attempt' }),
      }),
      unboundEnv,
    );
    expect(crossOrigin.status).toBe(403);

    // Non-JSON rejected
    const formPost = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain', Origin: 'https://denovo.run' },
        body: 'message=hello',
      }),
      unboundEnv,
    );
    expect(formPost.status).toBe(415);

    // Oversized body rejected
    const huge = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Origin: 'https://denovo.run' },
        body: JSON.stringify({ message: 'x'.repeat(MAX_FEEDBACK_BODY_BYTES + 200) }),
      }),
      unboundEnv,
    );
    expect(huge.status).toBe(413);

    // Phishing workflowUrl rejected
    const badUrl = sanitizeFeedbackSubmission(
      { message: 'Check my link', workflowUrl: 'https://phish.example/?g=abc' },
      'https://denovo.run',
    );
    expect(badUrl.ok).toBe(false);

    // Valid request when FEEDBACK_KV is not yet bound -> crisp 503 KV_NOT_BOUND
    const unboundRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Origin: 'https://denovo.run' },
        body: JSON.stringify({ message: 'Great workbench!' }),
      }),
      unboundEnv,
    );
    expect(unboundRes.status).toBe(503);
    const unboundJson = (await unboundRes.json()) as { ok: boolean; code: string };
    expect(unboundJson.code).toBe('KV_NOT_BOUND');

    // Daily KV write limit error -> crisp 503 KV_WRITE_LIMIT (triggers UI GitHub fallback, never 500)
    const quotaEnv = {
      ASSETS: { fetch: async () => new Response('nf', { status: 404 }) },
      FEEDBACK_KV: {
        async put() {
          throw new Error('KV put() limit exceeded for the day.');
        },
        async get() {
          return null;
        },
        async list() {
          return { keys: [], list_complete: true };
        },
      },
    };
    const quotaRes = await cfWorker.fetch(
      new Request('https://denovo.run/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Origin: 'https://denovo.run' },
        body: JSON.stringify({ message: 'Testing daily quota graceful fallback' }),
      }),
      quotaEnv,
    );
    expect(quotaRes.status).toBe(503);
    const quotaJson = (await quotaRes.json()) as { ok: boolean; code: string };
    expect(quotaJson.code).toBe('KV_WRITE_LIMIT');
  });
});
