/**
 * Zero-dependency Cloudflare Pages / Edge feedback handler backed by Cloudflare Workers KV (`FEEDBACK_KV`).
 *
 * Design invariants:
 * - Zero external packages (bundled directly by esbuild into `web/build/_worker.js`).
 * - Privacy-first: never persists raw client IPs, sequences, or uploaded PDB coordinates.
 * - Free-tier friendly: 1 KV write per valid submission; rate-limiting & honeypot checks run
 *   entirely in isolate memory (0 KV writes wasted on bots or bursts).
 * - Lexicographically newest-first KV keys (`fb:<reverseEpochMs>:<isoUtc>:<id>`) with rich KV
 *   metadata so Cloudflare Dashboard and `FEEDBACK_KV.list()` return recent items first.
 */

export type FeedbackCategory = 'idea' | 'bug' | 'general';

export interface FeedbackContext {
  blocks?: string;
  presetOrGraphId?: string;
  webgpu?: string;
  computeMode?: 'local' | 'remote';
  theme?: 'dark' | 'light';
  viewport?: string;
  userAgent?: string;
  errorSnippet?: string;
}

export interface FeedbackSubmissionPayload {
  category?: FeedbackCategory;
  message: string;
  contact?: string;
  workflowUrl?: string;
  context?: FeedbackContext;
  /** Hidden anti-bot honeypot field; legitimate users leave this empty. */
  website?: string;
}

export interface FeedbackRecord {
  id: string;
  createdAt: string;
  category: FeedbackCategory;
  message: string;
  contact?: string;
  workflowUrl?: string;
  context?: FeedbackContext;
  country?: string;
}

export interface FeedbackKvMetadata {
  id: string;
  createdAt: string;
  category: FeedbackCategory;
  preview: string;
  contact?: string;
  hasError?: boolean;
  hasWorkflowUrl?: boolean;
}

export interface CfKvListKey {
  name: string;
  expiration?: number;
  metadata?: unknown;
}

export interface CfKvNamespace {
  put(
    key: string,
    value: string,
    options?: { expirationTtl?: number; metadata?: Record<string, unknown> },
  ): Promise<void>;
  get(key: string, type?: 'text'): Promise<string | null>;
  list(options?: {
    prefix?: string;
    limit?: number;
    cursor?: string;
  }): Promise<{ keys: CfKvListKey[]; list_complete: boolean; cursor?: string }>;
}

export interface FeedbackEnv {
  FEEDBACK_KV?: CfKvNamespace;
  FEEDBACK_ADMIN_TOKEN?: string;
  FEEDBACK_WEBHOOK_URL?: string;
}

export const MAX_FEEDBACK_BODY_BYTES = 12 * 1024;
export const MAX_FEEDBACK_MESSAGE_CHARS = 2000;
const MAX_EPOCH_MS = 9_999_999_999_999;

const RATE_LIMIT_WINDOW_MS = 10 * 60 * 1000;
const RATE_LIMIT_MAX_PER_WINDOW = 5;
const RATE_LIMIT_MAX_TRACKED_CLIENTS = 2048;

let isolateSalt: string | null = null;

function getIsolateSalt(): string {
  if (isolateSalt !== null) return isolateSalt;
  const bytes = new Uint8Array(16);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes);
  }
  isolateSalt = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return isolateSalt;
}

const rateLimitBuckets = new Map<string, number[]>();

export function resetFeedbackRateLimiterForTests(): void {
  rateLimitBuckets.clear();
}

function jsonResponse(body: Record<string, unknown>, status: number, extraHeaders?: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
      ...extraHeaders,
    },
  });
}

function cleanSingleLine(val: unknown, maxLen: number): string {
  if (typeof val !== 'string') return '';
  return val
    .replace(/[\u0000-\u001F\u007F]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxLen);
}

function cleanMultiLine(val: unknown, maxLen: number): string {
  if (typeof val !== 'string') return '';
  return val
    .replace(/\r\n?/g, '\n')
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '')
    .trim()
    .slice(0, maxLen);
}

function generateShortId(): string {
  const alphabet = '23456789abcdefghjkmnpqrstuvwxyz';
  const bytes = new Uint8Array(8);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    out += alphabet[bytes[i] % alphabet.length];
  }
  return out;
}

/**
 * Build a KV key whose lexicographical ascending order is newest-first while remaining
 * human-readable in the Cloudflare KV dashboard (`fb:<revEpochMs>:<isoUtc>:<id>`).
 */
export function buildFeedbackKvKey(nowMs: number, id: string): string {
  const clamped = Math.max(0, Math.min(MAX_EPOCH_MS, Math.floor(nowMs)));
  const rev = String(MAX_EPOCH_MS - clamped).padStart(13, '0');
  const isoCompact = new Date(clamped).toISOString().replace(/\.\d{3}Z$/, 'Z');
  return `fb:${rev}:${isoCompact}:${id}`;
}

async function sha256Bytes(input: string): Promise<Uint8Array> {
  const encoded = new TextEncoder().encode(input);
  const buf = await crypto.subtle.digest('SHA-256', encoded);
  return new Uint8Array(buf);
}

async function constantTimeTokenEquals(a: string, b: string): Promise<boolean> {
  if (!a || !b) return false;
  const [da, db] = await Promise.all([sha256Bytes(a), sha256Bytes(b)]);
  if (da.length !== db.length) return false;
  let diff = a.length === b.length ? 0 : 1;
  for (let i = 0; i < da.length; i++) {
    diff |= da[i] ^ db[i];
  }
  return diff === 0;
}

async function checkClientRateLimit(clientIp: string, nowMs: number): Promise<boolean> {
  const digest = await sha256Bytes(`${getIsolateSalt()}:${clientIp}`);
  let bucketKey = '';
  for (let i = 0; i < 12; i++) {
    bucketKey += digest[i].toString(16).padStart(2, '0');
  }

  const cutoff = nowMs - RATE_LIMIT_WINDOW_MS;
  const prev = rateLimitBuckets.get(bucketKey);
  const recent = prev ? prev.filter((ts) => ts > cutoff) : [];

  if (recent.length >= RATE_LIMIT_MAX_PER_WINDOW) {
    rateLimitBuckets.set(bucketKey, recent);
    return false;
  }

  recent.push(nowMs);
  rateLimitBuckets.delete(bucketKey);
  rateLimitBuckets.set(bucketKey, recent);

  if (rateLimitBuckets.size > RATE_LIMIT_MAX_TRACKED_CLIENTS) {
    for (const [k, timestamps] of rateLimitBuckets) {
      const alive = timestamps.filter((ts) => ts > cutoff);
      if (alive.length === 0) {
        rateLimitBuckets.delete(k);
      } else {
        rateLimitBuckets.set(k, alive);
      }
      if (rateLimitBuckets.size <= RATE_LIMIT_MAX_TRACKED_CLIENTS) break;
    }
    while (rateLimitBuckets.size > RATE_LIMIT_MAX_TRACKED_CLIENTS) {
      const oldestKey = rateLimitBuckets.keys().next().value;
      if (oldestKey === undefined) break;
      rateLimitBuckets.delete(oldestKey);
    }
  }
  return true;
}

function isAllowedWorkflowShareUrl(rawUrl: string, requestOrigin: string): boolean {
  if (rawUrl.length > 4096) return false;
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return false;
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return false;
  const allowedOrigin = parsed.origin === requestOrigin || parsed.origin === 'https://denovo.run';
  if (!allowedOrigin) return false;
  const hasShareParam =
    parsed.searchParams.has('g') ||
    parsed.searchParams.has('p') ||
    /^#(?:g|p)=/.test(parsed.hash);
  return hasShareParam;
}

function stripCoordinateLines(text: string): string {
  return text
    .split(/\r?\n/)
    .filter((line) => !/^(?:ATOM  |HETATM)\s*\d+/.test(line))
    .join(' ');
}

export type SanitizeFeedbackResult =
  | { ok: true; honeypot: true }
  | {
      ok: true;
      honeypot: false;
      category: FeedbackCategory;
      message: string;
      contact?: string;
      workflowUrl?: string;
      context?: FeedbackContext;
    }
  | { ok: false; error: string };

export function sanitizeFeedbackSubmission(
  raw: unknown,
  requestOrigin: string,
): SanitizeFeedbackResult {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { ok: false, error: 'Request body must be a JSON object.' };
  }
  const obj = raw as Record<string, unknown>;

  if (typeof obj.website === 'string' && obj.website.trim().length > 0) {
    return { ok: true, honeypot: true };
  }

  let category: FeedbackCategory = 'general';
  if (obj.category !== undefined) {
    if (obj.category === 'idea' || obj.category === 'bug' || obj.category === 'general') {
      category = obj.category;
    } else {
      return { ok: false, error: 'Invalid feedback category.' };
    }
  }

  const message = cleanMultiLine(obj.message, MAX_FEEDBACK_MESSAGE_CHARS);
  if (message.length < 3) {
    return { ok: false, error: 'Please write at least a short note before sending.' };
  }

  const contact = cleanSingleLine(obj.contact, 120) || undefined;

  let workflowUrl: string | undefined;
  if (typeof obj.workflowUrl === 'string' && obj.workflowUrl.trim().length > 0) {
    const trimmedUrl = obj.workflowUrl.trim();
    if (!isAllowedWorkflowShareUrl(trimmedUrl, requestOrigin)) {
      return { ok: false, error: 'Invalid workflow share URL.' };
    }
    workflowUrl = trimmedUrl;
  }

  let context: FeedbackContext | undefined;
  if (obj.context && typeof obj.context === 'object' && !Array.isArray(obj.context)) {
    const rawCtx = obj.context as Record<string, unknown>;
    const nextCtx: FeedbackContext = {};
    const blocks = cleanSingleLine(
      typeof rawCtx.blocks === 'string' ? stripCoordinateLines(rawCtx.blocks) : '',
      400,
    );
    if (blocks) nextCtx.blocks = blocks;

    const presetOrGraphId = cleanSingleLine(rawCtx.presetOrGraphId, 120);
    if (presetOrGraphId) nextCtx.presetOrGraphId = presetOrGraphId;

    const webgpu = cleanSingleLine(rawCtx.webgpu, 160);
    if (webgpu) nextCtx.webgpu = webgpu;

    if (rawCtx.computeMode === 'local' || rawCtx.computeMode === 'remote') {
      nextCtx.computeMode = rawCtx.computeMode;
    }
    if (rawCtx.theme === 'dark' || rawCtx.theme === 'light') {
      nextCtx.theme = rawCtx.theme;
    }

    const viewport = cleanSingleLine(rawCtx.viewport, 32);
    if (viewport) nextCtx.viewport = viewport;

    const userAgent = cleanSingleLine(rawCtx.userAgent, 240);
    if (userAgent) nextCtx.userAgent = userAgent;

    const errorSnippet = cleanMultiLine(rawCtx.errorSnippet, 600);
    if (errorSnippet) nextCtx.errorSnippet = errorSnippet;

    if (Object.keys(nextCtx).length > 0) {
      context = nextCtx;
    }
  }

  return {
    ok: true,
    honeypot: false,
    category,
    message,
    ...(contact ? { contact } : {}),
    ...(workflowUrl ? { workflowUrl } : {}),
    ...(context ? { context } : {}),
  };
}

async function notifyOptionalWebhook(webhookUrl: string, record: FeedbackRecord): Promise<void> {
  try {
    const badge =
      record.category === 'bug' ? '[BUG]' : record.category === 'idea' ? '[IDEA]' : '[NOTE]';
    const lines = [
      `**${badge} denovo.run feedback** (${record.createdAt})`,
      record.message,
      ...(record.contact ? [`Contact: \`${record.contact}\``] : []),
      ...(record.context?.blocks ? [`Blocks: \`${record.context.blocks}\``] : []),
      ...(record.context?.webgpu ? [`WebGPU: \`${record.context.webgpu}\``] : []),
      ...(record.context?.errorSnippet ? [`Error: \`${record.context.errorSnippet.slice(0, 220)}\``] : []),
      ...(record.workflowUrl ? [`Workflow: ${record.workflowUrl}`] : []),
    ];
    const text = lines.join('\n').slice(0, 1950);
    await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text, text }),
    });
  } catch {
    /* Optional secondary notification — never fail KV submission */
  }
}

function renderFeedbackMarkdown(items: Array<FeedbackRecord | FeedbackKvMetadata>): string {
  if (items.length === 0) return '# denovo.run Feedback\n\nNo feedback entries yet.\n';
  const out = [`# denovo.run Feedback (${items.length})`, ''];
  for (const item of items) {
    const full = 'message' in item ? item : null;
    const tag = item.category.toUpperCase();
    out.push(`## [${tag}] ${item.createdAt} (\`${item.id}\`)`);
    if (item.contact) out.push(`- **Contact:** ${item.contact}`);
    if (full?.context?.blocks) out.push(`- **Blocks:** ${full.context.blocks}`);
    if (full?.context?.webgpu) out.push(`- **WebGPU:** ${full.context.webgpu}`);
    if (full?.workflowUrl) out.push(`- **Workflow:** ${full.workflowUrl}`);
    if (full?.context?.errorSnippet) {
      out.push(`- **Error:** \`${full.context.errorSnippet.replace(/\n+/g, ' ')}\``);
    }
    out.push('', 'message' in item ? item.message : item.preview, '');
  }
  return out.join('\n');
}

async function handleAdminListRequest(request: Request, env: FeedbackEnv): Promise<Response> {
  const adminToken = (env.FEEDBACK_ADMIN_TOKEN ?? '').trim();
  if (!adminToken) {
    return jsonResponse(
      { ok: false, error: 'Method not allowed.' },
      405,
      { Allow: 'POST' },
    );
  }

  const authHeader = request.headers.get('authorization') ?? '';
  const match = /^Bearer\s+(.+)$/i.exec(authHeader.trim());
  const provided = match?.[1]?.trim() ?? '';
  if (!(await constantTimeTokenEquals(provided, adminToken))) {
    return jsonResponse({ ok: false, error: 'Unauthorized.' }, 401);
  }

  if (!env.FEEDBACK_KV) {
    return jsonResponse(
      { ok: false, code: 'KV_NOT_BOUND', error: 'FEEDBACK_KV namespace is not bound.' },
      503,
    );
  }

  const url = new URL(request.url);
  const rawLimit = Number(url.searchParams.get('limit') ?? '50');
  const limit = Number.isFinite(rawLimit) ? Math.max(1, Math.min(100, Math.round(rawLimit))) : 50;
  const cursor = url.searchParams.get('cursor') || undefined;
  const includeFull = url.searchParams.get('full') === '1' || url.searchParams.get('format') === 'md';

  const listed = await env.FEEDBACK_KV.list({ prefix: 'fb:', limit, cursor });

  if (!includeFull) {
    const summaries = listed.keys.map((k) => ({
      key: k.name,
      ...(k.metadata && typeof k.metadata === 'object' ? (k.metadata as Record<string, unknown>) : {}),
    }));
    return jsonResponse({
      ok: true,
      count: summaries.length,
      listComplete: listed.list_complete,
      cursor: listed.cursor ?? null,
      items: summaries,
    }, 200);
  }

  const cappedKeys = listed.keys.slice(0, Math.min(limit, 50));
  const records = await Promise.all(
    cappedKeys.map(async (k) => {
      const raw = await env.FEEDBACK_KV!.get(k.name, 'text');
      if (!raw) return null;
      try {
        return JSON.parse(raw) as FeedbackRecord;
      } catch {
        return null;
      }
    }),
  );
  const validRecords = records.filter((r): r is FeedbackRecord => r !== null);

  if (url.searchParams.get('format') === 'md') {
    return new Response(renderFeedbackMarkdown(validRecords), {
      status: 200,
      headers: {
        'Content-Type': 'text/markdown; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      },
    });
  }

  return jsonResponse({
    ok: true,
    count: validRecords.length,
    listComplete: listed.list_complete,
    cursor: listed.cursor ?? null,
    items: validRecords,
  }, 200);
}

export async function handleFeedbackRequest(
  request: Request,
  env: FeedbackEnv,
  ctx?: { waitUntil(p: Promise<unknown>): void },
): Promise<Response> {
  if (request.method === 'GET') {
    return handleAdminListRequest(request, env);
  }

  if (request.method !== 'POST') {
    const allow = (env.FEEDBACK_ADMIN_TOKEN ?? '').trim() ? 'GET, POST' : 'POST';
    return jsonResponse({ ok: false, error: 'Method not allowed.' }, 405, { Allow: allow });
  }

  const url = new URL(request.url);
  const secFetchSite = (request.headers.get('sec-fetch-site') ?? '').toLowerCase();
  if (secFetchSite === 'cross-site') {
    return jsonResponse({ ok: false, error: 'Cross-site requests are not permitted.' }, 403);
  }

  const originHeader = request.headers.get('origin');
  if (originHeader && originHeader !== url.origin) {
    return jsonResponse({ ok: false, error: 'Origin mismatch.' }, 403);
  }

  const contentType = (request.headers.get('content-type') ?? '').toLowerCase();
  if (!contentType.includes('application/json')) {
    return jsonResponse({ ok: false, error: 'Expected Content-Type: application/json.' }, 415);
  }

  const declaredLen = Number(request.headers.get('content-length') ?? '0');
  if (Number.isFinite(declaredLen) && declaredLen > MAX_FEEDBACK_BODY_BYTES) {
    return jsonResponse({ ok: false, error: 'Feedback payload is too large.' }, 413);
  }

  let rawBytes: Uint8Array;
  try {
    rawBytes = new Uint8Array(await request.arrayBuffer());
  } catch {
    return jsonResponse({ ok: false, error: 'Could not read request body.' }, 400);
  }
  if (rawBytes.byteLength > MAX_FEEDBACK_BODY_BYTES) {
    return jsonResponse({ ok: false, error: 'Feedback payload is too large.' }, 413);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(rawBytes));
  } catch {
    return jsonResponse({ ok: false, error: 'Invalid JSON payload.' }, 400);
  }

  const sanitized = sanitizeFeedbackSubmission(parsed, url.origin);
  if (!sanitized.ok) {
    return jsonResponse({ ok: false, error: sanitized.error }, 400);
  }

  // Silently drop bot honeypot hits with 0 KV writes
  if (sanitized.honeypot) {
    return jsonResponse({ ok: true, id: 'ok' }, 200);
  }

  const nowMs = Date.now();
  const cfIp = (request.headers.get('cf-connecting-ip') ?? '').trim();
  if (cfIp) {
    const allowed = await checkClientRateLimit(cfIp, nowMs);
    if (!allowed) {
      return jsonResponse(
        { ok: false, error: 'Too many submissions in a short window — please try again in a few minutes.' },
        429,
        { 'Retry-After': '60' },
      );
    }
  }

  if (!env.FEEDBACK_KV) {
    return jsonResponse(
      {
        ok: false,
        code: 'KV_NOT_BOUND',
        error: 'Feedback KV store is not configured on this host.',
      },
      503,
    );
  }

  const id = generateShortId();
  const createdAt = new Date(nowMs).toISOString();
  const cfCountryRaw = (request as unknown as { cf?: { country?: unknown } }).cf?.country;
  const country =
    typeof cfCountryRaw === 'string' && /^[A-Z]{2}$/.test(cfCountryRaw) ? cfCountryRaw : undefined;

  const record: FeedbackRecord = {
    id,
    createdAt,
    category: sanitized.category,
    message: sanitized.message,
    ...(sanitized.contact ? { contact: sanitized.contact } : {}),
    ...(sanitized.workflowUrl ? { workflowUrl: sanitized.workflowUrl } : {}),
    ...(sanitized.context ? { context: sanitized.context } : {}),
    ...(country ? { country } : {}),
  };

  const metadata: FeedbackKvMetadata = {
    id,
    createdAt,
    category: record.category,
    preview: cleanSingleLine(record.message, 140),
    ...(record.contact ? { contact: record.contact.slice(0, 60) } : {}),
    ...(record.context?.errorSnippet ? { hasError: true } : {}),
    ...(record.workflowUrl ? { hasWorkflowUrl: true } : {}),
  };

  const kvKey = buildFeedbackKvKey(nowMs, id);
  try {
    await env.FEEDBACK_KV.put(kvKey, JSON.stringify(record), {
      metadata: metadata as unknown as Record<string, unknown>,
    });
  } catch {
    return jsonResponse(
      {
        ok: false,
        code: 'KV_WRITE_LIMIT',
        error: 'Daily feedback inbox quota reached — please send via GitHub Issues instead.',
      },
      503,
    );
  }

  const webhookUrl = (env.FEEDBACK_WEBHOOK_URL ?? '').trim();
  if (webhookUrl) {
    const hookPromise = notifyOptionalWebhook(webhookUrl, record);
    if (ctx?.waitUntil) ctx.waitUntil(hookPromise);
    else await hookPromise;
  }

  return jsonResponse({ ok: true, id }, 201);
}

/** Lightweight in-memory `CfKvNamespace` implementation for tests and local `vite dev`. */
export function createMemoryFeedbackKv(): CfKvNamespace & {
  entries: Map<string, { value: string; metadata?: Record<string, unknown> }>;
} {
  const entries = new Map<string, { value: string; metadata?: Record<string, unknown> }>();
  return {
    entries,
    async put(key, value, options) {
      entries.set(key, { value, metadata: options?.metadata });
    },
    async get(key) {
      return entries.get(key)?.value ?? null;
    },
    async list(options) {
      const prefix = options?.prefix ?? '';
      const limit = options?.limit ?? 1000;
      const startIdx = options?.cursor ? Number(options.cursor) || 0 : 0;
      const matching = [...entries.keys()]
        .filter((k) => k.startsWith(prefix))
        .sort();
      const slice = matching.slice(startIdx, startIdx + limit);
      const nextIdx = startIdx + slice.length;
      const list_complete = nextIdx >= matching.length;
      return {
        keys: slice.map((name) => ({
          name,
          metadata: entries.get(name)?.metadata,
        })),
        list_complete,
        ...(list_complete ? {} : { cursor: String(nextIdx) }),
      };
    },
  };
}
