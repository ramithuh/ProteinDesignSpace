import { decodeOgWorkflowFromQuery } from './og-graph';
import { fetchEdgeStructurePreview } from './edge-structure';
import { handleFeedbackRequest, type CfKvNamespace } from './feedback-api';
import { OG_HEIGHT, OG_WIDTH, renderDefaultOgPng, renderWorkflowOgPng } from './render-og-card';
import {
  MAX_ERROR_MESSAGE_CHARS,
  MAX_ERROR_TOKEN_CHARS,
  MAX_HASH_CHARS,
  MAX_INFLATED_BYTES,
  decodeWorkflowQuery,
  encodeCompactShareQuery,
  encodeGraphToFlow,
  parseFlowDsl,
  parseRecipeSafe,
  truncateErrorField,
  type FlowParseResult,
  type FlowValidationError,
  type LinkEdgeSummary,
  type LinkNodeSummary,
} from '../engine/flow-dsl';

export interface CfPagesEnv {
  ASSETS: { fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> };
  FEEDBACK_KV?: CfKvNamespace;
  FEEDBACK_ADMIN_TOKEN?: string;
  FEEDBACK_WEBHOOK_URL?: string;
}

export interface LinkApiResponse {
  ok: boolean;
  url: string | null;
  shortUrl: string | null;
  nodes: LinkNodeSummary[];
  edges: LinkEdgeSummary[];
  errors: FlowValidationError[];
}

function escapeHtmlAttr(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function hasWorkflowQuery(searchParams: URLSearchParams): boolean {
  return (
    searchParams.has('flow') ||
    searchParams.has('recipe') ||
    searchParams.has('g') ||
    searchParams.has('p')
  );
}

function buildOgMetaBlock(opts: {
  title: string;
  description: string;
  imageUrl: string;
  pageUrl: string;
}): string {
  const t = escapeHtmlAttr(opts.title);
  const d = escapeHtmlAttr(opts.description);
  const img = escapeHtmlAttr(opts.imageUrl);
  const url = escapeHtmlAttr(opts.pageUrl);
  return [
    `<meta property="og:title" content="${t}" />`,
    `<meta property="og:description" content="${d}" />`,
    `<meta property="og:image" content="${img}" />`,
    `<meta property="og:image:type" content="image/png" />`,
    `<meta property="og:image:width" content="${OG_WIDTH}" />`,
    `<meta property="og:image:height" content="${OG_HEIGHT}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta property="og:type" content="website" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${t}" />`,
    `<meta name="twitter:description" content="${d}" />`,
    `<meta name="twitter:image" content="${img}" />`,
  ].join('\n  ');
}

/** Rewrite default OG tags in index.html with workflow-specific title, description & og:image. */
export function injectOgMetaIntoHtml(
  html: string,
  opts: { title: string; description: string; imageUrl: string; pageUrl: string },
): string {
  // Strip static default og:/twitter: tags so scrapers see only the workflow-specific card
  let out = html.replace(
    /[ \t]*<meta\s+(?:property="og:[^"]*"|name="twitter:[^"]*")[^>]*\/?>\r?\n?/gi,
    '',
  );
  const safeTitle = escapeHtmlAttr(opts.title);
  if (/<title>[\s\S]*?<\/title>/i.test(out)) {
    out = out.replace(/<title>[\s\S]*?<\/title>/i, () => `<title>${safeTitle}</title>`);
  } else {
    out = out.replace(/<\/head>/i, () => `  <title>${safeTitle}</title>\n</head>`);
  }
  const block = buildOgMetaBlock(opts);
  return out.replace(/<\/head>/i, () => `  ${block}\n</head>`);
}

/** Generate a 1200x630 PNG Response for `/api/og?flow=...`, `?recipe=...`, `?g=...`, or `?p=...`. */
export async function handleOgImageRequest(
  request: Request,
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  const url = new URL(request.url);
  const hasPayload = hasWorkflowQuery(url.searchParams);
  let pngBytes: Uint8Array;
  try {
    if (hasPayload) {
      const workflow = await decodeOgWorkflowFromQuery(url.search);
      if (workflow) {
        const structure = workflow.structureTarget
          ? await fetchEdgeStructurePreview(workflow.structureTarget, fetchImpl)
          : null;
        pngBytes = await renderWorkflowOgPng(workflow, structure);
      } else {
        pngBytes = await renderDefaultOgPng();
      }
    } else {
      pngBytes = await renderDefaultOgPng();
    }
  } catch {
    pngBytes = await renderDefaultOgPng();
  }

  return new Response(new Uint8Array(pngBytes), {
    status: 200,
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=86400, s-maxage=604800, stale-while-revalidate=86400',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}

const LINK_CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
};

function linkJsonResponse(
  body: LinkApiResponse,
  status: number,
  extraHeaders?: Record<string, string>,
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...LINK_CORS_HEADERS,
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
      ...(extraHeaders ?? {}),
    },
  });
}

function linkErrorResponse(
  status: number,
  errors: FlowValidationError[],
  extraHeaders?: Record<string, string>,
): Response {
  const safeErrors: FlowValidationError[] = errors.map((e) => ({
    ...e,
    message: truncateErrorField(e.message, MAX_ERROR_MESSAGE_CHARS),
    ...(e.token !== undefined ? { token: truncateErrorField(e.token, MAX_ERROR_TOKEN_CHARS) } : {}),
    ...(e.nodeId !== undefined ? { nodeId: truncateErrorField(e.nodeId, MAX_ERROR_TOKEN_CHARS) } : {}),
  }));
  return linkJsonResponse(
    {
      ok: false,
      url: null,
      shortUrl: null,
      nodes: [],
      edges: [],
      errors: safeErrors,
    },
    status,
    extraHeaders,
  );
}

async function decodeLinkSearchParams(
  sp: URLSearchParams,
  rawSearch: string,
): Promise<FlowParseResult | null> {
  if (sp.has('flow')) {
    const rawFlow = sp.get('flow') ?? '';
    if (rawFlow.length > MAX_HASH_CHARS) {
      return {
        ok: false,
        graph: null,
        nodes: [],
        edges: [],
        errors: [
          {
            code: 'HASH_TOO_LARGE',
            message: `Shared workflow link is too large (max ${MAX_HASH_CHARS} chars).`,
          },
        ],
      };
    }
    return parseFlowDsl(rawFlow);
  }
  if (sp.has('recipe')) {
    return parseRecipeSafe(sp.get('recipe') ?? '');
  }
  if (sp.has('g') || sp.has('p')) {
    return decodeWorkflowQuery(rawSearch);
  }
  if (sp.has('url')) {
    return decodeWorkflowQuery(sp.get('url') ?? '');
  }
  return null;
}

function stripQuotedStringsForHeuristic(input: string): string {
  let out = '';
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quote === null) {
      if (ch === '"') {
        quote = '"';
        out += '"';
      } else if (ch === "'" && (i === 0 || /[=:,;(\s]/.test(input[i - 1]))) {
        quote = "'";
        out += "'";
      } else {
        out += ch;
      }
    } else if (ch === '\\' && i + 1 < input.length) {
      i++;
    } else if (ch === quote) {
      out += quote;
      quote = null;
    }
  }
  return out;
}

/**
 * Stateless `GET` / `POST` / `OPTIONS /api/link` endpoint validating workflow inputs
 * (`?flow=`, `?recipe=`, `?g=`, `?p=`, `?url=`, JSON `{ flow?, recipe?, url?, graph? }`,
 * form-urlencoded, or plain text / `denovo: v1` YAML) and returning canonical `?flow=`
 * and compact `?g=`/`?p=` share URLs.
 */
export async function handleLinkRequest(request: Request): Promise<Response> {
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        ...LINK_CORS_HEADERS,
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      },
    });
  }

  if (request.method !== 'GET' && request.method !== 'POST') {
    return linkErrorResponse(
      405,
      [{ code: 'METHOD_NOT_ALLOWED', message: 'Method not allowed. Use GET or POST.' }],
      { Allow: 'GET, POST, OPTIONS' },
    );
  }

  try {
    const url = new URL(request.url);
    let result: FlowParseResult | null = null;

    if (request.method === 'GET') {
      result = await decodeLinkSearchParams(url.searchParams, url.search);
      if (!result) {
        return linkErrorResponse(400, [
          {
            code: 'MISSING_INPUT',
            message: 'Provide flow, recipe, g, p, or url query parameter.',
          },
        ]);
      }
    } else {
      const contentLengthHeader = request.headers.get('content-length');
      if (contentLengthHeader !== null) {
        const contentLength = Number(contentLengthHeader);
        if (Number.isFinite(contentLength) && contentLength > MAX_INFLATED_BYTES) {
          return linkErrorResponse(413, [
            {
              code: 'PAYLOAD_TOO_LARGE',
              message: `Request body exceeds maximum size of ${MAX_INFLATED_BYTES} bytes.`,
            },
          ]);
        }
      }

      const rawCt = request.headers.get('content-type') ?? '';
      const mediaType = rawCt.split(';')[0].trim().toLowerCase();
      const isJsonCt =
        mediaType === 'application/json' ||
        mediaType.endsWith('+json') ||
        mediaType === 'text/json';
      const isFormCt = mediaType === 'application/x-www-form-urlencoded';
      const isTextCt =
        mediaType === '' ||
        mediaType === 'text/plain' ||
        mediaType === 'text/yaml' ||
        mediaType === 'application/yaml' ||
        mediaType === 'application/x-yaml';

      if (!isJsonCt && !isFormCt && !isTextCt) {
        return linkErrorResponse(415, [
          {
            code: 'UNSUPPORTED_MEDIA_TYPE',
            message: `Unsupported Content-Type "${mediaType}". Use application/json, application/x-www-form-urlencoded, text/plain, or text/yaml.`,
          },
        ]);
      }

      const buf = await request.arrayBuffer();
      if (buf.byteLength > MAX_INFLATED_BYTES) {
        return linkErrorResponse(413, [
          {
            code: 'PAYLOAD_TOO_LARGE',
            message: `Request body exceeds maximum size of ${MAX_INFLATED_BYTES} bytes.`,
          },
        ]);
      }

      const rawText = new TextDecoder().decode(buf);
      const trimmed = rawText.trim();
      if (!trimmed) {
        return linkErrorResponse(400, [
          { code: 'EMPTY_BODY', message: 'Request body is empty.' },
        ]);
      }

      if (isJsonCt || ((mediaType === '' || mediaType === 'text/plain') && trimmed.startsWith('{'))) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(trimmed);
        } catch {
          return linkErrorResponse(400, [
            { code: 'INVALID_JSON', message: 'Request body is not valid JSON.' },
          ]);
        }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          return linkErrorResponse(400, [
            {
              code: 'INVALID_INPUT',
              message: 'JSON body must be an object with flow, recipe, url, or graph.',
            },
          ]);
        }
        const body = parsed as Record<string, unknown>;
        if ('flow' in body && body.flow !== undefined) {
          if (typeof body.flow !== 'string') {
            return linkErrorResponse(400, [
              { code: 'INVALID_INPUT', message: 'Field "flow" must be a string.' },
            ]);
          }
          if (body.flow.length > MAX_HASH_CHARS) {
            return linkErrorResponse(400, [
              {
                code: 'HASH_TOO_LARGE',
                message: `Shared workflow link is too large (max ${MAX_HASH_CHARS} chars).`,
              },
            ]);
          }
          result = parseFlowDsl(body.flow);
        } else if ('recipe' in body && body.recipe !== undefined) {
          if (typeof body.recipe === 'string') {
            result = parseRecipeSafe(body.recipe);
          } else if (body.recipe && typeof body.recipe === 'object' && !Array.isArray(body.recipe)) {
            result = parseRecipeSafe(JSON.stringify(body.recipe));
          } else {
            return linkErrorResponse(400, [
              { code: 'INVALID_INPUT', message: 'Field "recipe" must be a string or workflow object.' },
            ]);
          }
        } else if ('url' in body && body.url !== undefined) {
          if (typeof body.url !== 'string') {
            return linkErrorResponse(400, [
              { code: 'INVALID_INPUT', message: 'Field "url" must be a string.' },
            ]);
          }
          result = await decodeWorkflowQuery(body.url);
        } else if ('graph' in body && body.graph !== undefined) {
          if (
            !body.graph ||
            typeof body.graph !== 'object' ||
            Array.isArray(body.graph) ||
            !Array.isArray((body.graph as Record<string, unknown>).nodes) ||
            !Array.isArray((body.graph as Record<string, unknown>).edges)
          ) {
            return linkErrorResponse(400, [
              {
                code: 'INVALID_INPUT',
                message: 'Field "graph" must be a WorkflowGraph object with nodes[] and edges[].',
              },
            ]);
          }
          result = parseRecipeSafe(JSON.stringify(body.graph));
        } else if (Array.isArray(body.nodes) && Array.isArray(body.edges)) {
          result = parseRecipeSafe(trimmed);
        } else {
          return linkErrorResponse(400, [
            {
              code: 'MISSING_INPUT',
              message: 'JSON body must include flow, recipe, url, or graph.',
            },
          ]);
        }
      } else if (isFormCt) {
        const formParams = new URLSearchParams(rawText);
        result = await decodeLinkSearchParams(formParams, `?${rawText}`);
        if (!result) {
          return linkErrorResponse(400, [
            {
              code: 'MISSING_INPUT',
              message: 'Form body must include flow, recipe, g, p, or url parameter.',
            },
          ]);
        }
      } else {
        const isExplicitYamlCt =
          mediaType === 'text/yaml' ||
          mediaType === 'application/yaml' ||
          mediaType === 'application/x-yaml';
        const unquoted = stripQuotedStringsForHeuristic(trimmed);
        const isYamlRecipe =
          isExplicitYamlCt ||
          /^[ \t]*denovo[ \t]*:[ \t]*(?:['"]?v\d+(?:\.\d+)*['"]?[ \t]*(?:#.*)?$|$)/im.test(unquoted) ||
          /^[ \t]*nodes[ \t]*:[ \t]*(?:#.*)?$/m.test(unquoted);
        const isQueryString =
          /^https?:\/\//i.test(trimmed) ||
          /^\/?(?:[?#]|(?:index\.html)?[?#])/i.test(trimmed) ||
          /^(?:flow|recipe|g|p)=/i.test(trimmed) ||
          (!unquoted.includes('\n') &&
            /^[A-Za-z0-9_.-]+=/.test(unquoted) &&
            /(?:^|&)(?:flow|recipe|g|p)=/i.test(unquoted));

        if (isYamlRecipe) {
          result = parseRecipeSafe(trimmed);
        } else if (isQueryString) {
          result = await decodeWorkflowQuery(trimmed);
        } else {
          if (trimmed.length > MAX_HASH_CHARS) {
            return linkErrorResponse(400, [
              {
                code: 'HASH_TOO_LARGE',
                message: `Shared workflow link is too large (max ${MAX_HASH_CHARS} chars).`,
              },
            ]);
          }
          result = parseFlowDsl(trimmed);
        }
      }
    }

    if (!result.ok || !result.graph || result.graph.nodes.length === 0) {
      const errors: FlowValidationError[] =
        result.errors.length > 0
          ? result.errors
          : [{ code: 'EMPTY_FLOW', message: 'Workflow contains no nodes.' }];
      return linkErrorResponse(400, errors);
    }

    const flowEnc = encodeGraphToFlow(result.graph);
    if (flowEnc.strippedPdb) {
      return linkErrorResponse(400, [
        {
          code: 'INLINE_PDB_TOO_LARGE',
          message:
            'Inline pdbText exceeds stateless URL limit (1500 chars); use pdbId, uniprotId, or a smaller structure.',
        },
      ]);
    }
    if (flowEnc.query.length > MAX_HASH_CHARS) {
      return linkErrorResponse(400, [
        {
          code: 'HASH_TOO_LARGE',
          message: `Encoded workflow URL exceeds maximum shareable length (${MAX_HASH_CHARS} chars).`,
        },
      ]);
    }
    let compactEnc = await encodeCompactShareQuery(result.graph);
    if (compactEnc.query.length > MAX_HASH_CHARS && compactEnc.mode === 'preset-diff') {
      compactEnc = await encodeCompactShareQuery(result.graph, { forceCompressedDag: true });
    }
    if (compactEnc.query.length > MAX_HASH_CHARS) {
      return linkErrorResponse(400, [
        {
          code: 'HASH_TOO_LARGE',
          message: `Encoded short workflow URL exceeds maximum shareable length (${MAX_HASH_CHARS} chars).`,
        },
      ]);
    }
    return linkJsonResponse(
      {
        ok: true,
        url: `${url.origin}/?${flowEnc.query}`,
        shortUrl: `${url.origin}/?${compactEnc.query}`,
        nodes: result.nodes,
        edges: result.edges,
        errors: [],
      },
      200,
    );
  } catch (err) {
    return linkErrorResponse(400, [
      {
        code: 'INVALID_INPUT',
        message: err instanceof Error ? err.message : 'Invalid workflow input.',
      },
    ]);
  }
}

/** Return true for automated vulnerability scanner probes that should receive 404 instead of SPA 200 HTML. */
export function isScannerProbePath(pathname: string): boolean {
  const p = pathname.toLowerCase();
  if (
    p.startsWith('/.env') ||
    p.startsWith('/.git') ||
    p.startsWith('/.aws') ||
    p.startsWith('/.ssh') ||
    p.startsWith('/.ds_store') ||
    p.startsWith('/wp-') ||
    p.startsWith('/sites/default/') ||
    p === '/config.js' ||
    p === '/config.json' ||
    p === '/xmlrpc.php' ||
    p === '/phpinfo.php'
  ) {
    return true;
  }
  return /\.(?:php|bak|old|save|swp|sql|ini|log)(?:\.|$)/.test(p);
}

export default {
  async fetch(request: Request, env: CfPagesEnv, ctx?: { waitUntil(p: Promise<unknown>): void }): Promise<Response> {
    const url = new URL(request.url);

    // 0. Drop automated dotfile/CMS/PHP scanner probes with a real 404 (not SPA 200 index.html)
    if (isScannerProbePath(url.pathname)) {
      return new Response('Not found', {
        status: 404,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'public, max-age=3600',
          'X-Content-Type-Options': 'nosniff',
        },
      });
    }

    // 1. Dynamic 1200x630 PNG OG preview endpoint
    if (url.pathname === '/api/og' || url.pathname === '/og.png') {
      const cfCaches = (globalThis as unknown as { caches?: { default?: Cache } }).caches?.default;
      const cacheKey = new Request(url.toString(), { method: 'GET' });
      if (cfCaches && request.method === 'GET') {
        const cached = await cfCaches.match(cacheKey);
        if (cached) return cached;
      }
      const res = await handleOgImageRequest(request);
      if (cfCaches && request.method === 'GET' && res.status === 200) {
        const putP = cfCaches.put(cacheKey, res.clone());
        if (ctx?.waitUntil) ctx.waitUntil(putP);
        else await putP.catch(() => {});
      }
      return res;
    }

    // 1b. Same-origin Cloudflare KV feedback endpoint (`POST /api/feedback`, optional admin `GET`)
    if (url.pathname === '/api/feedback') {
      return handleFeedbackRequest(request, env, ctx);
    }

    // 1c. Stateless LLM/share URL validation & canonicalizer endpoint (`GET` / `POST` / `OPTIONS /api/link`)
    if (url.pathname === '/api/link') {
      return handleLinkRequest(request);
    }

    // 1d. Unknown /api/* endpoints on public static Cloudflare Pages (e.g. /api/health) -> crisp 404 JSON
    if (url.pathname.startsWith('/api/')) {
      return new Response(
        JSON.stringify({ ok: false, error: 'No self-hosted GPU server at this URL' }),
        {
          status: 404,
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Cache-Control': 'public, max-age=300',
            'X-Content-Type-Options': 'nosniff',
          },
        },
      );
    }

    // 2. Shared workflow landing page (`/?flow=...`, `/?recipe=...`, `/?g=...`, or `/?p=...`) -> inject dynamic OG meta tags
    if (
      (url.pathname === '/' || url.pathname === '/index.html') &&
      hasWorkflowQuery(url.searchParams)
    ) {
      // IMPORTANT: Fetch `/` (not `/index.html`) from Cloudflare Pages ASSETS —
      // CF Pages ASSETS returns a 308 redirect from `/index.html` -> `/`!
      const assetRes = await env.ASSETS.fetch(new Request(new URL('/', url.origin), request));
      const contentType = assetRes.headers.get('content-type') ?? '';
      if (!assetRes.ok || !contentType.includes('text/html')) {
        return assetRes;
      }
      try {
        const workflow = await decodeOgWorkflowFromQuery(url.search);
        if (workflow) {
          const html = await assetRes.text();
          const ogImageUrl = `${url.origin}/api/og${url.search}`;
          const rewritten = injectOgMetaIntoHtml(html, {
            title: workflow.title,
            description: workflow.description,
            imageUrl: ogImageUrl,
            pageUrl: url.toString(),
          });
          const headers = new Headers(assetRes.headers);
          headers.set('Content-Type', 'text/html; charset=utf-8');
          headers.set('Cache-Control', 'public, max-age=0, must-revalidate');
          return new Response(rewritten, { status: assetRes.status, headers });
        }
      } catch {
        /* fall through to unmodified index.html */
      }
      return assetRes;
    }

    return env.ASSETS.fetch(request);
  },
};
