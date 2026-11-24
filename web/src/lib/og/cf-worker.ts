import { decodeOgWorkflowFromQuery } from './og-graph';
import { fetchEdgeStructurePreview } from './edge-structure';
import { OG_HEIGHT, OG_WIDTH, renderDefaultOgPng, renderWorkflowOgPng } from './render-og-card';

export interface CfPagesEnv {
  ASSETS: { fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> };
}

function escapeHtmlAttr(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
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
    out = out.replace(/<title>[\s\S]*?<\/title>/i, `<title>${safeTitle}</title>`);
  } else {
    out = out.replace(/<\/head>/i, `  <title>${safeTitle}</title>\n</head>`);
  }
  const block = buildOgMetaBlock(opts);
  return out.replace(/<\/head>/i, `  ${block}\n</head>`);
}

/** Generate a 1200x630 PNG Response for `/api/og?g=...` or `/api/og?p=...`. */
export async function handleOgImageRequest(
  request: Request,
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  const url = new URL(request.url);
  const hasPayload = url.searchParams.has('g') || url.searchParams.has('p');
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

export default {
  async fetch(request: Request, env: CfPagesEnv, ctx?: { waitUntil(p: Promise<unknown>): void }): Promise<Response> {
    const url = new URL(request.url);

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

    // 2. Shared workflow landing page (`/?g=...` or `/?p=...`) -> inject dynamic OG meta tags
    if (
      (url.pathname === '/' || url.pathname === '/index.html') &&
      (url.searchParams.has('g') || url.searchParams.has('p'))
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
