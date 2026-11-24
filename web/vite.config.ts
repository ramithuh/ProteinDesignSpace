import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import { resolve } from 'node:path';
import { createMemoryFeedbackKv, handleFeedbackRequest } from './src/lib/og/feedback-api';

/**
 * Local `vite dev` middleware for `/api/feedback` backed by an in-memory KV store
 * so the Feedback popover works end-to-end during local development without wrangler.
 */
function devFeedbackApiPlugin(): Plugin {
  const devKv = createMemoryFeedbackKv();
  return {
    name: 'dev-feedback-api',
    configureServer(server) {
      server.middlewares.use('/api/feedback', (req, res) => {
        void (async () => {
          const chunks: Buffer[] = [];
          for await (const chunk of req) {
            chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
          }
          const bodyBuf = Buffer.concat(chunks);
          const host = req.headers.host || 'localhost:5173';
          const rawProto = req.headers['x-forwarded-proto'];
          const proto =
            (Array.isArray(rawProto) ? rawProto[0] : rawProto)?.split(',')[0]?.trim() || 'http';
          const url = `${proto}://${host}${req.originalUrl || '/api/feedback'}`;
          const headers = new Headers();
          for (const [k, v] of Object.entries(req.headers)) {
            if (typeof v === 'string') headers.set(k, v);
            else if (Array.isArray(v)) headers.set(k, v.join(', '));
          }
          const method = req.method || 'GET';
          const webReq = new Request(url, {
            method,
            headers,
            ...(method !== 'GET' && method !== 'HEAD' ? { body: bodyBuf } : {}),
          });
          const webRes = await handleFeedbackRequest(webReq, {
            FEEDBACK_KV: devKv,
            FEEDBACK_ADMIN_TOKEN: process.env.FEEDBACK_ADMIN_TOKEN,
          });
          if (method === 'POST' && webRes.status === 201) {
            console.log(`[dev-feedback] Stored feedback (${devKv.entries.size} in dev memory KV)`);
          }
          res.statusCode = webRes.status;
          webRes.headers.forEach((val, key) => res.setHeader(key, val));
          res.end(Buffer.from(await webRes.arrayBuffer()));
        })().catch((err) => {
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify({ ok: false, error: String(err) }));
        });
      });
    },
  };
}

/**
 * mpnn.worker.ts loads first-party same-origin ONNX Runtime Web WASM binaries from
 * `/vendor/ort/` via `wasmPaths` (synced from npm-locked `node_modules/onnxruntime-web`).
 * Stripping Vite's auto-emitted duplicate `ort-wasm-*.wasm` assets keeps every
 * output file under Cloudflare Pages' 25 MiB single-file asset limit.
 */
function stripUnusedOrtWasm(): Plugin {
  return {
    name: 'strip-unused-ort-wasm',
    generateBundle(_options, bundle) {
      for (const fileName of Object.keys(bundle)) {
        if (/ort-wasm.*\.wasm$/i.test(fileName)) {
          delete bundle[fileName];
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [devFeedbackApiPlugin(), sveltekit(), stripUnusedOrtWasm()],
  resolve: {
    alias: [
      // proteinmpnn-web ships TS source in GitHub installs (prepack only runs on publish).
      // Alias straight to src/index.ts so Vite bundles it cleanly in workers & main thread.
      {
        find: /^proteinmpnn-web$/,
        replacement: resolve(__dirname, 'node_modules/proteinmpnn-web/src/index.ts'),
      },
      // In onnxruntime-web@1.24.3:
      // - `onnxruntime-web/webgpu` requires `ort-wasm-simd-threaded.asyncify.wasm` (25.93 MiB),
      //   exceeding Cloudflare Pages' 25 MiB per-file asset limit.
      // - The default `onnxruntime-web` JSEP build omits `maxStorageBuffersPerShaderStage` in
      //   its WebGPU `requestDevice()` call, causing silent Concat pipeline validation errors.
      // Route WebGPU sessions to the modern C++ WebGPU EP via JSPI (`16.11 MiB < 25 MiB`),
      // and WASM sessions (Safari/Firefox/CPU fallback) to the lean WASM runtime (`11.79 MiB`).
      {
        find: /^onnxruntime-web\/webgpu$/,
        replacement: resolve(__dirname, 'node_modules/onnxruntime-web/dist/ort.jspi.bundle.min.mjs'),
      },
      {
        find: /^onnxruntime-web$/,
        replacement: resolve(__dirname, 'node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs'),
      },
    ],
  },
  // onnxruntime-web ships its own WASM artifacts and uses dynamic imports —
  // letting Vite optimize-deps it usually breaks the WASM resolution.
  optimizeDeps: {
    exclude: ['onnxruntime-web', 'onnxruntime-web/webgpu', 'onnxruntime-web/jspi', 'onnxruntime-web/wasm'],
  },
  build: {
    rollupOptions: {
      output: {
        // Keep Svelte Flow in its own hashed immutable chunk while letting Vite naturally
        // code-split Mol*'s lazy dynamic imports.
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          if (id.includes('/@xyflow/')) return 'vendor-xyflow';
          return undefined;
        },
      },
    },
  },
  worker: {
    format: 'es',
    plugins: () => [stripUnusedOrtWasm()],
  },
  // Keep Vite's strict filesystem boundary enabled; allow optional explicit preview hosts via VITE_ALLOWED_HOSTS.
  server: {
    fs: {
      strict: true,
      allow: [
        resolve(__dirname),
        resolve(__dirname, 'static'),
        resolve(__dirname, 'node_modules'),
      ],
    },
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      ...(process.env.VITE_ALLOWED_HOSTS
        ? process.env.VITE_ALLOWED_HOSTS.split(',')
            .map((h) => h.trim())
            .filter(Boolean)
        : []),
    ],
  },
});
