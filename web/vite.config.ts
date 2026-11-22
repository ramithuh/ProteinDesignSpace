import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import { resolve } from 'node:path';

/**
 * mpnn.worker.ts loads ONNX Runtime Web WASM binaries from jsDelivr CDN via `wasmPaths`.
 * Stripping Vite's auto-emitted `ort-wasm-*.wasm` assets (~24–26 MiB each) keeps every
 * output file well under Cloudflare Pages' 25 MiB single-file asset limit.
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
  plugins: [sveltekit(), stripUnusedOrtWasm()],
  resolve: {
    alias: {
      // proteinmpnn-web ships TS source in GitHub installs (prepack only runs on publish).
      // Alias straight to src/index.ts so Vite bundles it cleanly in workers & main thread.
      'proteinmpnn-web': resolve(__dirname, 'node_modules/proteinmpnn-web/src/index.ts'),
    },
  },
  // onnxruntime-web ships its own WASM artifacts and uses dynamic imports —
  // letting Vite optimize-deps it usually breaks the WASM resolution.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
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
