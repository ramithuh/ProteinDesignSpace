import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [sveltekit()],
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
  worker: {
    format: 'es',
  },
  // Allow serving large model files via symlink out of /static + Tailscale HTTPS preview.
  server: {
    fs: { strict: false },
    allowedHosts: ['.ts.net', 'localhost'],
  },
});
