import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  // onnxruntime-web ships its own WASM artifacts and uses dynamic imports —
  // letting Vite optimize-deps it usually breaks the WASM resolution.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  // Allow serving large model files via symlink out of /static.
  server: {
    fs: { strict: false },
  },
});
