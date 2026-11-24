import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    alias: {
      'proteinmpnn-web': 'node_modules/proteinmpnn-web/src/index.ts',
    },
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: true,
    }),
    typescript: {
      config(tsConfig) {
        tsConfig.exclude = [...(tsConfig.exclude ?? []), '../tests/e2e-localfold-template/**'];
        return tsConfig;
      },
    },
  },
};

export default config;
