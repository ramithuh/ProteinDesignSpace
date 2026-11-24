import { defineConfig } from 'vitest/config';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';

export default defineConfig({
  server: {
    fs: {
      strict: true,
      allow: [resolve(__dirname), resolve(__dirname, 'static'), resolve(__dirname, 'node_modules'), tmpdir()],
    },
  },
  resolve: {
    alias: {
      $lib: resolve(__dirname, 'src/lib'),
      'proteinmpnn-web': resolve(__dirname, 'node_modules/proteinmpnn-web/src/index.ts'),
    },
  },
  test: {
    environment: 'node',
    include: ['tests/e2e-localfold-template/**/*.e2e.test.ts'],
    globals: false,
    testTimeout: 30000,
  },
});
