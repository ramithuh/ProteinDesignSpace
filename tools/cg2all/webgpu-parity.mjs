// WebGPU parity for the cg2all port in headless Chromium (SwiftShader WebGPU adapter): serves
// web/static/vendor/cg2all, weight bundles and reference_forward.py goldens over localhost and prints
// all-atom RMSD vs PyTorch per (variant, precision, input).
// usage: EXE=<chromium> CG2ALL_WEIGHTS=<dir> CG2ALL_GOLDEN=<dir> CG2ALL_INPUTS=<dir> \
//        CASES=ca:CalphaBasedModel:fp16,ca-fix:CalphaBasedModel-FIX:int8 node tools/cg2all/webgpu-parity.mjs
import http from 'node:http';
import { readFileSync, existsSync, rmSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { chromium } from '../../web/node_modules/playwright/index.mjs';
const env = (k, d) => (process.env[k] ? process.env[k].replace(/\/?$/, '/') : d);
const ROOTS = { '/vendor/': new URL('../../web/static/vendor/', import.meta.url).pathname, '/weights/': env('CG2ALL_WEIGHTS', '/tmp/cg2all-weights/'), '/wref/': env('CG2ALL_WEIGHTS_FP32', '/tmp/cg2all-weights-fp32/'), '/golden/': env('CG2ALL_GOLDEN', '/tmp/cg2all-golden/'), '/inputs/': env('CG2ALL_INPUTS', '/tmp/cg2all-inputs/') };
const TYPES = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.pdb': 'text/plain', '.bin': 'application/octet-stream' };
const srv = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  if (url === '/') { res.setHeader('content-type', 'text/html'); return res.end('<html><body>cg2all</body></html>'); }
  for (const [pre, dir] of Object.entries(ROOTS)) {
    if (url.startsWith(pre)) {
      const f = join(dir, url.slice(pre.length));
      if (existsSync(f) && statSync(f).isFile()) { res.setHeader('content-type', TYPES[extname(f)] ?? 'application/octet-stream'); return res.end(readFileSync(f)); }
    }
  }
  res.statusCode = 404; res.end('nf');
}).listen(18770);
const args = ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-dev-shm-usage', '--ozone-platform=headless'];
const profile = join(tmpdir(), `cg2all-webgpu-parity-${process.pid}`);
const ctx = await chromium.launchPersistentContext(profile, { headless: false, executablePath: process.env.EXE, args, env: { PATH: process.env.PATH, HOME: process.env.HOME } });
const page = ctx.pages()[0] ?? await ctx.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
await page.goto('http://localhost:18770/');
const cases = (process.env.CASES || 'ca:CalphaBasedModel:fp16,ca-fix:CalphaBasedModel-FIX:int8').split(',');
const names = (process.env.NAMES || 'helix16,crambin46,twochain').split(',');
const result = await page.evaluate(async ({ cases, names }) => {
  const mod = await import('/vendor/cg2all/webgpu/index.js');
  const { AMINO_ACIDS } = await import('/vendor/cg2all/shared/constants.js');
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  device.addEventListener('uncapturederror', (ev) => console.log('uncaptured', ev.error.message));
  const out = [];
  for (const c of cases) {
    const [variant, ckpt, prec] = c.split(':');
    const root = prec === 'fp32' ? '/wref' : '/weights';
    const base = `${root}/${variant}/${prec}`;
    const weights = await mod.loadWeights(base, { read: async (name) => new Uint8Array(await (await fetch(`${base}/${name}`)).arrayBuffer()) });
    const engine = await mod.createCg2allEngine({ device, weights });
    for (const name of names) {
      const g = await (await fetch(`/golden/${name}.${ckpt}.json`)).json();
      const pdb = await (await fetch(`/inputs/${name}.pdb`)).text();
      const prep = mod.prepareInput(pdb);
      const t0 = performance.now();
      const o = await engine.predict(prep);
      const ms = performance.now() - t0;
      const res = mod.assemble(prep, o, { fixAtom: weights.manifest.fixAtom, hydrogens: true });
      const mask = g.output_atom_mask.flat(); const gr = g.R.flat(Infinity);
      let ss = 0, na = 0, mx = 0, nan = 0;
      for (let i = 0; i < prep.n; i++) for (let a = 0; a < 24; a++) {
        if (!mask[i * 24 + a] || AMINO_ACIDS[prep.residueType[i]] === 'UNK') continue;
        let d2 = 0; for (let k = 0; k < 3; k++) { const v = res.R[(i * 24 + a) * 3 + k]; if (!Number.isFinite(v)) nan++; d2 += (100 * (gr[(i * 24 + a) * 3 + k] - v) ** 2); }
        ss += d2; na++; mx = Math.max(mx, Math.sqrt(d2));
      }
      let sdiff = 0; const gs = g.struct['0'].flat(); for (let i = 0; i < gs.length; i++) sdiff = Math.max(sdiff, Math.abs(gs[i] - o.s[i]));
      const ssMatch = g.ss.every((v, i) => v === res.ss[i]);
      out.push(`${variant}/${prec} ${name}: N=${prep.n} E=${prep.e} rmsd=${Math.sqrt(ss / na).toFixed(5)}Å max=${mx.toFixed(4)}Å nonfinite=${nan} ssMatch=${ssMatch} max|Δs|=${sdiff.toExponential(2)} ${ms.toFixed(0)}ms`);
    }
    engine.destroy();
  }
  return out;
}, { cases, names });
console.log(result.join('\n'));
await ctx.close(); srv.close();
rmSync(profile, { recursive: true, force: true });
