#!/usr/bin/env node
/**
 * Packs the three PULCHRA 3.06 statistical tables (nco_data.h, rot_data_idx.h,
 * rot_data_coords.h — ~7.4 MB of C initialiser text) into one compact,
 * zero-copy-readable binary: web/static/vendor/pulchra/pulchra-tables.bin.
 *
 * Usage: node tools/pack-pulchra-tables.mjs [pulchraSrcDir] [outFile]
 *   pulchraSrcDir defaults to /tmp/pulchra-ref (the upstream MIT sources).
 *
 * Every C table value is a decimal literal with <= 3 fractional digits stored
 * in a `float`, so it is encoded exactly as int16 fixed point (q = v * 1000):
 * the runtime decode `Math.fround(q / 1000)` reproduces the compiler's
 * (float)(double)literal bit-for-bit (IEEE division is correctly rounded, so
 * q / 1000 is the same double the C parser produces). The script asserts this
 * for every value and re-decodes the written file before exiting.
 *
 * Layout (little-endian; every section starts on an 8-byte boundary):
 *   header (64 B)
 *     0  char[8] magic "PULCHRA\0"
 *     8  u32 format version (1)
 *    12  u32 fixed-point scale (1000)
 *    16  u32 nco_stat entries        (incl. the {-1,-1,-1} terminator)
 *    20  u32 nco_stat_pro entries    (incl. terminator)
 *    24  u32 rot_stat_idx entries    (incl. {-1,...} terminator)
 *    28  u32 rot_stat_coords rows
 *    32  u32 offset ncoBins       int8  [nco * 3]
 *    36  u32 offset ncoData       int16 [nco * 24]   (8 points x xyz)
 *    40  u32 offset ncoProBins    int8  [ncoPro * 3]
 *    44  u32 offset ncoProData    int16 [ncoPro * 24]
 *    48  u32 offset rotIdx        int8  [rot * 5]    (aa, r13_1, r13_2, r14, nrot)
 *    52  u32 offset rotPos        u32   [rot]        (first row in rotCoords)
 *    56  u32 offset rotCoords     int16 [rows * 3]
 *    60  u32 total byte length
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(process.argv[2] ?? '/tmp/pulchra-ref');
const outFile = resolve(
  process.argv[3] ?? resolve(__dirname, '../web/static/vendor/pulchra/pulchra-tables.bin'),
);

const MAGIC = 'PULCHRA\0';
const FORMAT_VERSION = 1;
const SCALE = 1000;
const HEADER_BYTES = 64;

function fail(msg) {
  console.error(`[pack-pulchra-tables] FAIL: ${msg}`);
  process.exit(1);
}

/** Body of `<decl> = { ... };` for the named C array. */
function arrayBody(text, name) {
  const decl = new RegExp(`\\b${name}\\[\\](\\[\\d+\\])?\\s*=\\s*\\{`).exec(text);
  if (!decl) fail(`array ${name} not found`);
  const open = decl.index + decl[0].length;
  const end = text.indexOf('};', open);
  if (end < 0) fail(`array ${name} is not terminated`);
  return text.slice(open, end);
}

/** Exact int16 fixed-point encoding of a C float literal; asserts losslessness. */
function fixed(literal) {
  if (!/^-?\d+(\.\d{1,3})?$/.test(literal)) fail(`literal ${literal} has > 3 decimals`);
  const v = Number(literal);
  const q = Math.round(v * SCALE);
  if (q < -32768 || q > 32767) fail(`literal ${literal} overflows int16 fixed point`);
  if (Math.fround(q / SCALE) !== Math.fround(v)) fail(`literal ${literal} is not exact at scale ${SCALE}`);
  if (Object.is(v, -0)) fail(`negative zero literal ${literal} is not representable`);
  return q;
}

function parseNco(text, name) {
  const body = arrayBody(text, name);
  const re = /\{\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*,\s*\{([^}]*)\}\s*\}/g;
  const entries = [];
  for (const m of body.matchAll(re)) {
    const literals = m[4].split(',').map((s) => s.trim()).filter(Boolean);
    if (literals.length !== 24) fail(`${name}[${entries.length}] has ${literals.length} floats, expected 24`);
    entries.push({ bins: [+m[1], +m[2], +m[3]], literals });
  }
  const last = entries.at(-1);
  if (!last || last.bins[0] !== -1) fail(`${name} is missing its {-1,-1,-1} terminator`);
  return entries;
}

function parseRotIdx(text) {
  const body = arrayBody(text, 'rot_stat_idx');
  const re = /\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}/g;
  const rows = [...body.matchAll(re)].map((m) => m.slice(1, 7).map(Number));
  if (!rows.length || rows.at(-1)[0] !== -1) fail('rot_stat_idx is missing its terminator');
  return rows;
}

function parseRotCoords(text) {
  const body = arrayBody(text, 'rot_stat_coords');
  const re = /\{\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\}/g;
  return [...body.matchAll(re)].map((m) => [m[1], m[2], m[3]]);
}

const nco = readFileSync(resolve(srcDir, 'nco_data.h'), 'utf8');
const ncoStat = parseNco(nco, 'nco_stat');
const ncoStatPro = parseNco(nco, 'nco_stat_pro');
const rotIdx = parseRotIdx(readFileSync(resolve(srcDir, 'rot_data_idx.h'), 'utf8'));
const rotCoords = parseRotCoords(readFileSync(resolve(srcDir, 'rot_data_coords.h'), 'utf8'));

for (const [i, row] of rotIdx.entries()) {
  for (let c = 0; c < 5; c++) {
    if (row[c] < -128 || row[c] > 127) fail(`rot_stat_idx[${i}][${c}] = ${row[c]} overflows int8`);
  }
  if (row[0] >= 0 && row[5] + row[4] > rotCoords.length) fail(`rot_stat_idx[${i}] points past rot_stat_coords`);
}

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
  fail('packing requires a little-endian host (typed arrays are written natively)');
}

const align8 = (n) => (n + 7) & ~7;
const layout = [];
let cursor = HEADER_BYTES;
function section(bytes) {
  const offset = cursor;
  cursor = align8(cursor + bytes);
  layout.push(offset);
  return offset;
}
const offNcoBins = section(ncoStat.length * 3);
const offNcoData = section(ncoStat.length * 24 * 2);
const offNcoProBins = section(ncoStatPro.length * 3);
const offNcoProData = section(ncoStatPro.length * 24 * 2);
const offRotIdx = section(rotIdx.length * 5);
const offRotPos = section(rotIdx.length * 4);
const offRotCoords = section(rotCoords.length * 3 * 2);
const totalBytes = cursor;

const buf = new ArrayBuffer(totalBytes);
const view = new DataView(buf);
for (let i = 0; i < 8; i++) view.setUint8(i, MAGIC.charCodeAt(i));
const header = [
  FORMAT_VERSION, SCALE, ncoStat.length, ncoStatPro.length, rotIdx.length, rotCoords.length,
  ...layout, totalBytes,
];
header.forEach((v, i) => view.setUint32(8 + 4 * i, v, true));

function writeNco(entries, offBins, offData) {
  const bins = new Int8Array(buf, offBins, entries.length * 3);
  const data = new Int16Array(buf, offData, entries.length * 24);
  entries.forEach((e, i) => {
    bins.set(e.bins, i * 3);
    e.literals.forEach((lit, k) => (data[i * 24 + k] = fixed(lit)));
  });
}
writeNco(ncoStat, offNcoBins, offNcoData);
writeNco(ncoStatPro, offNcoProBins, offNcoProData);

const idx8 = new Int8Array(buf, offRotIdx, rotIdx.length * 5);
const pos32 = new Uint32Array(buf, offRotPos, rotIdx.length);
rotIdx.forEach((row, i) => {
  idx8.set(row.slice(0, 5), i * 5);
  pos32[i] = row[5];
});
const coords16 = new Int16Array(buf, offRotCoords, rotCoords.length * 3);
rotCoords.forEach((xyz, i) => xyz.forEach((lit, k) => (coords16[i * 3 + k] = fixed(lit))));

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, new Uint8Array(buf));

// Round-trip: re-read from disk and compare every decoded value to the literal.
const back = readFileSync(outFile);
const rb = back.buffer.slice(back.byteOffset, back.byteOffset + back.byteLength);
const dec = (arr, i) => Math.fround(arr[i] / SCALE);
function verifyNco(entries, offBins, offData, name) {
  const bins = new Int8Array(rb, offBins, entries.length * 3);
  const data = new Int16Array(rb, offData, entries.length * 24);
  entries.forEach((e, i) => {
    for (let k = 0; k < 3; k++) if (bins[i * 3 + k] !== e.bins[k]) fail(`${name}[${i}].bins mismatch`);
    for (let k = 0; k < 24; k++) {
      if (dec(data, i * 24 + k) !== Math.fround(Number(e.literals[k]))) fail(`${name}[${i}].data[${k}] mismatch`);
    }
  });
}
verifyNco(ncoStat, offNcoBins, offNcoData, 'nco_stat');
verifyNco(ncoStatPro, offNcoProBins, offNcoProData, 'nco_stat_pro');
const ri = new Int8Array(rb, offRotIdx, rotIdx.length * 5);
const rp = new Uint32Array(rb, offRotPos, rotIdx.length);
rotIdx.forEach((row, i) => {
  for (let c = 0; c < 5; c++) if (ri[i * 5 + c] !== row[c]) fail(`rot_stat_idx[${i}][${c}] mismatch`);
  if (rp[i] !== row[5]) fail(`rot_stat_idx[${i}][5] mismatch`);
});
const rc = new Int16Array(rb, offRotCoords, rotCoords.length * 3);
rotCoords.forEach((xyz, i) =>
  xyz.forEach((lit, k) => {
    if (dec(rc, i * 3 + k) !== Math.fround(Number(lit))) fail(`rot_stat_coords[${i}][${k}] mismatch`);
  }),
);

console.log(
  `[pack-pulchra-tables] wrote ${outFile} (${totalBytes} bytes): ` +
    `nco_stat=${ncoStat.length} nco_stat_pro=${ncoStatPro.length} ` +
    `rot_stat_idx=${rotIdx.length} rot_stat_coords=${rotCoords.length}; round-trip verified`,
);
