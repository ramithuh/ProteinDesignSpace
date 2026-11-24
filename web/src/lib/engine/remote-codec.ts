// Zero-dependency typed-array-preserving wire codec for self-hosted remote GPU execution.
// Hardened against prototype pollution (__proto__/constructor/prototype), excessive nesting,
// element-count / memory bombs, and unaligned TypedArray allocations.

type SupportedTypedArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int32Array
  | Uint16Array
  | Uint32Array;

type TypedArrayName =
  | 'Float32Array'
  | 'Float64Array'
  | 'Uint8Array'
  | 'Int32Array'
  | 'Uint16Array'
  | 'Uint32Array';

interface EncodedTypedArray {
  __denovoTypedArray: TypedArrayName;
  b64: string;
}

interface TypedArrayCtorSpec {
  ctor: new (buffer: ArrayBuffer, byteOffset?: number, length?: number) => SupportedTypedArray;
  bytesPerElement: number;
}

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const MAX_CODEC_DEPTH = 32;
const MAX_CODEC_NODES = 250_000;
const MAX_DECODED_BINARY_BYTES = 128 * 1024 * 1024; // 128 MiB aggregate TypedArray budget

const TYPED_ARRAY_MAP = new Map<TypedArrayName, TypedArrayCtorSpec>([
  ['Float32Array', { ctor: Float32Array, bytesPerElement: 4 }],
  ['Float64Array', { ctor: Float64Array, bytesPerElement: 8 }],
  ['Uint8Array', { ctor: Uint8Array, bytesPerElement: 1 }],
  ['Int32Array', { ctor: Int32Array, bytesPerElement: 4 }],
  ['Uint16Array', { ctor: Uint16Array, bytesPerElement: 2 }],
  ['Uint32Array', { ctor: Uint32Array, bytesPerElement: 4 }],
]);

function bytesToBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  }
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, Math.min(bytes.length, i + chunk));
    let part = '';
    for (let j = 0; j < sub.length; j++) {
      part += String.fromCharCode(sub[j]);
    }
    binary += part;
  }
  return btoa(binary);
}

function estimateBase64DecodedBytes(b64: string): number {
  const len = b64.length;
  if (len === 0) return 0;
  let pad = 0;
  if (b64.endsWith('==')) pad = 2;
  else if (b64.endsWith('=')) pad = 1;
  return Math.max(0, Math.floor((len * 3) / 4) - pad);
}

function base64ToBytes(b64: string): Uint8Array {
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(b64, 'base64');
    const copy = new Uint8Array(buf.byteLength);
    copy.set(buf);
    return copy;
  }
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

function detectTypedArrayName(val: unknown): TypedArrayName | null {
  if (val instanceof Float32Array) return 'Float32Array';
  if (val instanceof Float64Array) return 'Float64Array';
  if (val instanceof Uint8Array) return 'Uint8Array';
  if (val instanceof Int32Array) return 'Int32Array';
  if (val instanceof Uint16Array) return 'Uint16Array';
  if (val instanceof Uint32Array) return 'Uint32Array';
  return null;
}

function getEncodedTypedArraySpec(val: unknown): { spec: TypedArrayCtorSpec; b64: string } | null {
  if (!val || typeof val !== 'object' || Array.isArray(val)) return null;
  const o = val as Record<string, unknown>;
  const keys = Object.keys(o);
  if (keys.length !== 2) return null;
  if (typeof o.__denovoTypedArray !== 'string' || typeof o.b64 !== 'string') return null;
  const spec = TYPED_ARRAY_MAP.get(o.__denovoTypedArray as TypedArrayName);
  if (!spec) return null;
  return { spec, b64: o.b64 };
}

interface CodecBudget {
  nodes: number;
  binaryBytes: number;
}

function encodeRecursive(value: unknown, depth: number, budget: CodecBudget): unknown {
  if (depth > MAX_CODEC_DEPTH) {
    throw new Error(`Wire payload exceeds maximum nesting depth (${MAX_CODEC_DEPTH}).`);
  }
  if (++budget.nodes > MAX_CODEC_NODES) {
    throw new Error(`Wire payload exceeds maximum node count (${MAX_CODEC_NODES}).`);
  }
  if (value === null || value === undefined) return value;
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return { __denovoNum: 'NaN' };
    if (value === Infinity) return { __denovoNum: 'Inf' };
    if (value === -Infinity) return { __denovoNum: '-Inf' };
    return value;
  }
  if (typeof value !== 'object') return value;

  const taName = detectTypedArrayName(value);
  if (taName) {
    const view = value as SupportedTypedArray;
    budget.binaryBytes += view.byteLength;
    if (budget.binaryBytes > MAX_DECODED_BINARY_BYTES) {
      throw new Error('Wire payload exceeds maximum binary buffer budget.');
    }
    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    return {
      __denovoTypedArray: taName,
      b64: bytesToBase64(bytes),
    } satisfies EncodedTypedArray;
  }

  if (Array.isArray(value)) {
    return value.map((item) => encodeRecursive(item, depth + 1, budget));
  }

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(k) || v === undefined) continue;
    out[k] = encodeRecursive(v, depth + 1, budget);
  }
  return out;
}

function decodeRecursive(value: unknown, depth: number, budget: CodecBudget): unknown {
  if (depth > MAX_CODEC_DEPTH) {
    throw new Error(`Wire payload exceeds maximum nesting depth (${MAX_CODEC_DEPTH}).`);
  }
  if (++budget.nodes > MAX_CODEC_NODES) {
    throw new Error(`Wire payload exceeds maximum node count (${MAX_CODEC_NODES}).`);
  }
  if (value === null || value === undefined || typeof value !== 'object') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => decodeRecursive(item, depth + 1, budget));
  }

  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length === 1 && typeof obj.__denovoNum === 'string') {
    if (obj.__denovoNum === 'NaN') return Number.NaN;
    if (obj.__denovoNum === 'Inf') return Number.POSITIVE_INFINITY;
    if (obj.__denovoNum === '-Inf') return Number.NEGATIVE_INFINITY;
  }

  const taTag = getEncodedTypedArraySpec(obj);
  if (taTag) {
    const estBytes = estimateBase64DecodedBytes(taTag.b64);
    if (budget.binaryBytes + estBytes > MAX_DECODED_BINARY_BYTES) {
      throw new Error('Wire payload exceeds maximum decoded binary byte budget.');
    }
    const bytes = base64ToBytes(taTag.b64);
    budget.binaryBytes += bytes.byteLength;
    if (budget.binaryBytes > MAX_DECODED_BINARY_BYTES) {
      throw new Error('Wire payload exceeds maximum decoded binary byte budget.');
    }
    if (bytes.byteLength % taTag.spec.bytesPerElement !== 0) {
      throw new Error(
        `Invalid TypedArray byte length (${bytes.byteLength}) for element size ${taTag.spec.bytesPerElement}.`,
      );
    }
    const aligned = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(aligned).set(bytes);
    return new taTag.spec.ctor(aligned);
  }

  const out: Record<string, unknown> = {};
  for (const k of keys) {
    if (FORBIDDEN_KEYS.has(k)) {
      throw new Error(`Forbidden property key "${k}" in wire payload.`);
    }
    Object.defineProperty(out, k, {
      value: decodeRecursive(obj[k], depth + 1, budget),
      enumerable: true,
      writable: true,
      configurable: true,
    });
  }
  return out;
}

/** Recursively convert an artifact or payload into a JSON-safe object preserving TypedArrays. */
export function encodeWireValue(value: unknown): unknown {
  return encodeRecursive(value, 0, { nodes: 0, binaryBytes: 0 });
}

/** Recursively restore TypedArrays and special float values from a JSON wire payload. */
export function decodeWireValue<T = unknown>(value: unknown): T {
  return decodeRecursive(value, 0, { nodes: 0, binaryBytes: 0 }) as T;
}
