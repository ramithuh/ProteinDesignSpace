// Zero-dependency typed-array-preserving wire codec for self-hosted remote GPU execution.
// Preserves Float32Array, Uint8Array, Int32Array, Float64Array, Uint16Array, Uint32Array
// across JSON/HTTP boundaries without losing binary precision or prototype identity.

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

const TYPED_ARRAY_CTORS: Record<
  TypedArrayName,
  new (buffer: ArrayBuffer, byteOffset?: number, length?: number) => SupportedTypedArray
> = {
  Float32Array,
  Float64Array,
  Uint8Array,
  Int32Array,
  Uint16Array,
  Uint32Array,
};

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

function isEncodedTypedArray(val: unknown): val is EncodedTypedArray {
  if (!val || typeof val !== 'object') return false;
  const o = val as Record<string, unknown>;
  return (
    typeof o.__denovoTypedArray === 'string' &&
    o.__denovoTypedArray in TYPED_ARRAY_CTORS &&
    typeof o.b64 === 'string'
  );
}

/** Recursively convert an artifact or payload into a JSON-safe object preserving TypedArrays. */
export function encodeWireValue(value: unknown): unknown {
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
    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    return {
      __denovoTypedArray: taName,
      b64: bytesToBase64(bytes),
    } satisfies EncodedTypedArray;
  }

  if (Array.isArray(value)) {
    return value.map((item) => encodeWireValue(item));
  }

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (v === undefined) continue;
    out[k] = encodeWireValue(v);
  }
  return out;
}

/** Recursively restore TypedArrays and special float values from a JSON wire payload. */
export function decodeWireValue<T = unknown>(value: unknown): T {
  if (value === null || value === undefined || typeof value !== 'object') {
    return value as T;
  }
  if (Array.isArray(value)) {
    return value.map((item) => decodeWireValue(item)) as T;
  }

  const obj = value as Record<string, unknown>;
  if (typeof obj.__denovoNum === 'string' && Object.keys(obj).length === 1) {
    if (obj.__denovoNum === 'NaN') return Number.NaN as T;
    if (obj.__denovoNum === 'Inf') return Number.POSITIVE_INFINITY as T;
    if (obj.__denovoNum === '-Inf') return Number.NEGATIVE_INFINITY as T;
  }

  if (isEncodedTypedArray(obj)) {
    const bytes = base64ToBytes(obj.b64);
    const Ctor = TYPED_ARRAY_CTORS[obj.__denovoTypedArray];
    // Copy into an aligned ArrayBuffer so Float32Array/Float64Array/Int32Array never hit alignment issues
    const aligned = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(aligned).set(bytes);
    return new Ctor(aligned) as T;
  }

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    out[k] = decodeWireValue(v);
  }
  return out as T;
}
