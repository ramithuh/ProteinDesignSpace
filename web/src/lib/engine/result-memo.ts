// Estimate retained payload without serializing large PDB strings or traversing typed arrays.
// Views retain their complete backing buffer. Shared objects are charged once per entry.
function estimateBytes(value: unknown, seen: Set<unknown>): number {
  if (typeof value === 'string') {
    if (value.length > 256) {
      if (seen.has(value)) return 0;
      seen.add(value);
    }
    return 16 + 2 * value.length;
  }
  if (value === null || typeof value !== 'object') return 8;
  if (seen.has(value)) return 0;
  seen.add(value);
  if (ArrayBuffer.isView(value)) {
    return 64 + estimateBytes(value.buffer, seen);
  }
  if (value instanceof ArrayBuffer || (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer)) {
    return 32 + value.byteLength;
  }
  let bytes = 64;
  if (Array.isArray(value)) {
    for (const item of value) bytes += 8 + estimateBytes(item, seen);
  } else {
    for (const [key, item] of Object.entries(value)) {
      bytes += 16 + 2 * key.length + estimateBytes(item, seen);
    }
  }
  return bytes;
}

export class ResultMemo<T extends object> {
  private readonly entries = new Map<string, { value: T; bytes: number }>();
  private usedBytes = 0;

  constructor(private readonly maxBytes = 64 * 1024 * 1024, private readonly maxEntries = 64) {}

  get(key: string): T | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    this.entries.delete(key);
    this.entries.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T): void {
    const bytes = 64 + key.length * 2 + estimateBytes(value, new Set());
    this.remove(key);
    if (!(bytes <= this.maxBytes) || this.maxEntries < 1) return;
    while (this.usedBytes + bytes > this.maxBytes || this.entries.size >= this.maxEntries) {
      const oldest = this.entries.keys().next();
      if (oldest.done) break;
      this.remove(oldest.value);
    }
    this.entries.set(key, { value, bytes });
    this.usedBytes += bytes;
  }

  private remove(key: string): void {
    const entry = this.entries.get(key);
    if (entry) {
      this.usedBytes -= entry.bytes;
      this.entries.delete(key);
    }
  }

  clear(): void {
    this.entries.clear();
    this.usedBytes = 0;
  }
}
