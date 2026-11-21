import { describe, expect, it } from 'vitest';
import { ResultMemo } from '$lib/engine/result-memo';
import { ExecutionScheduler } from '$lib/engine/scheduler';
import type { NodeProgressEvent, WorkflowGraph } from '$lib/engine/types';

describe('bounded result memoization', () => {
  it('charges backing buffers once per entry and evicts by estimated bytes', () => {
    const memo = new ResultMemo<object>(1800);
    const buffer = new ArrayBuffer(1024);
    const outputs = { a: new Float32Array(buffer), b: new Uint8Array(buffer, 0, 1) };
    memo.set('a', outputs);
    expect(memo.get('a')).toBe(outputs);
    memo.set('b', { a: new Uint8Array(1024) });
    expect(memo.get('a')).toBeUndefined();
    expect(memo.get('b')).toBeDefined();
    memo.set('c', { tinyView: new Uint8Array(new ArrayBuffer(4096), 0, 1) });
    expect(memo.get('c')).toBeUndefined();
    expect(memo.get('b')).toBeDefined();
    memo.set('b', { text: 'x'.repeat(1800) });
    expect(memo.get('b')).toBeUndefined();
    memo.clear();
    memo.set('a', outputs);
    expect(memo.get('a')).toBe(outputs);

    // Shared large strings (e.g. designedPdb duplicated across scorer rows) are charged once per entry
    const sharedPdb = 'ATOM '.repeat(100); // 500 chars (~1016 bytes UTF-16)
    const scorerMemo = new ResultMemo<object>(2400);
    const rows = Array.from({ length: 8 }, () => ({ designedPdb: sharedPdb }));
    scorerMemo.set('metrics', { rows });
    expect(scorerMemo.get('metrics')).toEqual({ rows });

    // NaN maxBytes must safely reject entries instead of admitting unbounded payloads
    const nanMemo = new ResultMemo<object>(Number.NaN);
    nanMemo.set('x', { ok: true });
    expect(nanMemo.get('x')).toBeUndefined();
  });

  it('refreshes recent hits and keeps caller-held outputs valid after scheduler eviction', async () => {
    const scheduler = new ExecutionScheduler();
    async function run(label: string) {
      const graph: WorkflowGraph = {
        id: 'memo', name: 'Memo',
        nodes: [{ id: 's', adapterId: 'sequence-input', params: { sequence: 'ACDEF', label } }],
        edges: [],
      };
      const events: NodeProgressEvent[] = [];
      const result = await scheduler.run(graph, (ev) => events.push(ev));
      return { result, cached: events.some((ev) => ev.status === 'cached') };
    }
    const first = await run('0');
    const second = await run('1');
    for (let i = 2; i < 64; i++) await run(String(i));
    expect((await run('0')).cached).toBe(true);
    await run('64');
    expect((await run('0')).cached).toBe(true);
    expect((await run('1')).cached).toBe(false);
    expect(second.result.outputsByNode.s.sequences).toMatchObject({ candidates: [{ sequence: 'ACDEF', id: '1' }] });
    expect(first.result.refsByNode.s.sequences.summary.count).toBe(1);
  });
});
