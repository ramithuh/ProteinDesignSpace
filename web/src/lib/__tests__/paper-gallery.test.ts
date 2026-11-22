import { describe, expect, it } from 'vitest';
import { PAPER_INSIGHT_EXAMPLES } from '$lib/engine/paper-gallery';
import { decodeHashToGraph } from '$lib/engine/share';
import { validateGraph } from '$lib/engine/graph-utils';

describe('PAPER_INSIGHT_EXAMPLES gallery', () => {
  it('contains valid, reloadable #g= workflows with clean citations and tools', async () => {
    expect(PAPER_INSIGHT_EXAMPLES.length).toBe(3);
    for (const ex of PAPER_INSIGHT_EXAMPLES) {
      expect(ex.citation.trim().length).toBeGreaterThan(5);
      expect(ex.tools.length).toBeGreaterThanOrEqual(2);
      expect(ex.hash.startsWith('#g=')).toBe(true);
      expect(ex.hash).not.toMatch(/[_-]$/);
      const decoded = await decodeHashToGraph(ex.hash);
      expect(decoded, `Failed to decode ${ex.id}`).not.toBeNull();
      expect(decoded!.nodes.length).toBeGreaterThanOrEqual(3);
      expect(decoded!.nodes.length).toBeLessThanOrEqual(4);
      expect(validateGraph(decoded!)).toEqual([]);
    }
  });
});
