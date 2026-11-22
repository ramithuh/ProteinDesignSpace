import { describe, expect, it } from 'vitest';
import { PAPER_INSIGHT_EXAMPLES } from '$lib/engine/paper-gallery';
import { decodeHashToGraph } from '$lib/engine/share';
import { validateGraph } from '$lib/engine/graph-utils';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';

describe('PAPER_INSIGHT_EXAMPLES gallery', () => {
  it('contains valid, reloadable #g= workflows with clean citations, tools, and valid walkthrough steps', async () => {
    expect(PAPER_INSIGHT_EXAMPLES.length).toBe(3);
    for (const ex of PAPER_INSIGHT_EXAMPLES) {
      expect(ex.citation.trim().length).toBeGreaterThan(5);
      expect(ex.tools.length).toBeGreaterThanOrEqual(2);
      expect(ex.walkthrough.length).toBeGreaterThanOrEqual(2);
      expect(ex.hash.startsWith('#g=')).toBe(true);
      expect(ex.hash).not.toMatch(/[_-]$/);

      const decoded = await decodeHashToGraph(ex.hash);
      expect(decoded, `Failed to decode ${ex.id}`).not.toBeNull();
      expect(decoded!.nodes.length).toBeGreaterThanOrEqual(3);
      expect(decoded!.nodes.length).toBeLessThanOrEqual(4);
      expect(validateGraph(decoded!)).toEqual([]);

      const nodeById = new Map(decoded!.nodes.map((n) => [n.id, n]));
      for (const step of ex.walkthrough) {
        const targetNode = nodeById.get(step.nodeId);
        expect(targetNode, `${ex.id} walkthrough references unknown nodeId "${step.nodeId}"`).toBeDefined();
        if (step.outputPort) {
          const manifest = ADAPTER_REGISTRY[targetNode!.adapterId];
          expect(
            Object.hasOwn(manifest.outputs, step.outputPort),
            `${ex.id} walkthrough references unknown outputPort "${step.outputPort}" on ${targetNode!.adapterId}`,
          ).toBe(true);
        }
      }
    }
  });
});
