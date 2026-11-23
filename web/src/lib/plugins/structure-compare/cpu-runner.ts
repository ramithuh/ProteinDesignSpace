// Layer 3 Plugin CPU Runner — structure-compare
// Multi-structure fan-in comparator packaging a reference and upstream structures into a Collection.

import type {
  CanonicalArtifact,
  NodeProgressEvent,
  StructureArtifact,
  WorkflowNode,
} from '$lib/engine/types';

export async function runStructureCompareCpu(
  node: WorkflowNode,
  inputs: Record<string, CanonicalArtifact>,
  multiInputs?: Record<string, CanonicalArtifact[]>,
  onProgress?: (ev: NodeProgressEvent) => void,
): Promise<Record<string, CanonicalArtifact>> {
  const ref = inputs.reference as StructureArtifact | undefined;
  const structList = (
    multiInputs?.structures ?? (inputs.structures ? [inputs.structures] : [])
  ).filter((a): a is StructureArtifact => a?.kind === 'StructureArtifact');
  const primary = structList[0] ?? (ref?.kind === 'StructureArtifact' ? ref : undefined);
  if (!primary) {
    throw new Error('structure-compare requires at least one upstream StructureArtifact');
  }
  const layout = String(node.params.layout ?? 'overlay');
  const colorMode = String(node.params.colorMode ?? 'by-model');
  const alignKabsch = node.params.alignKabsch !== false;
  onProgress?.({
    nodeId: node.id,
    status: 'running',
    phase: 'compare',
    progress: 0.5,
    detail: `Comparing structures (${layout}, ${alignKabsch ? 'Kabsch' : 'raw'}, ${colorMode})...`,
  });
  const seenIds = new Set<string>();
  const combinedItems: StructureArtifact[] = [];
  const addItem = (item: StructureArtifact | undefined) => {
    if (!item || seenIds.has(item.id)) return;
    seenIds.add(item.id);
    combinedItems.push(item);
  };
  addItem(ref);
  for (const s of structList) addItem(s);

  if (combinedItems.length <= 1 && !ref) {
    return { structure: primary };
  }
  const keys = combinedItems.map((item, idx) => item.designKey ?? item.id ?? `item-${idx}`);
  const outputStructure: StructureArtifact = {
    ...primary,
    ...(ref && primary.id !== ref.id ? { parentStructureId: ref.id } : {}),
    ...(combinedItems.length > 1
      ? {
          collection: {
            kind: 'Collection',
            schema: 'denovo.collection/v1',
            id: `${primary.id}::compare-${combinedItems.length}`,
            collectionKind: 'independent-candidates',
            requestedCount: combinedItems.length,
            complete: true,
            items: combinedItems,
            keys,
          },
        }
      : {}),
  };
  return { structure: outputStructure };
}
