/**
 * Post-parse sanitizers for Mol* atomic models.
 *
 * When Mol* parses a PDB file (which lacks mmCIF `_chem_comp.type`), its
 * `ComponentBuilder.getType()` infers component types solely from atom names:
 * any residue containing `O3'`/`C4'`/`O2'` (such as ribonucleoside/cofactor
 * ligands like `DHZ`, `SAM`, `SAH`, `NAD`, `FAD`, `CMP`, `UMP`, `CTP`, `UTP`)
 * is misclassified as `'rna linking'` (`MoleculeType.RNA`) unless hardcoded in
 * Mol*'s tiny 11-entry `NonPolymerNames` set.
 *
 * Because Mol*'s `getNucleotideElements()` and `createNucleotideRingMesh()`
 * iterate all residues where `isNucleic(moleculeType[ri])` — bypassing
 * `polymerRanges` — a single-residue nucleoside ligand gets a giant cartoon
 * base-stick (`C4'`/`C3'` -> `N1`/`N9`) and solid filled ring slab drawn on
 * top of its ball-and-stick representation.
 */

// Numeric enum values from molstar `mol-model/structure/model/types.ts`:
// MoleculeType: Other = 1, RNA = 6, DNA = 7, PNA = 8
// PolymerType: NA = 0
const MOLECULE_TYPE_OTHER = 1;
const MOLECULE_TYPE_RNA = 6;
const MOLECULE_TYPE_DNA = 7;
const MOLECULE_TYPE_PNA = 8;
const POLYMER_TYPE_NA = 0;

const STANDARD_NUCLEIC_COMP_IDS = new Set([
  'A',
  'C',
  'G',
  'U',
  'I',
  'T',
  'DA',
  'DC',
  'DG',
  'DT',
  'DI',
  'DU',
]);

const O3_ATOM_IDS = new Set(["O3'", 'O3*']);
const P_ATOM_IDS = new Set(['P']);
const PHOSPHODIESTER_MAX_DIST_SQ = 2.5 * 2.5;

function isNucleicMolType(mt: number): boolean {
  return mt === MOLECULE_TYPE_RNA || mt === MOLECULE_TYPE_DNA || mt === MOLECULE_TYPE_PNA;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function arePhosphodiesterLinked(riPrev: number, riNext: number, hierarchy: any, conf: any): boolean {
  if (!conf?.xyzDefined) return false;
  const idx = hierarchy?.index;
  if (!idx?.findAtomsOnResidue) return false;
  const o3 = idx.findAtomsOnResidue(riPrev, O3_ATOM_IDS);
  const p = idx.findAtomsOnResidue(riNext, P_ATOM_IDS);
  if (o3 === -1 || p === -1) return false;
  const dx = conf.x[o3] - conf.x[p];
  const dy = conf.y[o3] - conf.y[p];
  const dz = conf.z[o3] - conf.z[p];
  return dx * dx + dy * dy + dz * dz <= PHOSPHODIESTER_MAX_DIST_SQ;
}

/**
 * Demote isolated / unlinked mononucleoside and cofactor ligands (e.g. `DHZ`,
 * `SAM`, `NAD`, `FAD`, `CMP`, `UMP`) that Mol*'s PDB parser misclassified as
 * `MoleculeType.RNA` / `DNA` back to `MoleculeType.Other` (`PolymerType.NA`)
 * so Mol*'s `nucleotide-ring` cartoon visual never draws phantom sticks or
 * filled base slabs through small-molecule ligands.
 *
 * Real DNA/RNA chains (>= 2 standard nucleotides, or modified nucleotides
 * covalently phosphodiester-linked into a strand) are preserved untouched.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function pruneFalseNucleicMonomers(trajSel: any, modelSel?: any): number {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const models = new Set<any>();
  const frames = trajSel?.data?.frames ?? trajSel?.cell?.obj?.data?.frames ?? trajSel?.frames;
  if (Array.isArray(frames)) {
    for (const f of frames) if (f) models.add(f);
  }
  const activeModel =
    modelSel?.data ?? modelSel?.cell?.obj?.data ?? (modelSel?.atomicHierarchy ? modelSel : null);
  if (activeModel) models.add(activeModel);
  if (models.size === 0 && trajSel?.atomicHierarchy) models.add(trajSel);
  if (models.size === 0) return 0;

  let demoted = 0;

  for (const m of models) {
    const hierarchy = m?.atomicHierarchy;
    const derivedRes = hierarchy?.derived?.residue;
    const moleculeType: Uint8Array | undefined = derivedRes?.moleculeType;
    const polymerType: Uint8Array | undefined = derivedRes?.polymerType;
    const traceElementIndex: Int32Array | undefined = derivedRes?.traceElementIndex;
    const chainSegs = hierarchy?.chainAtomSegments;
    const resSegs = hierarchy?.residueAtomSegments;
    const compIds = hierarchy?.atoms?.label_comp_id;
    const conf = m?.atomicConformation;

    if (!moleculeType || !polymerType || !chainSegs || !resSegs || !compIds) continue;

    const chainCount: number = chainSegs.count ?? 0;
    for (let cI = 0; cI < chainCount; cI += 1) {
      const aStart = chainSegs.offsets[cI];
      const aEnd = chainSegs.offsets[cI + 1];
      if (aEnd <= aStart) continue;

      const rStart = resSegs.index[aStart];
      const rEnd = resSegs.index[aEnd - 1];
      const chainResCount = rEnd - rStart + 1;

      const toDemote: number[] = [];
      for (let ri = rStart; ri <= rEnd; ri += 1) {
        if (!isNucleicMolType(moleculeType[ri])) continue;

        // A single-residue chain can never be a nucleic-acid polymer ribbon.
        if (chainResCount < 2) {
          toDemote.push(ri);
          continue;
        }

        const compId = String(compIds.value(resSegs.offsets[ri]) ?? '')
          .trim()
          .toUpperCase();
        if (STANDARD_NUCLEIC_COMP_IDS.has(compId)) continue;

        // Non-standard nucleoside in a multi-residue chain (e.g. two copies of
        // DHZ/SAM/NAD sharing a chain ID, or a modified nucleotide embedded in
        // a real RNA/DNA strand): keep only if covalently phosphodiester-linked
        // to an adjacent nucleic residue in the same chain.
        const linkedPrev =
          ri > rStart &&
          isNucleicMolType(moleculeType[ri - 1]) &&
          arePhosphodiesterLinked(ri - 1, ri, hierarchy, conf);
        const linkedNext =
          ri < rEnd &&
          isNucleicMolType(moleculeType[ri + 1]) &&
          arePhosphodiesterLinked(ri, ri + 1, hierarchy, conf);

        if (!linkedPrev && !linkedNext) {
          toDemote.push(ri);
        }
      }

      for (const ri of toDemote) {
        moleculeType[ri] = MOLECULE_TYPE_OTHER;
        polymerType[ri] = POLYMER_TYPE_NA;
        if (traceElementIndex) traceElementIndex[ri] = -1;
        demoted += 1;
      }
    }
  }

  return demoted;
}
