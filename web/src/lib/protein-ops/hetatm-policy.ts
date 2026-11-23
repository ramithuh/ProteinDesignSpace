// Unified Chemical Entity & HETATM Policy (Primitive 2)
// Classifies non-standard PDB/mmCIF HETATM records once into 4 disjoint classes,
// inspects organic ligands for missing hydrogens using quote-aware mmCIF parsing,
// and enforces one coherent protein-only structure policy across Genie 3 preview,
// compiler, static validation, runtime validation, and WebGPU worker execution.

export type HetatmEntityClass =
  | 'modified-amino-acid'
  | 'buffer-or-solvent'
  | 'monoatomic-ion'
  | 'organic-ligand';

/** Standard 20 L-amino acids (plus UNK) — polymer residues, not organic ligands. */
export const STANDARD_AA_RESNAMES: ReadonlySet<string> = new Set([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
  'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO',
  'SER', 'THR', 'TRP', 'TYR', 'VAL',
  'UNK',
]);

/** Standard DNA/RNA nucleotide residue names. */
export const NUCLEIC_ACID_RESNAMES: ReadonlySet<string> = new Set([
  'DA', 'DC', 'DG', 'DT', 'DU', 'DI',
  'A', 'C', 'G', 'U', 'T', 'I',
  'ADE', 'CYT', 'GUA', 'THY', 'URA',
]);

/**
 * Class 1: Post-translationally modified amino acids, selenomethionine, and peptide terminal caps.
 * Disjoint from buffer-or-solvent, monoatomic-ion, and organic-ligand.
 */
export const MODIFIED_AMINO_ACID_RESNAMES: ReadonlySet<string> = new Set([
  'MSE', 'M3L', 'SEP', 'TPO', 'PTR', 'CSO', 'KCX', 'MLY', 'HYP', 'PCA',
  'ACE', 'NME', 'NH2', 'FOR',
  'CSD', 'CME', 'OCS', 'LLP', 'MLZ', 'ALY', 'TYS', 'PYL', 'SEC', 'CGU',
  'ABA', 'AIB', 'NLE', 'ORN', 'CSX', 'CSS', 'SMC', 'SNC', 'SCH', 'CAS',
  'CAF', 'FME', 'MHO', 'OMT', 'HIC', 'NEP', 'HIP', 'HID', 'HIE', 'CYX',
  'CYM', 'ASH', 'GLH', 'LYN',
]);

/**
 * Class 2: Waters, crystallization buffers, cryoprotectants, and common polyatomic precipitant salts.
 * Disjoint from modified-amino-acid, monoatomic-ion, and organic-ligand.
 */
export const BUFFER_OR_SOLVENT_RESNAMES: ReadonlySet<string> = new Set([
  'HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3',
  'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PG4', 'PGE', '1PE', 'PE4', 'P6G',
  'DMS', 'DMSO', 'ACT', 'ACY', 'FMT', 'CIT', 'MPD', 'TRS', 'MES', 'EPE',
  'HEP', 'IPA', 'BME', 'DTT', 'TLA', 'UNX', 'FLC', 'NO3', 'CO3', 'SCN',
  'AZI', 'EOH', 'MOH', 'TAR', 'MLI', 'SUC', 'TRE', 'BOG', 'B3P', 'BTB',
  'IMD', 'NO2', 'POP', 'PYR', 'SIN', 'TAM',
]);

/**
 * Class 3: Monoatomic metal and halide ions (zero C–H / N–H / O–H protons by definition).
 * Disjoint from modified-amino-acid, buffer-or-solvent, and organic-ligand.
 */
export const MONOATOMIC_ION_RESNAMES: ReadonlySet<string> = new Set([
  'ZN', 'MG', 'CA', 'FE', 'FE2', 'NA', 'CL', 'K', 'MN', 'CU', 'CU1',
  'CO', '3CO', 'NI', 'CD', 'HG', 'BR', 'IOD', 'F', 'LI', 'RB', 'CS',
  'SR', 'BA', 'AL', 'GA', 'IN', 'TL', 'PB', 'PT', 'AU', 'AG', 'RU',
  'RH', 'PD', 'OS', 'IR', 'CR', 'V', 'MO', 'W', 'Y', 'LA', 'CE', 'EU',
  'GD', 'TB', 'YB', 'LU', 'SM', 'HO', 'ER', 'U', 'BE', 'SE', 'AS', 'SB',
  'BI', 'XE', 'KR', 'AR',
]);

/**
 * Classify a non-standard PDB/mmCIF residue / CCD code into one of 4 disjoint classes:
 * `modified-amino-acid` | `buffer-or-solvent` | `monoatomic-ion` | `organic-ligand`.
 */
export function classifyHetatmCompId(compId: string): HetatmEntityClass {
  const code = compId.trim().toUpperCase();
  if (MODIFIED_AMINO_ACID_RESNAMES.has(code)) return 'modified-amino-acid';
  if (BUFFER_OR_SOLVENT_RESNAMES.has(code)) return 'buffer-or-solvent';
  if (MONOATOMIC_ION_RESNAMES.has(code)) return 'monoatomic-ion';
  return 'organic-ligand';
}

/**
 * True if a HETATM compId is silently ignored by protein-only models (modified amino acids,
 * waters/buffers, or crystal Ca²⁺ ions on the Calmodulin EF-hand 1PRW motif benchmark).
 */
export function isSilentProteinOnlyHetatm(
  compId: string,
  opts?: { pdbId?: string; benchmarkId?: string },
): boolean {
  const code = compId.trim().toUpperCase();
  if (!code) return true;
  const cls = classifyHetatmCompId(code);
  if (cls === 'modified-amino-acid' || cls === 'buffer-or-solvent') return true;
  const pdbUpper = opts?.pdbId?.trim().toUpperCase() ?? '';
  if (code === 'CA' && (pdbUpper === '1PRW' || opts?.benchmarkId === 'rfd_1prw_efhand')) {
    return true;
  }
  return false;
}

/**
 * Quote-aware mmCIF data-row tokenizer respecting single- and double-quoted strings
 * (never naive whitespace splitting).
 */
export function tokenizeCifLine(line: string): string[] {
  const toks: string[] = [];
  const n = line.length;
  let i = 0;
  while (i < n) {
    while (i < n && (line[i] === ' ' || line[i] === '\t' || line[i] === '\r')) i++;
    if (i >= n) break;
    const ch = line[i];
    if (ch === '#') break;
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i++;
      const start = i;
      while (i < n) {
        if (
          line[i] === quote &&
          (i + 1 === n || line[i + 1] === ' ' || line[i + 1] === '\t' || line[i + 1] === '\r')
        ) {
          break;
        }
        i++;
      }
      toks.push(line.slice(start, i));
      if (i < n && line[i] === quote) i++;
    } else {
      const start = i;
      while (i < n && line[i] !== ' ' && line[i] !== '\t' && line[i] !== '\r') i++;
      toks.push(line.slice(start, i));
    }
  }
  return toks;
}

function isHydrogenAtom(elementRaw: string, atomNameRaw: string, compId: string): boolean {
  const elem = elementRaw.trim().toUpperCase();
  if (elem && elem !== '.' && elem !== '?') {
    return elem === 'H' || elem === 'D';
  }
  const atom = atomNameRaw.trim().toUpperCase();
  if (!atom) return false;
  // Avoid mistaking mercury (HG), helium (HE), hafnium (HF), holmium (HO) for hydrogen
  if (compId === 'HG' || compId === 'HE' || compId === 'HF' || compId === 'HO') return false;
  if (/^(?:HG|HE|HF|HO|HS)$/.test(atom)) return false;
  return /^(?:\d*H[A-Z0-9']*|\d*D[0-9']*)$/.test(atom);
}

export interface UnprotonatedOrganicLigandReport {
  hasUnprotonated: boolean;
  /** All distinct `organic-ligand` compIds found in HETATM records. */
  organicLigands: string[];
  /** Organic ligands with heavy atoms and 0 explicit H/D atoms. */
  unprotonatedLigands: string[];
  /** Organic ligands with at least 1 explicit H/D atom. */
  protonatedLigands: string[];
  /** Non-organic HETATMs ignored by protonation checks (ions, buffers, modified AAs). */
  ignoredNonOrganic: string[];
}

/**
 * Inspect a PDB or mmCIF structure for unprotonated `organic-ligand` HETATM records.
 * Strictly ignores `monoatomic-ion` (`ZN`, `CA`, `MG`, `FE2`, ...), `buffer-or-solvent`
 * (`HOH`, `SO4`, `GOL`, ...), and `modified-amino-acid` (`MSE`, `SEP`, ...) records,
 * and uses quote-aware mmCIF tokenization for `_atom_site.` loops.
 */
export function inspectUnprotonatedOrganicLigands(
  rawText: string,
): UnprotonatedOrganicLigandReport {
  const empty: UnprotonatedOrganicLigandReport = {
    hasUnprotonated: false,
    organicLigands: [],
    unprotonatedLigands: [],
    protonatedLigands: [],
    ignoredNonOrganic: [],
  };
  if (!rawText || !rawText.trim()) return empty;

  const organicStats = new Map<string, { heavy: number; hydrogen: number }>();
  const ignoredSet = new Set<string>();

  const recordAtom = (compRaw: string, elemRaw: string, atomNameRaw: string) => {
    const comp = compRaw.trim().toUpperCase();
    if (!comp || comp === '.' || comp === '?') return;
    if (STANDARD_AA_RESNAMES.has(comp) || NUCLEIC_ACID_RESNAMES.has(comp)) return;
    const cls = classifyHetatmCompId(comp);
    if (cls !== 'organic-ligand') {
      ignoredSet.add(comp);
      return;
    }
    let st = organicStats.get(comp);
    if (!st) {
      st = { heavy: 0, hydrogen: 0 };
      organicStats.set(comp, st);
    }
    if (isHydrogenAtom(elemRaw, atomNameRaw, comp)) {
      st.hydrogen++;
    } else {
      st.heavy++;
    }
  };

  const trimmed = rawText.trimStart();
  const isMmcif = trimmed.startsWith('data_') || rawText.includes('_atom_site.');

  if (isMmcif) {
    const lines = rawText.split(/\r?\n/);
    let inLoop = false;
    let inAtomSite = false;
    let cols: string[] = [];
    let iGroup = -1;
    let iCompAuth = -1;
    let iCompLabel = -1;
    let iElem = -1;
    let iAtomAuth = -1;
    let iAtomLabel = -1;
    let iModel = -1;
    let firstModel: string | null = null;
    let pendingRow: string[] = [];

    const flushRow = (row: string[]) => {
      const grp = iGroup >= 0 ? (row[iGroup] ?? '').toUpperCase() : 'ATOM';
      if (grp !== 'HETATM') return;
      if (iModel >= 0) {
        const m = row[iModel] ?? '1';
        if (firstModel === null) firstModel = m;
        else if (m !== firstModel) return;
      }
      const comp =
        (iCompLabel >= 0 && row[iCompLabel] && row[iCompLabel] !== '.' && row[iCompLabel] !== '?'
          ? row[iCompLabel]
          : iCompAuth >= 0
            ? row[iCompAuth]
            : '') ?? '';
      const elem = iElem >= 0 ? row[iElem] ?? '' : '';
      const atomName =
        (iAtomLabel >= 0 && row[iAtomLabel] && row[iAtomLabel] !== '.' && row[iAtomLabel] !== '?'
          ? row[iAtomLabel]
          : iAtomAuth >= 0
            ? row[iAtomAuth]
            : '') ?? '';
      recordAtom(comp, elem, atomName);
    };

    for (let li = 0; li < lines.length; li++) {
      const tr = lines[li].trim();
      if (!tr || tr.startsWith('#')) continue;
      if (tr === 'loop_') {
        if (inAtomSite) break; // only inspect first _atom_site loop
        inLoop = true;
        inAtomSite = false;
        cols = [];
        pendingRow = [];
        continue;
      }
      if (inLoop && tr.startsWith('_')) {
        const colName = tr.split(/\s+/)[0];
        if (colName.startsWith('_atom_site.')) {
          inAtomSite = true;
          cols.push(colName);
          const idx = cols.length - 1;
          if (colName === '_atom_site.group_PDB') iGroup = idx;
          else if (colName === '_atom_site.label_comp_id') iCompLabel = idx;
          else if (colName === '_atom_site.auth_comp_id') iCompAuth = idx;
          else if (colName === '_atom_site.type_symbol') iElem = idx;
          else if (colName === '_atom_site.label_atom_id') iAtomLabel = idx;
          else if (colName === '_atom_site.auth_atom_id') iAtomAuth = idx;
          else if (colName === '_atom_site.pdbx_PDB_model_num') iModel = idx;
        } else {
          inAtomSite = false;
        }
        continue;
      }
      if (inAtomSite) {
        if (
          tr.startsWith('_') ||
          tr.startsWith('data_') ||
          tr.startsWith('save_') ||
          tr === 'stop_'
        ) {
          break;
        }
        inLoop = false;
        const toks = tokenizeCifLine(tr);
        if (toks.length === 0) continue;
        if (pendingRow.length === 0 && toks.length >= cols.length) {
          flushRow(toks);
        } else {
          pendingRow.push(...toks);
          while (cols.length > 0 && pendingRow.length >= cols.length) {
            flushRow(pendingRow.slice(0, cols.length));
            pendingRow = pendingRow.slice(cols.length);
          }
        }
      } else {
        inLoop = false;
      }
    }
  } else {
    const lines = rawText.split(/\r?\n/);
    let seenModel = false;
    for (const line of lines) {
      if (line.startsWith('MODEL ')) {
        if (seenModel) break;
        seenModel = true;
        continue;
      }
      if (line.startsWith('ENDMDL')) {
        if (seenModel) break;
        continue;
      }
      if (!line.startsWith('HETATM')) continue;
      const atomName = line.slice(12, 16);
      const comp = line.slice(17, 20);
      const elem = line.length >= 78 ? line.slice(76, 78) : '';
      recordAtom(comp, elem, atomName);
    }
  }

  const organicLigands = [...organicStats.keys()];
  const unprotonatedLigands: string[] = [];
  const protonatedLigands: string[] = [];
  for (const [comp, st] of organicStats.entries()) {
    if (st.heavy > 0 && st.hydrogen === 0) {
      unprotonatedLigands.push(comp);
    } else if (st.hydrogen > 0) {
      protonatedLigands.push(comp);
    }
  }

  return {
    hasUnprotonated: unprotonatedLigands.length > 0,
    organicLigands,
    unprotonatedLigands,
    protonatedLigands,
    ignoredNonOrganic: [...ignoredSet],
  };
}

/**
 * Convenience predicate returning true ONLY when the PDB/mmCIF structure contains
 * at least one `organic-ligand` HETATM with heavy atoms and 0 explicit hydrogens.
 */
export function hasUnprotonatedOrganicLigands(rawText: string): boolean {
  return inspectUnprotonatedOrganicLigands(rawText).hasUnprotonated;
}

export interface ProteinOnlyPolicyInput {
  modelName?: string;
  benchmark?: { id?: string; shortLabel?: string; supportsGenie3?: boolean } | null;
  pdbId?: string;
  /** Protein Cα residue count (omit if structure is not yet loaded/parsed). */
  proteinResidueCount?: number;
  ligandNames?: readonly string[];
  nucleicAcidChains?: readonly string[];
  /** Optional fallback when diagnostics only reported boolean flags without names. */
  hasLigands?: boolean;
  hasNucleicAcids?: boolean;
}

export interface ProteinOnlyPolicyAssessment {
  action: 'hard-block' | 'warn' | 'allow';
  reason?: 'unsupported-benchmark' | 'insufficient-protein' | 'non-protein-entities';
  message?: string;
  significantLigands: string[];
  nucleicAcidChains: string[];
}

/**
 * Single coherent protein-only structure policy shared by Genie 3 across
 * `bio-capabilities.ts` (Layer 1 & Layer 2), `conditioning.ts`, and `genie3.worker.ts`:
 * - Hard-block ONLY explicit incompatible all-atom benchmarks (`!benchmark.supportsGenie3`)
 *   OR structures with `< 3` protein Cα residues (ligand/DNA-only crops).
 * - Silent pass when HETATMs are strictly `modified-amino-acid` or `buffer-or-solvent`
 *   (plus crystal `CA` on `1PRW`).
 * - Non-blocking warning (NEVER a runtime throw) when `>= 3` protein Cα residues coexist
 *   with uncropped ions/ligands/nucleic chains — Genie 3 ignores non-protein records cleanly.
 */
export function evaluateProteinOnlyStructurePolicy(
  input: ProteinOnlyPolicyInput,
): ProteinOnlyPolicyAssessment {
  const modelName = input.modelName ?? 'Genie 3';
  const benchmark = input.benchmark ?? undefined;
  const pdbId = input.pdbId?.trim().toUpperCase() || undefined;
  const rawLigands = input.ligandNames ?? [];
  const nucleicAcidChains = [
    ...new Set((input.nucleicAcidChains ?? []).map((c) => c.trim()).filter(Boolean)),
  ];

  const significantSet = new Set<string>();
  for (const lig of rawLigands) {
    const code = lig.trim().toUpperCase();
    if (!code) continue;
    if (!isSilentProteinOnlyHetatm(code, { pdbId, benchmarkId: benchmark?.id })) {
      significantSet.add(code);
    }
  }
  const significantLigands = [...significantSet];

  // 1. Hard-block explicit incompatible all-atom enzyme/ligand/DNA benchmarks
  if (benchmark && benchmark.supportsGenie3 === false) {
    return {
      action: 'hard-block',
      reason: 'unsupported-benchmark',
      message: `${benchmark.shortLabel ?? benchmark.id ?? 'This benchmark'} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than ${modelName}.`,
      significantLigands,
      nucleicAcidChains,
    };
  }

  // 2. Hard-block structures with < 3 protein Cα residues (e.g. ligand/DNA-only crops)
  if (input.proteinResidueCount !== undefined && input.proteinResidueCount < 3) {
    return {
      action: 'hard-block',
      reason: 'insufficient-protein',
      message: `${modelName} is a protein-only Cα backbone diffusion model and requires at least 3 protein Cα residues (received ${input.proteinResidueCount}). Use RFD4-Proteina for ligand/DNA-only conditioning.`,
      significantLigands,
      nucleicAcidChains,
    };
  }

  // 3. Check for non-silent ions/organic-ligands/nucleic chains coexisting with >= 3 protein Cα residues
  const hasUnnamedLigands = rawLigands.length === 0 && Boolean(input.hasLigands);
  const hasAnyNucleic = nucleicAcidChains.length > 0 || Boolean(input.hasNucleicAcids);

  if (significantLigands.length === 0 && !hasUnnamedLigands && !hasAnyNucleic) {
    return {
      action: 'allow',
      significantLigands: [],
      nucleicAcidChains: [],
    };
  }

  const extras: string[] = [];
  if (significantLigands.length > 0) {
    extras.push(`non-protein HETATMs (${significantLigands.join(', ')})`);
  } else if (hasUnnamedLigands) {
    extras.push('non-protein ligands');
  }
  if (nucleicAcidChains.length > 0) {
    extras.push(`nucleic acid chains (${nucleicAcidChains.join(', ')})`);
  } else if (hasAnyNucleic) {
    extras.push('nucleic acid chains');
  }

  return {
    action: 'warn',
    reason: 'non-protein-entities',
    message: `Upstream structure contains ${extras.join(' and ')}; ${modelName} (protein-only Cα diffusion) ignores non-protein records cleanly and conditions on protein Cα coordinates only. Crop to protein chains or use RFD4-Proteina for all-atom ligand/DNA conditioning.`,
    significantLigands,
    nucleicAcidChains,
  };
}
