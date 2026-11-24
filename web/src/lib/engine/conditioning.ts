// Model-agnostic conditioning compiler:
// 1. Builds a canonical CanonicalConditionIntent from block parameters.
// 2. Compiles canonical intent -> RFD4-Proteina `sel(...)` / `C_*` dict (decoupled from scheduler).
// 3. Compiles canonical intent + template structure -> Genie 3 `createFeaturesFromSpec(...)` batch
//    for both protein motif scaffolding and PPI binder design!

import {
  AA1_TO_AA3,
  AA3_TO_AA1,
  parseStructureText,
  projectChainIdsToPdb,
  type ParsedStructure,
  type SourceResidueId,
} from '$lib/protein-ops/pdb';
import { evaluateProteinOnlyStructurePolicy } from '$lib/protein-ops/hetatm-policy';
import { getBenchmarkById } from './benchmarks';
import {
  isLocalFoldTemplateCapable,
  localFoldTemplateFamily,
  resolveLocalFoldTemplateSource,
  type LocalFoldTemplateFamily,
  type ResolvedLocalFoldTemplateSource,
} from './localfold-template';
import type {
  CanonicalConditionIntent,
  ConditionSpecArtifact,
  MotifAtomTarget,
  MsaMode,
  StructureArtifact,
} from './types';

export interface ResidueRangeToken {
  chain: string;
  start: number;
  end: number;
  startInsCode?: string;
  endInsCode?: string;
}

/**
 * Explicit correspondence linking a selected motif/target residue in a compiled conditioning spec
 * back to its original source structure residue (preserving multi-char chains & insertion codes).
 */
export interface SelectedMotifSourceResidue {
  /** 0-based index in the compiled output sequence/coords (or 0-based selection order for standalone resolution). */
  outputIndex: number;
  /** 0-based index in the parsed template structure (`parsed.sourceResidues[templateIndex]`). */
  templateIndex: number;
  /** Original author chain ID (`auth_asym_id` / PDB col 22). */
  authChainId: string;
  /** Optional mmCIF `label_asym_id` when present in template. */
  labelChainId?: string;
  /** Original author residue sequence number (`auth_seq_id` / PDB cols 23-26). */
  resNum: number;
  /** Optional mmCIF `label_seq_id` when present and numeric. */
  labelSeqId?: number;
  /** Original insertion code (`''` when absent, e.g. `'A'`, `'B'`). */
  insCode: string;
  /** Original 3-letter residue name (e.g. `'ALA'`, `'HIS'`, `'MSE'`). */
  resName: string;
  /** 1-letter residue code. */
  aa1: string;
  /** Deterministic bijective 1-char PDB chain ID assigned for PDB projection. */
  projectedChainId: string;
}

const GENIE3_RESTYPE_1 = 'ARNDCQEGHILKMFPSTWYV';
const WATER_COMP_IDS = new Set(['HOH', 'DOD', 'WAT', 'H2O']);
const DNA_COMP_IDS = new Set(['DA', 'DC', 'DG', 'DT', 'DI', 'DU', 'T', 'THY']);
const RNA_COMP_IDS = new Set(['A', 'C', 'G', 'U', 'I', 'ADE', 'CYT', 'GUA', 'URA']);

/**
 * Common crystallographic buffers, precipitants, cryoprotectants, and waters that should NOT
 * be silently auto-injected as conditioned RFD4 ligands unless explicitly named by the user
 * in `contig` / `cropSegments` / `hbondSelection`.
 * Preserves biological metals (ZN, MG, MN, CA, FE, FE2, CU, CO, NI) and cofactors/substrates.
 */
export const CRYSTALLIZATION_BUFFERS: ReadonlySet<string> = new Set([
  'SO4',
  'PO4',
  'GOL',
  'EDO',
  'PEG',
  'ACT',
  'DMS',
  'CL',
  'NA',
  'FMT',
  'MPD',
  'TRS',
  'MES',
  'EPE',
  'CIT',
  'TLA',
  'BME',
  'DTT',
  'ACY',
  'IPA',
  'PGE',
  'PG4',
  '1PE',
  'P6G',
  'ACE',
  'NME',
  'NH2',
  'FOR',
  'HOH',
  'DOD',
  'WAT',
  'H2O',
]);

export function isCrystallizationBuffer(compId: string): boolean {
  return CRYSTALLIZATION_BUFFERS.has(compId.trim().toUpperCase());
}

export interface Rfd4PolymerResidueEntry {
  authChain: string;
  authSeqId: number;
  insCode: string;
  labelChain: string;
  labelSeqId: number;
  compId: string;
  isNucleic: boolean;
  /** 0-based polymer residue order in the template structure. */
  templateIndex: number;
}

export interface Rfd4PolymerChainInfo {
  authChain: string;
  labelChain: string;
  minLabelSeq: number;
  maxLabelSeq: number;
  minAuthSeq: number;
  maxAuthSeq: number;
  isNucleic: boolean;
  residues: Rfd4PolymerResidueEntry[];
}

export interface Rfd4LigandEntry {
  labelChain: string;
  authChain: string;
  compId: string;
  /** Non-polymer residue ID in RFD4 (`parseCifToAtomArray` uses `auth_seq_id` for non-polymers). */
  resId: number;
}

export interface Rfd4TemplateContext {
  /** Polymer residues keyed by `${authChain}:${authSeqId}`. */
  byAuthRes: Map<string, Rfd4PolymerResidueEntry>;
  /** Polymer residues keyed by `${labelChain}:${labelSeqId}`. */
  byLabelRes: Map<string, Rfd4PolymerResidueEntry>;
  /** Polymer chains keyed by `authChain`. */
  polymerByAuthChain: Map<string, Rfd4PolymerChainInfo>;
  /** Polymer chains keyed by `labelChain`. */
  polymerByLabelChain: Map<string, Rfd4PolymerChainInfo>;
  /** Distinct non-polymer ligands/ions (waters excluded). */
  ligands: Rfd4LigandEntry[];
  /** Distinct nucleic acid polymer chains (DNA/RNA). */
  nucleicChains: Rfd4PolymerChainInfo[];
  remapPolymerChain: (chain: string) => string;
  remapPolymerRange: (
    chain: string,
    start: number,
    end: number,
    startInsCode?: string,
    endInsCode?: string,
  ) => { labelChain: string; start: number; end: number };
  remapContig: (contig: string) => string;
  remapResidueList: (tokens: string[]) => string[];
  remapHbondSelection: (selStr: string) => string;
  synthesizeBinderContig: (lockChain: string | undefined, binderLength: number) => string | null;
}

function tokenizeCifRow(line: string): string[] {
  const out: string[] = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    while (i < n && (line[i] === ' ' || line[i] === '\t')) i++;
    if (i >= n) break;
    const ch = line[i];
    if (ch === "'" || ch === '"') {
      const q = ch;
      i++;
      let tok = '';
      while (i < n && !(line[i] === q && (i + 1 >= n || line[i + 1] === ' ' || line[i + 1] === '\t'))) {
        tok += line[i++];
      }
      if (i < n && line[i] === q) i++;
      out.push(tok);
    } else {
      let tok = '';
      while (i < n && line[i] !== ' ' && line[i] !== '\t') tok += line[i++];
      out.push(tok);
    }
  }
  return out;
}

function nextUnusedChainId(used: Set<string>): string {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (let i = 0; i < letters.length; i++) {
    if (!used.has(letters[i])) {
      used.add(letters[i]);
      return letters[i];
    }
  }
  for (let i = 0; i < letters.length; i++) {
    for (let j = 0; j < letters.length; j++) {
      const cid = letters[i] + letters[j];
      if (!used.has(cid)) {
        used.add(cid);
        return cid;
      }
    }
  }
  const fallback = `L${used.size + 1}`;
  used.add(fallback);
  return fallback;
}

/**
 * Convert raw PDB text into mmCIF text for RFD4-Proteina, splitting non-polymer `HETATM`
 * ligands/ions onto distinct `label_asym_id` and `label_entity_id` chains while preserving
 * `auth_asym_id`/`auth_seq_id` and `_entity`/`_entity_poly` metadata.
 * Prevents RFD4's `parseCifToAtomArray` from misclassifying PDB HETATM ligands as `POLYPEPTIDE_L`.
 */
export function pdbTextToSplitCifText(pdbText: string): string {
  interface RawAtomLine {
    group: 'ATOM' | 'HETATM';
    atomName: string;
    altId: string;
    resName: string;
    authChain: string;
    authSeq: string;
    insCode: string;
    x: string;
    y: string;
    z: string;
    occ: string;
    bfac: string;
    elem: string;
    isPolymer: boolean;
  }

  const lines = pdbText.split(/\r?\n/);
  const parsed: RawAtomLine[] = [];
  const polymerChains = new Set<string>();
  const reconstructedOKeys = new Set<string>();
  let seenAtom = false;

  for (const line of lines) {
    if (line.startsWith('REMARK 250 RECONSTRUCTED_O ')) {
      const keys = line.slice('REMARK 250 RECONSTRUCTED_O '.length).trim().split(/\s+/).filter(Boolean);
      for (const k of keys) reconstructedOKeys.add(k);
      continue;
    }
    if (line.startsWith('ENDMDL') && seenAtom) break;
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) continue;
    seenAtom = true;
    const group = line.startsWith('HETATM') ? 'HETATM' : 'ATOM';
    const atomName = line.slice(12, 16).trim() || 'CA';
    const altId = line.slice(16, 17).trim() || '.';
    const resName = (line.slice(17, 20).trim() || 'ALA').toUpperCase();
    if (WATER_COMP_IDS.has(resName)) continue;
    const authChain = line.slice(21, 22).trim() || 'A';
    const authSeq = line.slice(22, 26).trim() || '1';
    const rawIns = line.slice(26, 27).trim();
    const insCode = rawIns || '.';
    // Never emit parser-reconstructed O atoms into mmCIF as if they were observed coordinates
    if (atomName === 'O' && reconstructedOKeys.has(`${authChain}:${authSeq}:${rawIns}`)) continue;
    const x = line.slice(30, 38).trim() || '0.0';
    const y = line.slice(38, 46).trim() || '0.0';
    const z = line.slice(46, 54).trim() || '0.0';
    const occ = line.slice(54, 60).trim() || '1.00';
    const bfac = line.slice(60, 66).trim() || '0.00';
    let elem = line.slice(76, 78).trim();
    if (!elem) elem = atomName.replace(/[0-9']/g, '').slice(0, 1) || 'C';
    const isPolymer = group === 'ATOM' || resName === 'MSE';
    if (isPolymer) polymerChains.add(authChain);
    parsed.push({
      group,
      atomName,
      altId,
      resName,
      authChain,
      authSeq,
      insCode,
      x,
      y,
      z,
      occ,
      bfac,
      elem: elem.toUpperCase(),
      isPolymer,
    });
  }

  const usedChains = new Set<string>(polymerChains);
  const entityRows: Array<{ id: number; type: 'polymer' | 'non-polymer' }> = [];
  const entityPolyRows: Array<{ entityId: number; polyType: string; strandId: string }> = [];
  let nextEntityId = 1;

  // Assign entity IDs and sequential label_seq_id per polymer chain
  const polyEntityByChain = new Map<string, number>();
  const polyResSeqCounters = new Map<string, Map<string, number>>();
  const polyFirstCompByChain = new Map<string, string>();
  for (const a of parsed) {
    if (!a.isPolymer) continue;
    if (!polyFirstCompByChain.has(a.authChain)) polyFirstCompByChain.set(a.authChain, a.resName);
    let resMap = polyResSeqCounters.get(a.authChain);
    if (!resMap) {
      resMap = new Map<string, number>();
      polyResSeqCounters.set(a.authChain, resMap);
    }
    const rKey = `${a.authSeq}:${a.insCode}`;
    if (!resMap.has(rKey)) resMap.set(rKey, resMap.size + 1);
  }
  for (const ch of polymerChains) {
    const eid = nextEntityId++;
    polyEntityByChain.set(ch, eid);
    entityRows.push({ id: eid, type: 'polymer' });
    const firstComp = polyFirstCompByChain.get(ch) ?? 'ALA';
    const polyType = DNA_COMP_IDS.has(firstComp)
      ? "'polydeoxyribonucleotide'"
      : RNA_COMP_IDS.has(firstComp)
        ? "'polyribonucleotide'"
        : "'polypeptide(L)'";
    entityPolyRows.push({ entityId: eid, polyType, strandId: ch });
  }

  // Assign distinct label_asym_id and non-polymer label_entity_id per HETATM ligand/ion instance
  const hetChainMap = new Map<string, string>();
  const hetEntityByComp = new Map<string, number>();
  const atomSiteRows: string[] = [];
  let serial = 0;

  for (const a of parsed) {
    serial += 1;
    const qAtom = a.atomName.includes("'") ? `"${a.atomName}"` : a.atomName;
    if (a.isPolymer) {
      const eid = polyEntityByChain.get(a.authChain) ?? 1;
      const labelSeq = polyResSeqCounters.get(a.authChain)?.get(`${a.authSeq}:${a.insCode}`) ?? 1;
      atomSiteRows.push(
        `${a.group} ${serial} ${a.elem} ${qAtom} ${a.altId} ${a.resName} ${a.authChain} ${eid} ${labelSeq} ${a.insCode} ${a.x} ${a.y} ${a.z} ${a.occ} ${a.bfac} 0 ${a.authSeq} ${a.resName} ${a.authChain} ${qAtom} 1`,
      );
    } else {
      const instKey = `${a.authChain}:${a.resName}:${a.authSeq}:${a.insCode}`;
      let labelChain = hetChainMap.get(instKey);
      if (!labelChain) {
        labelChain = nextUnusedChainId(usedChains);
        hetChainMap.set(instKey, labelChain);
      }
      let eid = hetEntityByComp.get(a.resName);
      if (eid === undefined) {
        eid = nextEntityId++;
        hetEntityByComp.set(a.resName, eid);
        entityRows.push({ id: eid, type: 'non-polymer' });
      }
      atomSiteRows.push(
        `HETATM ${serial} ${a.elem} ${qAtom} ${a.altId} ${a.resName} ${labelChain} ${eid} . ${a.insCode} ${a.x} ${a.y} ${a.z} ${a.occ} ${a.bfac} 0 ${a.authSeq} ${a.resName} ${a.authChain} ${qAtom} 1`,
      );
    }
  }

  const out: string[] = ['data_pdb'];
  if (entityRows.length > 0) {
    out.push('loop_', '_entity.id', '_entity.type');
    for (const e of entityRows) out.push(`${e.id} ${e.type}`);
    out.push('#');
  }
  if (entityPolyRows.length > 0) {
    out.push('loop_', '_entity_poly.entity_id', '_entity_poly.type', '_entity_poly.pdbx_strand_id');
    for (const ep of entityPolyRows) out.push(`${ep.entityId} ${ep.polyType} ${ep.strandId}`);
    out.push('#');
  }
  out.push(
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.pdbx_formal_charge',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
    ...atomSiteRows,
    '#',
  );
  return out.join('\n') + '\n';
}

/**
 * Inspect an input mmCIF (or PDB) template and build an exact `auth_asym_id`/`auth_seq_id`
 * -> `label_asym_id`/`label_seq_id` (+ non-polymer ligand/ion) remapper for RFD4-Proteina.
 */
export function extractRfd4TemplateContext(rawStructureText: string): Rfd4TemplateContext {
  const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(rawStructureText);
  const cifText = isCif ? rawStructureText : pdbTextToSplitCifText(rawStructureText);

  const byAuthRes = new Map<string, Rfd4PolymerResidueEntry>();
  const byLabelRes = new Map<string, Rfd4PolymerResidueEntry>();
  const polymerByAuthChain = new Map<string, Rfd4PolymerChainInfo>();
  const polymerByLabelChain = new Map<string, Rfd4PolymerChainInfo>();
  const ligands: Rfd4LigandEntry[] = [];
  const seenLigandKeys = new Set<string>();

  const lines = cifText.split(/\r?\n/);
  const cols: string[] = [];
  let inAtomSite = false;
  let firstModelNum: string | null = null;
  const hasObservedCaByLabelKey = new Set<string>();

  for (const raw of lines) {
    const tr = raw.trim();
    if (tr.startsWith('_atom_site.')) {
      inAtomSite = true;
      cols.push(tr.slice('_atom_site.'.length));
      continue;
    }
    if (!inAtomSite) continue;
    if (tr === '' || tr.startsWith('#') || tr.startsWith('loop_') || tr.startsWith('_')) {
      inAtomSite = false;
      continue;
    }
    const toks = tokenizeCifRow(tr);
    if (toks.length < cols.length) continue;

    const get = (name: string) => {
      const idx = cols.indexOf(name);
      return idx >= 0 ? toks[idx] : '';
    };

    const modelNum = get('pdbx_PDB_model_num');
    if (modelNum && modelNum !== '.' && modelNum !== '?') {
      if (firstModelNum === null) firstModelNum = modelNum;
      else if (modelNum !== firstModelNum) continue;
    }
    const altId = get('label_alt_id');
    if (altId && altId !== '.' && altId !== '?' && altId !== 'A' && altId !== '1') continue;

    const grp = (get('group_PDB') || 'ATOM').toUpperCase();
    const comp = (get('label_comp_id') || get('auth_comp_id') || '').toUpperCase();
    if (!comp || WATER_COMP_IDS.has(comp)) continue;

    const labelChain = get('label_asym_id') || get('auth_asym_id') || 'A';
    const authChain = get('auth_asym_id') || labelChain;
    const labelSeqRaw = get('label_seq_id');
    const authSeqRaw = get('auth_seq_id') || labelSeqRaw;
    const rawIns = get('pdbx_PDB_ins_code');
    const insCode = rawIns && rawIns !== '.' && rawIns !== '?' ? rawIns : '';
    const labelSeq = parseInt(labelSeqRaw, 10);
    const authSeq = parseInt(authSeqRaw, 10);

    const isNucleic = DNA_COMP_IDS.has(comp) || RNA_COMP_IDS.has(comp);
    const isPolymer =
      (grp === 'ATOM' || comp === 'MSE' || isNucleic) &&
      labelSeqRaw !== '.' &&
      labelSeqRaw !== '?' &&
      Number.isFinite(labelSeq);

    if (isPolymer) {
      const resolvedAuthSeq = Number.isFinite(authSeq) ? authSeq : labelSeq;
      const authKeyWithIns = `${authChain}:${resolvedAuthSeq}:${insCode}`;
      const authKeyPlain = `${authChain}:${resolvedAuthSeq}`;
      const labelKey = `${labelChain}:${labelSeq}`;
      if (!isNucleic) {
        const rawAtom = get('label_atom_id') || get('auth_atom_id');
        if (rawAtom === 'CA') {
          const hasCoordCols =
            cols.includes('Cartn_x') && cols.includes('Cartn_y') && cols.includes('Cartn_z');
          if (
            !hasCoordCols ||
            (Number.isFinite(parseFloat(get('Cartn_x'))) &&
              Number.isFinite(parseFloat(get('Cartn_y'))) &&
              Number.isFinite(parseFloat(get('Cartn_z'))))
          ) {
            hasObservedCaByLabelKey.add(labelKey);
          }
        }
      }
      if (!byLabelRes.has(labelKey)) {
        const entry: Rfd4PolymerResidueEntry = {
          authChain,
          authSeqId: resolvedAuthSeq,
          insCode,
          labelChain,
          labelSeqId: labelSeq,
          compId: comp,
          isNucleic,
          templateIndex: -1,
        };
        byLabelRes.set(labelKey, entry);
        if (!byAuthRes.has(authKeyWithIns)) byAuthRes.set(authKeyWithIns, entry);
        if (insCode === '' && !byAuthRes.has(authKeyPlain)) byAuthRes.set(authKeyPlain, entry);

        let authInfo = polymerByAuthChain.get(authChain);
        if (!authInfo) {
          authInfo = {
            authChain,
            labelChain,
            minLabelSeq: labelSeq,
            maxLabelSeq: labelSeq,
            minAuthSeq: resolvedAuthSeq,
            maxAuthSeq: resolvedAuthSeq,
            isNucleic,
            residues: [],
          };
          polymerByAuthChain.set(authChain, authInfo);
        }
        authInfo.minLabelSeq = Math.min(authInfo.minLabelSeq, labelSeq);
        authInfo.maxLabelSeq = Math.max(authInfo.maxLabelSeq, labelSeq);
        authInfo.minAuthSeq = Math.min(authInfo.minAuthSeq, resolvedAuthSeq);
        authInfo.maxAuthSeq = Math.max(authInfo.maxAuthSeq, resolvedAuthSeq);
        authInfo.residues.push(entry);

        let labelInfo = polymerByLabelChain.get(labelChain);
        if (!labelInfo) {
          labelInfo = {
            authChain,
            labelChain,
            minLabelSeq: labelSeq,
            maxLabelSeq: labelSeq,
            minAuthSeq: resolvedAuthSeq,
            maxAuthSeq: resolvedAuthSeq,
            isNucleic,
            residues: [],
          };
          polymerByLabelChain.set(labelChain, labelInfo);
        }
        labelInfo.minLabelSeq = Math.min(labelInfo.minLabelSeq, labelSeq);
        labelInfo.maxLabelSeq = Math.max(labelInfo.maxLabelSeq, labelSeq);
        labelInfo.minAuthSeq = Math.min(labelInfo.minAuthSeq, resolvedAuthSeq);
        labelInfo.maxAuthSeq = Math.max(labelInfo.maxAuthSeq, resolvedAuthSeq);
        labelInfo.residues.push(entry);
      }
    } else {
      const resId = Number.isFinite(authSeq) ? authSeq : Number.isFinite(labelSeq) ? labelSeq : 1;
      const ligKey = `${labelChain}:${resId}:${comp}`;
      if (!seenLigandKeys.has(ligKey)) {
        seenLigandKeys.add(ligKey);
        ligands.push({ labelChain, authChain, compId: comp, resId });
      }
    }
  }

  // Align protein templateIndex 1:1 with `parseStructureText` (non-nucleic residues with observed finite CA)
  let nextProteinIdx = 0;
  for (const [labelKey, entry] of byLabelRes.entries()) {
    if (!entry.isNucleic && hasObservedCaByLabelKey.has(labelKey)) {
      entry.templateIndex = nextProteinIdx++;
    } else {
      entry.templateIndex = -1;
    }
  }

  const nucleicChains = [...polymerByLabelChain.values()].filter((c) => c.isNucleic);

  const lookupAuthChain = (chain: string): Rfd4PolymerChainInfo | undefined => {
    const t = chain.trim();
    return polymerByAuthChain.get(t) ?? polymerByAuthChain.get(t.toUpperCase());
  };
  const lookupLabelChain = (chain: string): Rfd4PolymerChainInfo | undefined => {
    const t = chain.trim();
    return polymerByLabelChain.get(t) ?? polymerByLabelChain.get(t.toUpperCase());
  };

  const remapPolymerChain = (chain: string): string => {
    const t = chain.trim();
    const byAuth = lookupAuthChain(t);
    if (byAuth) return byAuth.labelChain;
    const byLabel = lookupLabelChain(t);
    if (byLabel) return byLabel.labelChain;
    return t;
  };

  const remapPolymerRange = (
    chain: string,
    start: number,
    end: number,
    startInsCode?: string,
    endInsCode?: string,
  ): { labelChain: string; start: number; end: number } => {
    const t = chain.trim();
    const hasExplicitIns = Boolean(startInsCode || endInsCode);
    const authInfo = lookupAuthChain(t);
    if (authInfo) {
      const hits = authInfo.residues.filter((r) =>
        matchesResNumInsBounds(r.authSeqId, r.insCode, start, end, startInsCode, endInsCode),
      );
      if (hits.length > 0) {
        let minL = Infinity;
        let maxL = -Infinity;
        for (const h of hits) {
          if (h.labelSeqId < minL) minL = h.labelSeqId;
          if (h.labelSeqId > maxL) maxL = h.labelSeqId;
        }
        return { labelChain: authInfo.labelChain, start: minL, end: maxL };
      }
      // Never silently fall back to label_seq_id when an explicit author chain exists and the
      // requested author residue number missed (e.g. A1 on A101-A103 must NOT silently lock A101!).
      return { labelChain: authInfo.labelChain, start, end };
    }
    if (!hasExplicitIns) {
      const labelInfo = lookupLabelChain(t);
      if (labelInfo) {
        const hits = labelInfo.residues.filter((r) => r.labelSeqId >= start && r.labelSeqId <= end);
        if (hits.length > 0) {
          let minL = Infinity;
          let maxL = -Infinity;
          for (const h of hits) {
            if (h.labelSeqId < minL) minL = h.labelSeqId;
            if (h.labelSeqId > maxL) maxL = h.labelSeqId;
          }
          return { labelChain: labelInfo.labelChain, start: minL, end: maxL };
        }
      }
    }
    return { labelChain: remapPolymerChain(t), start, end };
  };

  const hasExactAuthResidueMatch = (
    chain: string,
    start: number,
    end: number,
    startInsCode?: string,
    endInsCode?: string,
  ): boolean => {
    const authInfo = lookupAuthChain(chain.trim());
    if (!authInfo) return false;
    return authInfo.residues.some((r) =>
      matchesResNumInsBounds(r.authSeqId, r.insCode, start, end, startInsCode, endInsCode),
    );
  };

  const remapResidueList = (tokens: string[]): string[] => {
    const out: string[] = [];
    for (const r of parseResidueRangeTokens(tokens.join(', '))) {
      const hasExplicitIns = Boolean(r.startInsCode || r.endInsCode);
      if (hasExplicitIns && !hasExactAuthResidueMatch(r.chain, r.start, r.end, r.startInsCode, r.endInsCode)) {
        // Preserve absent insertion code token verbatim so it never silently aliases an unrelated label_seq_id
        const startTok = `${r.start}${r.startInsCode ?? ''}`;
        const endTok = `${r.end}${r.endInsCode ?? ''}`;
        out.push(startTok === endTok ? `${r.chain}${startTok}` : `${r.chain}${startTok}-${endTok}`);
        continue;
      }
      const mapped = remapPolymerRange(r.chain, r.start, r.end, r.startInsCode, r.endInsCode);
      out.push(
        mapped.start === mapped.end
          ? `${mapped.labelChain}${mapped.start}`
          : `${mapped.labelChain}${mapped.start}-${mapped.end}`,
      );
    }
    return out;
  };

  const remapContig = (contig: string): string => {
    const parts = contig
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    const out: string[] = [];
    for (const tok of parts) {
      if (tok === '/0') {
        out.push('/0');
        continue;
      }
      // Bare expandable segment (e.g. "70-70", "5-20")
      if (/^\d+(?:-\d+)?[PRDN]?$/i.test(tok)) {
        out.push(tok);
        continue;
      }
      // Named expandable segment (e.g. "h1:16-16P" or "h1:5-5") — check polymer match first so "AA:10D" / "AA:1-5" is not shadowed
      if (/^[A-Za-z0-9_]+:\d+-\d+[PRDN]?$/i.test(tok)) {
        const prefix = tok.slice(0, tok.indexOf(':'));
        if (!lookupAuthChain(prefix) && !lookupLabelChain(prefix)) {
          out.push(tok);
          continue;
        }
      }
      // Explicit ligand selector like "A:SAM" or "A:ZN"
      const chainLig = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(tok);
      if (chainLig && !/^\d+[A-Za-z]?$/.test(chainLig[2])) {
        const ch = chainLig[1];
        const chUpper = ch.toUpperCase();
        const comp = chainLig[2].toUpperCase();
        const matches = ligands.filter(
          (l) =>
            l.compId === comp &&
            (l.authChain === ch ||
              l.labelChain === ch ||
              l.authChain.toUpperCase() === chUpper ||
              l.labelChain.toUpperCase() === chUpper),
        );
        const picked = matches.length > 0 ? matches : ligands.filter((l) => l.compId === comp);
        if (picked.length > 0) {
          for (let i = 0; i < picked.length; i++) {
            if (i > 0) out.push('/0');
            out.push(`${picked[i].labelChain}${picked[i].resId}`);
          }
          continue;
        }
      }
      // Preserved residue/range token like "A17-132", "A102", "A10A", "AA1-5", "AA:1-5", "AA:10D"
      const parsedTokens = /^[A-Za-z]/.test(tok) || tok.includes(':') ? parseResidueRangeTokens(tok) : [];
      if (parsedTokens.length === 1) {
        const pt = parsedTokens[0];
        const ch = pt.chain;
        const start = pt.start;
        const end = pt.end;
        const hasExplicitIns = Boolean(pt.startInsCode || pt.endInsCode);
        // Check polymer match first (explicit insertion codes must ONLY match author numbering, never label_seq_id)
        const authPoly = lookupAuthChain(ch);
        const labelPoly = lookupLabelChain(ch);
        const hasAuthHit = authPoly?.residues.some((r) =>
          matchesResNumInsBounds(r.authSeqId, r.insCode, start, end, pt.startInsCode, pt.endInsCode),
        );
        const hasLabelHit = !hasExplicitIns && labelPoly?.residues.some((r) => r.labelSeqId >= start && r.labelSeqId <= end);
        if (hasAuthHit || hasLabelHit) {
          const m = remapPolymerRange(ch, start, end, pt.startInsCode, pt.endInsCode);
          out.push(m.start === m.end ? `${m.labelChain}${m.start}` : `${m.labelChain}${m.start}-${m.end}`);
          continue;
        }
        if (hasExplicitIns) {
          out.push(tok);
          continue;
        }
        // Check ligand match by authChain/labelChain + resId
        const chUpper = ch.toUpperCase();
        const ligHit = ligands.find(
          (l) =>
            (l.labelChain === ch ||
              l.authChain === ch ||
              l.labelChain.toUpperCase() === chUpper ||
              l.authChain.toUpperCase() === chUpper) &&
            l.resId >= start &&
            l.resId <= end,
        );
        if (ligHit) {
          out.push(`${ligHit.labelChain}${ligHit.resId}`);
          continue;
        }
        const bareTokenLigs = ligands.filter((l) => l.compId === tok.toUpperCase());
        if (bareTokenLigs.length > 0) {
          for (let i = 0; i < bareTokenLigs.length; i++) {
            if (i > 0) out.push('/0');
            out.push(`${bareTokenLigs[i].labelChain}${bareTokenLigs[i].resId}`);
          }
          continue;
        }
        out.push(tok);
        continue;
      }
      // Whole polymer chain ID like "B", "C", "AA"
      if (/^[A-Za-z][A-Za-z0-9]*$/.test(tok)) {
        const info = lookupAuthChain(tok) ?? lookupLabelChain(tok);
        if (info) {
          out.push(`${info.labelChain}${info.minLabelSeq}-${info.maxLabelSeq}`);
          continue;
        }
      }
      // Bare ligand comp_id like "SAM"
      const bareComp = tok.toUpperCase();
      const bareLigs = ligands.filter((l) => l.compId === bareComp);
      if (bareLigs.length > 0) {
        for (let i = 0; i < bareLigs.length; i++) {
          if (i > 0) out.push('/0');
          out.push(`${bareLigs[i].labelChain}${bareLigs[i].resId}`);
        }
        continue;
      }
      out.push(tok);
    }
    return out.join(',');
  };

  const splitBracketSafeCommas = (raw: string): string[] => {
    const items: string[] = [];
    let cur = '';
    let depth = 0;
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      if (ch === '[') depth++;
      else if (ch === ']') depth = Math.max(0, depth - 1);
      else if (ch === ',' && depth === 0) {
        items.push(cur);
        cur = '';
        continue;
      }
      cur += ch;
    }
    items.push(cur);
    return items;
  };

  const remapHbondSelection = (selStr: string): string => {
    const trimmed = selStr.trim();
    if (!trimmed) return trimmed;
    const remapPath = (path: string): string => {
      const parts = path.split('/');
      if (parts.length < 2) return path;
      const chainPart = parts[0].trim();
      const compPart = parts[1].trim().toUpperCase();
      const resPart = parts[2]?.trim() ?? '*';

      if (compPart && compPart !== '*') {
        const exactLig = ligands.filter(
          (l) =>
            l.compId === compPart &&
            (chainPart === '*' || l.authChain === chainPart || l.labelChain === chainPart),
        );
        const byAuth =
          exactLig.length > 0
            ? exactLig
            : ligands.filter(
                (l) =>
                  l.compId === compPart &&
                  (l.authChain.toUpperCase() === chainPart.toUpperCase() ||
                    l.labelChain.toUpperCase() === chainPart.toUpperCase()),
              );
        const matches = byAuth.length > 0 ? byAuth : ligands.filter((l) => l.compId === compPart);
        if (matches.length > 0) {
          const uniqueChains = [...new Set(matches.map((m) => m.labelChain))];
          parts[0] = uniqueChains.length === 1 ? uniqueChains[0] : `[${uniqueChains.join(',')}]`;
          return parts.join('/');
        }
      }

      if (chainPart && chainPart !== '*' && /^[A-Za-z][A-Za-z0-9_]*$/.test(chainPart)) {
        const origChain = chainPart;
        parts[0] = remapPolymerChain(origChain);
        if (resPart && resPart !== '*') {
          const inner = resPart.startsWith('[') && resPart.endsWith(']') ? resPart.slice(1, -1) : resPart;
          const rawItems = inner.split(',').map((s) => s.trim()).filter(Boolean);
          const parsedBounds = rawItems.map((item) => {
            const m = /^(-?\d+)([A-Za-z])?(?:-(-?\d+)([A-Za-z])?)?$/.exec(item);
            if (!m) return null;
            const lo = parseInt(m[1], 10);
            const loIns = m[2] || undefined;
            const hasHi = m[3] !== undefined;
            const hi = hasHi ? parseInt(m[3], 10) : lo;
            const hiIns = hasHi ? m[4] || undefined : loIns;
            return { item, lo, hi, loIns, hiIns };
          });
          // Support ligand selection by author chain + residue ID when compPart is '*' (e.g. sel('A/*/296/*'))
          if (
            compPart === '*' &&
            parsedBounds.length > 0 &&
            parsedBounds.every(
              (b) =>
                b !== null &&
                !b.loIns &&
                !b.hiIns &&
                !hasExactAuthResidueMatch(origChain, b.lo, b.hi),
            )
          ) {
            const origUpper = origChain.toUpperCase();
            const ligChains = new Set<string>();
            const allMatchedLigands = parsedBounds.every((b) => {
              const hit = ligands.find(
                (l) =>
                  (l.authChain === origChain ||
                    l.labelChain === origChain ||
                    l.authChain.toUpperCase() === origUpper ||
                    l.labelChain.toUpperCase() === origUpper) &&
                  l.resId >= b!.lo &&
                  l.resId <= b!.hi,
              );
              if (hit) ligChains.add(hit.labelChain);
              return Boolean(hit);
            });
            if (allMatchedLigands && ligChains.size > 0) {
              const uniqueLigChains = [...ligChains];
              parts[0] =
                uniqueLigChains.length === 1 ? uniqueLigChains[0] : `[${uniqueLigChains.join(',')}]`;
              return parts.join('/');
            }
          }

          const remappedNums: string[] = [];
          for (let idx = 0; idx < rawItems.length; idx++) {
            const item = rawItems[idx];
            const b = parsedBounds[idx];
            if (!b) {
              remappedNums.push(item);
              continue;
            }
            if ((b.loIns || b.hiIns) && !hasExactAuthResidueMatch(origChain, b.lo, b.hi, b.loIns, b.hiIns)) {
              remappedNums.push(item);
              continue;
            }
            const mapped = remapPolymerRange(origChain, b.lo, b.hi, b.loIns, b.hiIns);
            remappedNums.push(mapped.start === mapped.end ? `${mapped.start}` : `${mapped.start}-${mapped.end}`);
          }
          parts[2] =
            remappedNums.length > 1 || remappedNums[0]?.includes('-')
              ? `[${remappedNums.join(',')}]`
              : (remappedNums[0] ?? resPart);
        }
      }
      return parts.join('/');
    };

    if (trimmed.includes('sel(')) {
      return trimmed.replace(/sel\(\s*(?:"([^"]+)"|'([^']+)')\s*\)/g, (_m, dq: string | undefined, sq: string | undefined) => {
        const inner = dq ?? sq ?? '';
        const remapped = splitBracketSafeCommas(inner)
          .map((p) => remapPath(p.trim()))
          .join(',');
        return wrapRfd4Sel(remapped);
      });
    }
    return remapPath(trimmed);
  };

  const synthesizeBinderContig = (
    lockChain: string | undefined,
    binderLength: number,
  ): string | null => {
    const proteinChains = [...polymerByAuthChain.values()].filter((c) => !c.isNucleic);
    let target: Rfd4PolymerChainInfo | undefined;
    if (lockChain) {
      const ch = lockChain.trim();
      target =
        polymerByAuthChain.get(ch) ??
        polymerByLabelChain.get(ch) ??
        polymerByAuthChain.get(ch.toUpperCase()) ??
        polymerByLabelChain.get(ch.toUpperCase());
    }
    if (!target) target = proteinChains[0];
    if (!target) return null;
    const clamped = Math.max(30, Math.min(256, Math.round(binderLength || 70)));
    return `${target.labelChain}${target.minLabelSeq}-${target.maxLabelSeq},/0,${clamped}-${clamped}`;
  };

  return {
    byAuthRes,
    byLabelRes,
    polymerByAuthChain,
    polymerByLabelChain,
    ligands,
    nucleicChains,
    remapPolymerChain,
    remapPolymerRange,
    remapContig,
    remapResidueList,
    remapHbondSelection,
    synthesizeBinderContig,
  };
}

/** True when a contig string contains only a bare expandable monomer span (e.g. "70-70", "150-150"). */
export function isBareLengthContig(contig: string): boolean {
  const t = contig.trim();
  if (!t) return true;
  if (t.includes('/0') || t.includes(',')) return false;
  return /^\d+(?:-\d+)?[PRDN]?$/i.test(t);
}

function contigHasPreservedResidues(contig: string): boolean {
  return contig
    .split(',')
    .map((s) => s.trim())
    .some(
      (tok) =>
        /^[A-Za-z]+-?\d+[A-Za-z]?(?:--?\d+[A-Za-z]?)?$/.test(tok) ||
        (/^[A-Za-z0-9_]+:-?\d+[A-Za-z]?(?:--?\d+[A-Za-z]?)?$/.test(tok) &&
          !/^[A-Za-z0-9_]+:\d+-\d+[PRDN]$/i.test(tok)),
    );
}

function preservedChainsInContig(contig: string): Set<string> {
  const chains = new Set<string>();
  for (const raw of contig.split(',')) {
    const t = raw.trim();
    const m = /^([A-Za-z]+)-?\d+[A-Za-z]?(?:--?\d+[A-Za-z]?)?$/.exec(t);
    if (m) chains.add(m[1].toUpperCase());
  }
  return chains;
}

function formatResidueRangeNumSpec(r: ResidueRangeToken): string {
  const startTok = `${r.start}${r.startInsCode ?? ''}`;
  const endTok = `${r.end}${r.endInsCode ?? ''}`;
  return startTok === endTok ? startTok : `${startTok}-${endTok}`;
}

function matchesResNumInsBounds(
  resNum: number,
  insCode: string,
  start: number,
  end: number,
  startInsCode?: string,
  endInsCode?: string,
): boolean {
  if (start === end && !startInsCode && !endInsCode) {
    return resNum === start && insCode === '';
  }
  if (start === end && startInsCode && endInsCode && startInsCode === endInsCode) {
    return resNum === start && insCode === startInsCode;
  }
  const aboveLo =
    resNum > start || (resNum === start && (startInsCode ? insCode >= startInsCode : true));
  const belowHi =
    resNum < end || (resNum === end && (endInsCode ? insCode <= endInsCode : true));
  return aboveLo && belowHi;
}

function chainMatchesExactOrFallback(
  candidateChain: string,
  requestedChain: string,
  exactChainsInStructure: ReadonlySet<string>,
): boolean {
  if (candidateChain === requestedChain) return true;
  if (exactChainsInStructure.has(requestedChain)) return false;
  return candidateChain.toUpperCase() === requestedChain.toUpperCase();
}

/**
 * Parse comma/whitespace-separated tokens like `"A54, A56, A16-35"`, `"A10A, A10B"`,
 * `"AA1, AB1"`, or `"AA:1-5"` into structured residue range tokens preserving multi-char
 * chain IDs, exact chain case, and insertion codes.
 */
export function parseResidueRangeTokens(raw: string): ResidueRangeToken[] {
  const out: ResidueRangeToken[] = [];
  for (const tok of raw.split(/[,\s]+/).filter(Boolean)) {
    const trimmed = tok.trim();
    const colonMatch =
      /^([A-Za-z0-9_]+):(-?\d+)([A-Za-z])?(?:-(-?\d+)([A-Za-z])?)?$/.exec(trimmed);
    const bareMatch = !colonMatch
      ? /^([A-Za-z]+)?(-?\d+)([A-Za-z])?(?:-(-?\d+)([A-Za-z])?)?$/.exec(trimmed)
      : null;
    const m = colonMatch ?? bareMatch;
    if (!m) continue;
    const chain = m[1] || 'A';
    const start = parseInt(m[2], 10);
    const startInsCode = m[3] || undefined;
    const hasEnd = m[4] !== undefined;
    const end = hasEnd ? parseInt(m[4], 10) : start;
    const endInsCode = hasEnd ? m[5] || undefined : startInsCode;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) continue;
    if (end === start && startInsCode && endInsCode && endInsCode < startInsCode) continue;
    out.push({
      chain,
      start,
      end,
      ...(startInsCode ? { startInsCode } : {}),
      ...(endInsCode ? { endInsCode } : {}),
    });
  }
  return out;
}

/**
 * Resolve a motif selection specification (motif spans such as `"AA1, AB1, A10A-10B"` or a contig
 * such as `"10-10,A10-20,10-10"`) against a `ParsedStructure`, returning the exact ordered list of
 * `SelectedMotifSourceResidue` entries linking each selected motif residue back to its original
 * `{ authChainId, resNum, insCode, resName }` identity.
 */
export function resolveMotifSourceResidues(
  parsed: ParsedStructure,
  selectionSpec: string | readonly string[],
): SelectedMotifSourceResidue[] {
  const rawTokens = (Array.isArray(selectionSpec) ? selectionSpec : [selectionSpec])
    .flatMap((s) => String(s).split(/[,\s]+/))
    .map((s) => s.trim())
    .filter((s) => Boolean(s) && s !== '/0');

  if (rawTokens.length === 0 || parsed.nResidues === 0) return [];

  const hasChainPrefixedToken = rawTokens.some((t) => /^[A-Za-z]/.test(t) || t.includes(':'));
  const filteredTokens = hasChainPrefixedToken
    ? rawTokens.filter((t) => /^[A-Za-z]/.test(t) || t.includes(':'))
    : rawTokens;
  const ranges = parseResidueRangeTokens(filteredTokens.join(', '));
  if (ranges.length === 0) return [];

  const exactChains = new Set<string>(parsed.chainIds);
  const selected: SelectedMotifSourceResidue[] = [];

  for (const r of ranges) {
    for (let i = 0; i < parsed.nResidues; i++) {
      const src: SourceResidueId = parsed.sourceResidues?.[i] ?? {
        index: i,
        authChainId: parsed.chainIds[i] ?? 'A',
        resNum: parsed.resNums[i] ?? i + 1,
        insCode: parsed.insCodes?.[i] ?? '',
        resName: AA1_TO_AA3[parsed.sequence[i]?.toUpperCase() ?? 'X'] ?? 'UNK',
        projectedChainId: parsed.projectedChainIds?.[i] ?? (parsed.chainIds[i] ?? 'A')[0] ?? 'A',
      };
      if (
        chainMatchesExactOrFallback(src.authChainId, r.chain, exactChains) &&
        matchesResNumInsBounds(src.resNum, src.insCode, r.start, r.end, r.startInsCode, r.endInsCode)
      ) {
        selected.push({
          outputIndex: selected.length,
          templateIndex: src.index,
          authChainId: src.authChainId,
          ...(src.labelChainId ? { labelChainId: src.labelChainId } : {}),
          resNum: src.resNum,
          ...(src.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
          insCode: src.insCode,
          resName: src.resName,
          aa1: parsed.sequence[i] ?? AA3_TO_AA1[src.resName] ?? 'X',
          projectedChainId: src.projectedChainId,
        });
      }
    }
  }
  return selected;
}

/**
 * Wrap a path selection string in RFD4 `sel(...)`, automatically choosing double quotes when
 * atom names contain single-quote primes (e.g. ribose `O2'`, `O3'`, `O4'`) so RFD4's query
 * tokenizer never truncates at the prime character.
 */
function wrapRfd4Sel(selPath: string): string {
  const trimmed = selPath.trim();
  if (trimmed.includes('sel(')) return trimmed;
  if (trimmed.includes("'")) {
    return `sel("${trimmed.replace(/"/g, '\\"')}")`;
  }
  return `sel('${trimmed}')`;
}

const TIP_ATOMS_BY_RESNAME: Record<string, readonly string[]> = {
  ALA: ['CB', 'CA'],
  ARG: ['CZ', 'NH1', 'NH2'],
  ASN: ['CG', 'OD1', 'ND2'],
  ASP: ['CG', 'OD1', 'OD2'],
  CYS: ['CB', 'SG'],
  GLN: ['CD', 'OE1', 'NE2'],
  GLU: ['CD', 'OE1', 'OE2'],
  GLY: ['N', 'CA', 'C', 'O'],
  HIS: ['CG', 'ND1', 'CE1', 'NE2', 'CD2'],
  ILE: ['CB', 'CG1', 'CG2', 'CD1'],
  LEU: ['CG', 'CD1', 'CD2'],
  LYS: ['CE', 'NZ'],
  MET: ['CG', 'SD', 'CE'],
  PHE: ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
  PRO: ['CB', 'CG', 'CD'],
  SER: ['CB', 'OG'],
  THR: ['CB', 'OG1', 'CG2'],
  TRP: ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1'],
  TYR: ['CZ', 'OH', 'CE1', 'CE2'],
  VAL: ['CB', 'CG1', 'CG2'],
};

export interface ParsedMotifAtomEntry {
  range: ResidueRangeToken;
  /** Explicit uppercased atom names, or shorthand `'TIP' | 'BKBN' | 'ALL'`. */
  atoms: string[] | 'TIP' | 'BKBN' | 'ALL';
}

/**
 * Parse a `motifAtoms` specification into either a global shorthand (`'TIP' | 'BKBN' | 'ALL'`)
 * or per-residue atom-subset entries (e.g. `A102:ND1,CG,CE1; A104:OE1,CD; A129:SG,CB; A132:SG,CB,CA`).
 */
export function parseMotifAtomsSpec(raw: string | undefined): {
  shorthand?: 'TIP' | 'BKBN' | 'ALL';
  entries: ParsedMotifAtomEntry[];
} {
  const trimmed = String(raw ?? '').trim();
  if (!trimmed) return { entries: [] };
  const upper = trimmed.toUpperCase();
  if (upper === 'ALL' || upper === '*') return { shorthand: 'ALL', entries: [] };
  if (upper === 'BKBN' || upper === 'BACKBONE') return { shorthand: 'BKBN', entries: [] };
  if (upper === 'TIP') return { shorthand: 'TIP', entries: [] };

  const entries: ParsedMotifAtomEntry[] = [];
  for (const part of trimmed.split(/[;\n]+/).map((s) => s.trim()).filter(Boolean)) {
    const colonIdx = part.lastIndexOf(':');
    if (colonIdx <= 0 || colonIdx >= part.length - 1) continue;
    const lhs = part.slice(0, colonIdx).trim();
    const rhs = part.slice(colonIdx + 1).trim();
    if (!lhs || !rhs) continue;
    const ranges = parseResidueRangeTokens(lhs);
    if (ranges.length === 0) continue;
    const rhsUpper = rhs.toUpperCase();
    let atoms: string[] | 'TIP' | 'BKBN' | 'ALL';
    if (rhsUpper === 'ALL' || rhsUpper === '*') {
      atoms = 'ALL';
    } else if (rhsUpper === 'BKBN' || rhsUpper === 'BACKBONE') {
      atoms = 'BKBN';
    } else if (rhsUpper === 'TIP') {
      atoms = 'TIP';
    } else {
      const list = rhs
        .replace(/^\[|\]$/g, '')
        .split(/[,\s]+/)
        .map((a) => a.trim().toUpperCase())
        .filter(Boolean);
      if (list.length === 0) continue;
      atoms = list;
    }
    for (const range of ranges) {
      entries.push({ range, atoms });
    }
  }
  return { entries };
}

/** Extract a clean, model-agnostic CanonicalConditionIntent from node params. */
export function buildCanonicalConditionIntent(
  params: Record<string, unknown>,
): CanonicalConditionIntent {
  const contig = String(params.contig ?? '70-70').trim() || '70-70';
  const lockChain = String(params.lockChain ?? '').trim() || undefined;
  const hotspots = String(params.hotspots ?? '')
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const motifSpans = String(params.motifSpan ?? '')
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const motifAtoms = String(params.motifAtoms ?? '').trim() || undefined;
  const unindexedMotif = Boolean(params.unindexedMotif);
  const hbondMode = (String(params.hbondMode ?? 'none') as CanonicalConditionIntent['hbondMode']) || 'none';
  const hbondSelection = String(params.hbondSelection ?? '').trim() || undefined;
  const hbondDonorSelection = String(params.hbondDonorSelection ?? '').trim() || undefined;
  const hbondAcceptorSelection = String(params.hbondAcceptorSelection ?? '').trim() || undefined;
  const rasaBuriedSelection = String(params.rasaBuriedSelection ?? '').trim() || undefined;
  const piStackSelection = String(params.piStackSelection ?? '').trim() || undefined;
  const cathCode = String(params.cathCode ?? '').trim() || undefined;
  const benchmarkId = String(params.benchmarkId ?? '').trim() || undefined;
  const pdbId = String(params.pdbId ?? '').trim().toUpperCase() || undefined;

  return {
    contig,
    ...(lockChain ? { lockChain } : {}),
    ...(hotspots.length > 0 ? { hotspots } : {}),
    ...(motifSpans.length > 0 ? { motifSpans } : {}),
    ...(motifAtoms ? { motifAtoms } : {}),
    ...(unindexedMotif ? { unindexedMotif: true } : {}),
    ...(hbondMode && hbondMode !== 'none' ? { hbondMode } : {}),
    ...(hbondSelection ? { hbondSelection } : {}),
    ...(hbondDonorSelection ? { hbondDonorSelection } : {}),
    ...(hbondAcceptorSelection ? { hbondAcceptorSelection } : {}),
    ...(rasaBuriedSelection ? { rasaBuriedSelection } : {}),
    ...(piStackSelection ? { piStackSelection } : {}),
    ...(cathCode ? { cathCode } : {}),
    ...(benchmarkId ? { benchmarkId } : {}),
    ...(pdbId ? { pdbId } : {}),
  };
}

/**
 * Compile model-agnostic CanonicalConditionIntent -> RFD4-Proteina `conditions` dictionary.
 * Keeps all RFD4-specific `sel(...)` and `C_*` syntax cleanly isolated from the core DAG engine.
 *
 * When `templateContext` is supplied:
 * - Remaps `auth_asym_id`/`auth_seq_id` (and ligand `auth_asym_id:comp_id`) to RFD4's `label_*` IDs.
 * - Injects `motifSpans` into bare-length contigs so `applyContig` preserves motif atoms.
 * - Injects preserved template ligands/ions and DNA/RNA chains into `contig` and `C_CRDGRP`/`C_CRD`/`C_SEQ`.
 */
export function compileConditionToRfd4(
  intent: CanonicalConditionIntent,
  templateContext?: Rfd4TemplateContext,
  opts?: { omitContig?: boolean; cropSegments?: string; contigAlreadyLabelSpace?: boolean },
): {
  contig: string;
  conditions: Record<string, unknown>;
  tags: string[];
  selectedSourceResidues?: SelectedMotifSourceResidue[];
} {
  let contig = intent.contig.trim() || '70-70';
  if (templateContext && !opts?.omitContig && !opts?.contigAlreadyLabelSpace) {
    contig = templateContext.remapContig(contig);
  }

  const effectiveLockChain = intent.lockChain
    ? templateContext
      ? templateContext.remapPolymerChain(intent.lockChain)
      : intent.lockChain.trim().toUpperCase()
    : undefined;

  const effectiveHotspots =
    intent.hotspots && intent.hotspots.length > 0
      ? templateContext
        ? templateContext.remapResidueList(intent.hotspots)
        : intent.hotspots
      : [];

  const effectiveMotifSpans =
    intent.motifSpans && intent.motifSpans.length > 0
      ? templateContext
        ? templateContext.remapResidueList(intent.motifSpans)
        : intent.motifSpans
      : [];

  // Inject motif spans into bare-length contigs (e.g. AME benchmarks `150-150` + `A102, A104, A129, A132`)
  // so RFD4's `applyContig` preserves the motif atoms instead of discarding the input AtomArray!
  if (!opts?.omitContig && effectiveMotifSpans.length > 0 && !contigHasPreservedResidues(contig)) {
    const parsedSpans = parseResidueRangeTokens(effectiveMotifSpans.join(', '));
    if (parsedSpans.length > 0) {
      const spanTokens = parsedSpans.map((r) => `${r.chain}${formatResidueRangeNumSpec(r)}`);
      const uniqueMotifChains = new Set(parsedSpans.map((r) => r.chain));
      if (intent.unindexedMotif) {
        if (uniqueMotifChains.size === 1) {
          // Matches upstream RFD4 `motif_unidx_1prw` syntax: `150-150,A102,A104,A129,A132`
          contig = `${contig},${spanTokens.join(',')}`;
        } else {
          // Separate preserved segments from different chains with `/0` to satisfy `applyContig`
          const groupedByChain = new Map<string, string[]>();
          for (let i = 0; i < parsedSpans.length; i++) {
            const list = groupedByChain.get(parsedSpans[i].chain) ?? [];
            list.push(spanTokens[i]);
            groupedByChain.set(parsedSpans[i].chain, list);
          }
          const chainParts = [...groupedByChain.values()].map((list) => list.join(','));
          contig = `${contig},/0,${chainParts.join(',/0,')}`;
        }
      } else {
        // Indexed motif with bare scaffold length: interleave balanced flanking scaffold segments
        const targetTotal = parseLengthRangeMid(contig) ?? 80;
        const motifTotal = parsedSpans.reduce((s, r) => s + (r.end - r.start + 1), 0);
        const remaining = Math.max(parsedSpans.length + 1, targetTotal - motifTotal);
        const gapSlots = parsedSpans.length + 1;
        const baseGap = Math.max(4, Math.floor(remaining / gapSlots));
        const pieces: string[] = [];
        let prevChain: string | null = null;
        for (let i = 0; i < parsedSpans.length; i++) {
          if (prevChain !== null && parsedSpans[i].chain !== prevChain) {
            pieces.push('/0');
          }
          pieces.push(`${baseGap}-${baseGap}`, spanTokens[i]);
          prevChain = parsedSpans[i].chain;
        }
        const tailGap = Math.max(4, remaining - baseGap * parsedSpans.length);
        pieces.push(`${tailGap}-${tailGap}`);
        contig = pieces.join(',');
      }
    }
  }

  // Inject preserved template ligands/ions and nucleic acid chains into contig.
  // Skip crystallographic buffers/additives unless explicitly named by the user in contig,
  // cropSegments, or hbondSelection (preserving biological metals like ZN/MG/FE2 and cofactors).
  const explicitlyNamedLigands = new Set<string>();
  const collectExplicitLigandRefs = (raw: string | undefined) => {
    if (!raw) return;
    for (const rawTok of raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)) {
      if (rawTok === '/0') continue;
      const chainComp = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(rawTok);
      if (chainComp && !/^\d+[A-Za-z]?$/.test(chainComp[2])) {
        explicitlyNamedLigands.add(chainComp[2].toUpperCase());
        continue;
      }
      if (/^[A-Za-z0-9]{1,5}$/.test(rawTok) && !/^\d+$/.test(rawTok)) {
        explicitlyNamedLigands.add(rawTok.toUpperCase());
      }
    }
  };
  collectExplicitLigandRefs(intent.contig);
  collectExplicitLigandRefs(opts?.cropSegments);
  const collectLigandsFromPathSel = (rawSel: string | undefined) => {
    if (!rawSel) return;
    for (const seg of rawSel.split(/[,|)\s'"]+/).filter(Boolean)) {
      const parts = seg.split('/');
      if (parts.length >= 2 && parts[1] && parts[1] !== '*') {
        explicitlyNamedLigands.add(parts[1].trim().toUpperCase());
      }
    }
  };
  collectLigandsFromPathSel(intent.hbondSelection);
  collectLigandsFromPathSel(intent.hbondDonorSelection);
  collectLigandsFromPathSel(intent.hbondAcceptorSelection);
  collectLigandsFromPathSel(intent.rasaBuriedSelection);
  collectLigandsFromPathSel(intent.piStackSelection);

  const injectedLigands: Rfd4LigandEntry[] = [];
  const injectedNucleic: Rfd4PolymerChainInfo[] = [];
  if (templateContext) {
    const existingChains = preservedChainsInContig(contig);
    for (const lig of templateContext.ligands) {
      const upperComp = lig.compId.toUpperCase();
      const upperLabelChain = lig.labelChain.toUpperCase();
      const explicitlySelected =
        existingChains.has(upperLabelChain) || explicitlyNamedLigands.has(upperComp);
      if (isCrystallizationBuffer(upperComp) && !explicitlySelected) {
        continue;
      }
      if (!existingChains.has(upperLabelChain)) {
        if (!opts?.omitContig) {
          contig = `${contig},/0,${lig.labelChain}${lig.resId}`;
        }
        existingChains.add(upperLabelChain);
      }
      injectedLigands.push(lig);
    }
    for (const na of templateContext.nucleicChains) {
      if (!existingChains.has(na.labelChain.toUpperCase())) {
        if (!opts?.omitContig) {
          contig = `${contig},/0,${na.labelChain}${na.minLabelSeq}-${na.maxLabelSeq}`;
        }
        existingChains.add(na.labelChain.toUpperCase());
      }
      injectedNucleic.push(na);
    }
  }

  const conditions: Record<string, unknown> = {};
  const tags: string[] = [contig];

  const appendSelect = (
    key: 'C_CRD' | 'C_SEQ',
    selector: string,
  ) => {
    const prev = (conditions[key] as { true?: Array<{ select: string }> } | undefined)?.true ?? [];
    conditions[key] = { true: [...prev, { select: selector }] };
  };

  if (effectiveLockChain) {
    appendSelect('C_CRD', `sel('${effectiveLockChain}/*/*/*')`);
    appendSelect('C_SEQ', `sel('${effectiveLockChain}/*/*/*')`);
    tags.push(`lock:${effectiveLockChain}`);
  }

  if (effectiveHotspots.length > 0) {
    const byChain = new Map<string, string[]>();
    for (const r of parseResidueRangeTokens(effectiveHotspots.join(', '))) {
      const list = byChain.get(r.chain) ?? [];
      list.push(formatResidueRangeNumSpec(r));
      byChain.set(r.chain, list);
    }
    const exprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
    if (exprs.length > 0) {
      conditions.C_HOT = { true: [{ select: exprs.join(' | ') }] };
      tags.push(`hot:${effectiveHotspots.join(', ')}`);
    }
  }

  let hasMotifCrdGrp = false;
  if (effectiveMotifSpans.length > 0) {
    const parsedMotifSpans = parseResidueRangeTokens(effectiveMotifSpans.join(', '));
    const byChain = new Map<string, string[]>();
    for (const r of parsedMotifSpans) {
      const list = byChain.get(r.chain) ?? [];
      list.push(formatResidueRangeNumSpec(r));
      byChain.set(r.chain, list);
    }
    const resExprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
    if (resExprs.length > 0) {
      const resSel = resExprs.join(' | ');
      const parsedAtoms = parseMotifAtomsSpec(intent.motifAtoms);
      const formatResSpec = (start: number, end: number, startIns?: string, endIns?: string): string => {
        const sTok = `${start}${startIns ?? ''}`;
        const eTok = `${end}${endIns ?? ''}`;
        return sTok === eTok ? sTok : `[${sTok}-${eTok}]`;
      };
      const emitTipAtomSelectors = (chain: string, start: number, end: number): string[] => {
        if (!templateContext) {
          return [`sel('${chain}/*/${formatResSpec(start, end)}/*')`];
        }
        const out: string[] = [];
        for (let resId = start; resId <= end; resId++) {
          const entry = templateContext.byLabelRes.get(`${chain}:${resId}`);
          const tipAtoms = entry ? TIP_ATOMS_BY_RESNAME[entry.compId.toUpperCase()] : undefined;
          if (tipAtoms && tipAtoms.length > 0) {
            out.push(`sel('${chain}/*/${resId}/[${tipAtoms.join(',')}]')`);
          } else {
            out.push(`sel('${chain}/*/${resId}/*')`);
          }
        }
        return out;
      };

      let crdGrpSel = resSel;
      if (parsedAtoms.shorthand === 'BKBN') {
        crdGrpSel = [...byChain.entries()]
          .map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/[N,CA,C,O]')`)
          .join(' | ');
      } else if (parsedAtoms.shorthand === 'TIP') {
        const tipExprs: string[] = [];
        for (const r of parsedMotifSpans) {
          tipExprs.push(...emitTipAtomSelectors(r.chain, r.start, r.end));
        }
        if (tipExprs.length > 0) crdGrpSel = tipExprs.join(' | ');
      } else if (parsedAtoms.entries.length > 0) {
        const atomExprs: string[] = [];
        const coveredKeys = new Set<string>();
        for (const entry of parsedAtoms.entries) {
          const r = entry.range;
          const mapped = templateContext
            ? templateContext.remapPolymerRange(r.chain, r.start, r.end, r.startInsCode, r.endInsCode)
            : { labelChain: r.chain, start: r.start, end: r.end };
          for (let k = mapped.start; k <= mapped.end; k++) {
            coveredKeys.add(`${mapped.labelChain}:${k}`);
          }
          if (entry.atoms === 'TIP') {
            atomExprs.push(...emitTipAtomSelectors(mapped.labelChain, mapped.start, mapped.end));
          } else {
            const atomPart =
              entry.atoms === 'ALL'
                ? '*'
                : entry.atoms === 'BKBN'
                  ? '[N,CA,C,O]'
                  : entry.atoms.length === 1 && entry.atoms[0] === '*'
                    ? '*'
                    : `[${entry.atoms.join(',')}]`;
            const resPart = formatResSpec(mapped.start, mapped.end);
            atomExprs.push(wrapRfd4Sel(`${mapped.labelChain}/*/${resPart}/${atomPart}`));
          }
        }
        // Preserve full-residue coverage for any motifSpan residues not explicitly subsetted in motifAtoms
        for (const r of parsedMotifSpans) {
          const uncovered: number[] = [];
          for (let k = r.start; k <= r.end; k++) {
            if (!coveredKeys.has(`${r.chain}:${k}`)) uncovered.push(k);
          }
          if (uncovered.length > 0) {
            const resPart = uncovered.length === 1 ? `${uncovered[0]}` : `[${uncovered.join(',')}]`;
            atomExprs.push(`sel('${r.chain}/*/${resPart}/*')`);
          }
        }
        if (atomExprs.length > 0) crdGrpSel = atomExprs.join(' | ');
      }

      conditions.C_CRDGRP = { 1: [{ select: crdGrpSel }] };
      hasMotifCrdGrp = true;
      // Keep C_SEQ and C_IDX at residue level (/*) so sequence identity & unindexed insertion apply to the whole residue
      appendSelect('C_SEQ', resSel);
      if (intent.unindexedMotif) {
        // P0-2: Emit distinct unindexed group (`motif_0`, `motif_1`, ...) per disjoint catalytic residue / island
        conditions.C_IDX = {
          false: parsedMotifSpans.map((r, idx) => ({
            select: `sel('${r.chain}/*/${formatResSpec(r.start, r.end, r.startInsCode, r.endInsCode)}/*')`,
            group: `motif_${idx}`,
          })),
        };
      }
      tags.push(
        intent.unindexedMotif
          ? `unidx-motif:${effectiveMotifSpans.join(', ')}`
          : `motif:${effectiveMotifSpans.join(', ')}`,
      );
      if (parsedAtoms.shorthand && parsedAtoms.shorthand !== 'ALL') {
        tags.push(`atoms:${parsedAtoms.shorthand}`);
      } else if (parsedAtoms.entries.length > 0) {
        tags.push('atoms:tip');
      }
    }
  }

  // Condition template ligands/ions:
  // - With motif spans (e.g. AME enzymes 1CTT/1OS7): join C_CRDGRP group 1 so catalytic residues,
  //   metals, and substrates preserve their exact mutual 3D catalytic geometry!
  // - Without motif spans (e.g. 7C7M SAM pocket or custom holo uploads): condition via C_CRD=true.
  if (injectedLigands.length > 0) {
    const ligChains = [...new Set(injectedLigands.map((l) => l.labelChain))];
    const ligSel = ligChains.map((ch) => `sel('${ch}/*/*/*')`).join(' | ');
    if (hasMotifCrdGrp) {
      const grp = conditions.C_CRDGRP as { 1: Array<{ select: string }> };
      grp[1] = [{ select: `${grp[1][0].select} | ${ligSel}` }];
    } else {
      appendSelect('C_CRD', ligSel);
    }
    tags.push(`lig:${[...new Set(injectedLigands.map((l) => l.compId))].join('+')}`);
  }

  // Condition template DNA/RNA chains (e.g. 7M5W B-DNA or custom DNA uploads) via C_CRD + C_SEQ
  if (injectedNucleic.length > 0) {
    const naChains = [...new Set(injectedNucleic.map((c) => c.labelChain))];
    const naSel = naChains.map((ch) => `sel('${ch}/*/*/*')`).join(' | ');
    appendSelect('C_CRD', naSel);
    appendSelect('C_SEQ', naSel);
    tags.push(`dna:${naChains.join(',')}`);
  }

  const hbondEntries: Record<string, Array<{ select: string }>> = {};
  const appendHbondRole = (role: string, rawSel: string) => {
    const remappedSel = templateContext ? templateContext.remapHbondSelection(rawSel) : rawSel;
    const expr = wrapRfd4Sel(remappedSel);
    hbondEntries[role] = [...(hbondEntries[role] ?? []), { select: expr }];
    if (!tags.includes(`HB:${role}`)) tags.push(`HB:${role}`);
  };
  if (intent.hbondDonorSelection) {
    appendHbondRole('DONOR', intent.hbondDonorSelection);
  }
  if (intent.hbondAcceptorSelection) {
    appendHbondRole('ACCEPTOR', intent.hbondAcceptorSelection);
  }
  if (intent.hbondMode && intent.hbondMode !== 'none' && intent.hbondSelection) {
    appendHbondRole(intent.hbondMode, intent.hbondSelection);
  }
  if (Object.keys(hbondEntries).length > 0) {
    conditions.C_HB = hbondEntries;
  }

  if (intent.rasaBuriedSelection) {
    const remappedSel = templateContext
      ? templateContext.remapHbondSelection(intent.rasaBuriedSelection)
      : intent.rasaBuriedSelection;
    conditions.C_RAS = { BURIED: [{ select: wrapRfd4Sel(remappedSel) }] };
    tags.push('RASA:BURIED');
  }

  if (intent.piStackSelection) {
    const remappedSel = templateContext
      ? templateContext.remapHbondSelection(intent.piStackSelection)
      : intent.piStackSelection;
    conditions.C_PISTACK = { true: [{ select: wrapRfd4Sel(remappedSel) }] };
    tags.push('PISTACK');
  }

  if (intent.cathCode) {
    conditions.C_CATH = { [intent.cathCode]: [{ select: "sel('*/*/*/*')" }] };
    tags.push(`CATH:${intent.cathCode}`);
  }

  let selectedSourceResidues: SelectedMotifSourceResidue[] | undefined;
  if (templateContext) {
    const allPolyRes = [...templateContext.byLabelRes.values()].filter((r) => !r.isNucleic);
    if (allPolyRes.length > 0) {
      const exactAuthChains = new Set(allPolyRes.map((r) => r.authChain));
      const exactLabelChains = new Set(allPolyRes.map((r) => r.labelChain));
      const proj = projectChainIdsToPdb(allPolyRes.map((r) => r.authChain));
      const makeEntry = (
        entry: Rfd4PolymerResidueEntry,
        outputIndex: number,
      ): SelectedMotifSourceResidue => ({
        outputIndex,
        templateIndex: entry.templateIndex,
        authChainId: entry.authChain,
        labelChainId: entry.labelChain,
        resNum: entry.authSeqId,
        labelSeqId: entry.labelSeqId,
        insCode: entry.insCode,
        resName: entry.compId,
        aa1: AA3_TO_AA1[entry.compId] ?? 'X',
        projectedChainId: proj.sourceToProjected[entry.authChain] ?? entry.authChain[0] ?? 'A',
      });

      // Walk the compiled contig (in label_* space) to assign exact compiled outputIndex offsets
      const compiledParts = contig
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const motifLabelKeys = new Set<string>();
      if (intent.motifSpans && intent.motifSpans.length > 0) {
        for (const r of parseResidueRangeTokens(intent.motifSpans.join(', '))) {
          for (const entry of allPolyRes) {
            if (
              chainMatchesExactOrFallback(entry.authChain, r.chain, exactAuthChains) &&
              matchesResNumInsBounds(entry.authSeqId, entry.insCode, r.start, r.end, r.startInsCode, r.endInsCode)
            ) {
              motifLabelKeys.add(`${entry.labelChain}:${entry.labelSeqId}`);
            }
          }
        }
      }

      const resolved: SelectedMotifSourceResidue[] = [];
      let cursor = 0;
      const filterToMotif = motifLabelKeys.size > 0;
      for (const tok of compiledParts) {
        if (tok === '/0') continue;
        const lenMid = parseLengthRangeMid(tok.replace(/[PRDN]$/i, ''));
        if (lenMid !== null && /^\d+(?:-\d+)?[PRDN]?$/i.test(tok)) {
          cursor += lenMid;
          continue;
        }
        const namedExp = /^([A-Za-z0-9_]+):(\d+(?:-\d+)?)[PRDN]?$/i.exec(tok);
        if (namedExp) {
          const prefix = namedExp[1];
          const isPolymerChain =
            exactLabelChains.has(prefix) ||
            exactAuthChains.has(prefix) ||
            [...exactLabelChains].some((c) => c.toUpperCase() === prefix.toUpperCase()) ||
            [...exactAuthChains].some((c) => c.toUpperCase() === prefix.toUpperCase());
          if (!isPolymerChain) {
            const mid = parseLengthRangeMid(namedExp[2]);
            if (mid !== null) cursor += mid;
            continue;
          }
        }
        const ranges = parseResidueRangeTokens(tok);
        for (const r of ranges) {
          if (r.startInsCode || r.endInsCode) continue;
          const hits = allPolyRes.filter(
            (e) =>
              chainMatchesExactOrFallback(e.labelChain, r.chain, exactLabelChains) &&
              e.labelSeqId >= r.start &&
              e.labelSeqId <= r.end,
          );
          for (const entry of hits) {
            const labelKey = `${entry.labelChain}:${entry.labelSeqId}`;
            if (!filterToMotif || motifLabelKeys.has(labelKey)) {
              resolved.push(makeEntry(entry, cursor));
            }
            cursor++;
          }
        }
      }

      // Fallback when omitContig is true or motifSpans exist without compiled contig placement
      if (resolved.length === 0 && intent.motifSpans && intent.motifSpans.length > 0) {
        for (const r of parseResidueRangeTokens(intent.motifSpans.join(', '))) {
          for (const entry of allPolyRes) {
            if (
              chainMatchesExactOrFallback(entry.authChain, r.chain, exactAuthChains) &&
              matchesResNumInsBounds(entry.authSeqId, entry.insCode, r.start, r.end, r.startInsCode, r.endInsCode)
            ) {
              resolved.push(makeEntry(entry, resolved.length));
            }
          }
        }
      }
      if (resolved.length > 0) selectedSourceResidues = resolved;
    }
  }

  return {
    contig,
    conditions,
    tags,
    ...(selectedSourceResidues ? { selectedSourceResidues } : {}),
  };
}

/**
 * Verify that every compiled RFD4 condition selector matches > 0 atoms after applying `contig`.
 * Prevents silent fallback to an unconditional monomer when a chain/residue selector misses.
 */
export function assertRfd4ConditionSelectorsMatch(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  inputAtomArray: any,
  contig: string | null | undefined,
  conditions: Record<string, unknown>,
  helpers: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    applyContig: (atoms: any, contig: string) => { atomArray: any; originalResId: number[] };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    evalSelection: (atoms: any, queryStr: string) => boolean[];
  },
): void {
  const condEntries = Object.entries(conditions ?? {});
  if (condEntries.length === 0) return;

  const { atomArray: contigAtoms, originalResId } = contig
    ? helpers.applyContig(inputAtomArray, contig)
    : {
        atomArray: inputAtomArray,
        originalResId: Array.isArray(inputAtomArray?.res_id) ? inputAtomArray.res_id.slice() : [],
      };

  const savedResId = contigAtoms.res_id;
  contigAtoms.res_id = originalResId.slice();
  try {
    for (const [condName, valueMap] of condEntries) {
      if (!valueMap || typeof valueMap !== 'object') continue;
      for (const [valKey, entries] of Object.entries(valueMap as Record<string, unknown>)) {
        if (!Array.isArray(entries)) continue;
        for (const entry of entries as Array<{ select?: string }>) {
          if (!entry?.select) continue;
          const mask = helpers.evalSelection(contigAtoms, entry.select);
          let matched = 0;
          for (let i = 0; i < mask.length; i++) {
            if (mask[i]) matched++;
          }
          if (matched === 0) {
            throw new Error(
              `RFD4 condition ${condName}=${valKey} selector ${entry.select} matched 0 atoms` +
                (contig ? ` after applying contig "${contig}"` : '') +
                '. Check chain IDs, residue numbering, and template structure.',
            );
          }
        }
      }
    }
  } finally {
    contigAtoms.res_id = savedResId;
  }
}

export interface Rfd4SelectorValidationOptions {
  contig?: string | null;
  motifSpan?: string | readonly string[] | null;
  motifAtoms?: string | null;
  hotspots?: string | readonly string[] | null;
  templateLabel?: string | null;
  /** Allow label_asym_id + label_seq_id fallback even when authChain exists (for raw label-space overrides). */
  allowLabelSpaceFallback?: boolean;
}

function lookupTemplateAuthChain(
  ctx: Rfd4TemplateContext,
  chain: string,
): Rfd4PolymerChainInfo | undefined {
  const t = chain.trim();
  return ctx.polymerByAuthChain.get(t) ?? ctx.polymerByAuthChain.get(t.toUpperCase());
}

function lookupTemplateLabelChain(
  ctx: Rfd4TemplateContext,
  chain: string,
): Rfd4PolymerChainInfo | undefined {
  const t = chain.trim();
  return ctx.polymerByLabelChain.get(t) ?? ctx.polymerByLabelChain.get(t.toUpperCase());
}

function summarizePolymerChainResidues(
  info: Rfd4PolymerChainInfo,
  useLabelSpace = false,
): string {
  const cid = useLabelSpace ? info.labelChain : info.authChain;
  const runs: string[] = [];
  let runStart: number | null = null;
  let runEnd: number | null = null;
  const flushRun = () => {
    if (runStart === null || runEnd === null) return;
    runs.push(runStart === runEnd ? `${cid}${runStart}` : `${cid}${runStart}-${runEnd}`);
    runStart = null;
    runEnd = null;
  };
  for (const r of info.residues) {
    const seq = useLabelSpace ? r.labelSeqId : r.authSeqId;
    const ins = useLabelSpace ? '' : r.insCode;
    if (ins) {
      flushRun();
      runs.push(`${cid}${seq}${ins}`);
      continue;
    }
    if (runStart === null || runEnd === null) {
      runStart = seq;
      runEnd = seq;
    } else if (seq === runEnd) {
      continue;
    } else if (seq === runEnd + 1) {
      runEnd = seq;
    } else {
      flushRun();
      runStart = seq;
      runEnd = seq;
    }
  }
  flushRun();
  if (runs.length <= 12) {
    return runs.join(', ');
  }
  const tail = runs[runs.length - 1];
  return `${runs.slice(0, 10).join(', ')}, … (${info.residues.length} residues through ${tail})`;
}

function formatRangeTokenDisplay(r: ResidueRangeToken): string {
  return `${r.chain}${formatResidueRangeNumSpec(r)}`;
}

/**
 * Validate every preserved residue/range selector in `contig`, `motifSpan`, `motifAtoms`, and
 * `hotspots` against `templateContext` BEFORE downloading 330-468 MB of RFD4 weights (#97).
 * Throws an actionable error listing the unmatched selector(s) and available residues on that chain.
 */
export function validateRfd4TemplateSelectors(
  templateContext: Rfd4TemplateContext,
  opts: Rfd4SelectorValidationOptions = {},
): void {
  const unmatched: Array<{ display: string; chain: string; isRange: boolean }> = [];
  const seenUnmatched = new Set<string>();
  const recordUnmatched = (display: string, chain: string, isRange: boolean) => {
    if (seenUnmatched.has(display)) return;
    seenUnmatched.add(display);
    unmatched.push({ display, chain, isRange });
  };

  const matchesRange = (r: ResidueRangeToken, allowLigand: boolean): boolean => {
    const authPoly = lookupTemplateAuthChain(templateContext, r.chain);
    const labelPoly = lookupTemplateLabelChain(templateContext, r.chain);
    const hasExplicitIns = Boolean(r.startInsCode || r.endInsCode);
    if (authPoly) {
      const authHit = authPoly.residues.some((res) =>
        matchesResNumInsBounds(
          res.authSeqId,
          res.insCode,
          r.start,
          r.end,
          r.startInsCode,
          r.endInsCode,
        ),
      );
      if (authHit) return true;
      if (
        opts.allowLabelSpaceFallback &&
        !hasExplicitIns &&
        labelPoly?.residues.some((res) => res.labelSeqId >= r.start && res.labelSeqId <= r.end)
      ) {
        return true;
      }
    } else if (!hasExplicitIns && labelPoly) {
      if (labelPoly.residues.some((res) => res.labelSeqId >= r.start && res.labelSeqId <= r.end)) {
        return true;
      }
    }
    if (allowLigand && !hasExplicitIns) {
      const chUpper = r.chain.toUpperCase();
      const ligHit = templateContext.ligands.some(
        (l) =>
          (l.labelChain === r.chain ||
            l.authChain === r.chain ||
            l.labelChain.toUpperCase() === chUpper ||
            l.authChain.toUpperCase() === chUpper) &&
          l.resId >= r.start &&
          l.resId <= r.end,
      );
      if (ligHit) return true;
    }
    return false;
  };

  const rawContig = typeof opts.contig === 'string' ? opts.contig.trim() : '';
  if (rawContig) {
    const parts = rawContig
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    for (const tok of parts) {
      if (tok === '/0') continue;
      if (/^\d+(?:-\d+)?[PRDN]?$/i.test(tok)) continue;
      if (/^[A-Za-z0-9_]+:\d+(?:-\d+)?[PRDN]?$/i.test(tok)) {
        const prefix = tok.slice(0, tok.indexOf(':'));
        if (
          !lookupTemplateAuthChain(templateContext, prefix) &&
          !lookupTemplateLabelChain(templateContext, prefix)
        ) {
          continue;
        }
      }
      const chainLig = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(tok);
      if (chainLig && !/^-?\d+[A-Za-z]?$/.test(chainLig[2])) {
        const comp = chainLig[2].toUpperCase();
        if (templateContext.ligands.some((l) => l.compId === comp)) continue;
        recordUnmatched(tok, chainLig[1], false);
        continue;
      }
      if (/^[A-Za-z][A-Za-z0-9]*$/.test(tok) && !/\d/.test(tok)) {
        if (
          lookupTemplateAuthChain(templateContext, tok) ||
          lookupTemplateLabelChain(templateContext, tok) ||
          templateContext.ligands.some((l) => l.compId === tok.toUpperCase())
        ) {
          continue;
        }
        recordUnmatched(tok, tok, false);
        continue;
      }
      if (templateContext.ligands.some((l) => l.compId === tok.toUpperCase())) {
        continue;
      }
      const parsed =
        /^[A-Za-z]/.test(tok) || tok.includes(':') ? parseResidueRangeTokens(tok) : [];
      if (parsed.length === 1) {
        const pt = parsed[0];
        if (!matchesRange(pt, true)) {
          const isRange = pt.start !== pt.end || pt.startInsCode !== pt.endInsCode;
          recordUnmatched(formatRangeTokenDisplay(pt), pt.chain, isRange);
        }
        continue;
      }
      recordUnmatched(tok, tok.replace(/[^A-Za-z0-9_].*$/, '') || 'A', false);
    }
  }

  const checkPolymerTokenList = (raw: string | readonly string[] | null | undefined) => {
    if (!raw) return;
    const joined = Array.isArray(raw) ? raw.join(', ') : String(raw);
    if (!joined.trim()) return;
    for (const r of parseResidueRangeTokens(joined)) {
      if (!matchesRange(r, false)) {
        const isRange = r.start !== r.end || r.startInsCode !== r.endInsCode;
        recordUnmatched(formatRangeTokenDisplay(r), r.chain, isRange);
      }
    }
  };

  checkPolymerTokenList(opts.motifSpan);
  const parsedMotifAtoms = parseMotifAtomsSpec(opts.motifAtoms ?? undefined);
  for (const entry of parsedMotifAtoms.entries) {
    const r = entry.range;
    if (!matchesRange(r, false)) {
      const isRange = r.start !== r.end || r.startInsCode !== r.endInsCode;
      recordUnmatched(formatRangeTokenDisplay(r), r.chain, isRange);
    }
  }
  checkPolymerTokenList(opts.hotspots);

  if (unmatched.length === 0) return;

  const tplName = String(opts.templateLabel ?? '').trim();
  const tplPart = tplName ? ` in template "${tplName}"` : ' in wired template';
  const headPart =
    unmatched.length === 1
      ? unmatched[0].isRange
        ? `Residue range ${unmatched[0].display} not found${tplPart}`
        : `Residue ${unmatched[0].display} not found${tplPart}`
      : `Residues ${unmatched.map((u) => u.display).join(', ')} not found${tplPart}`;

  const requestedChains = [...new Set(unmatched.map((u) => u.chain.trim()).filter(Boolean))];
  const resolvedChains: Array<{ chainLabel: string; summary: string }> = [];
  let missingChain = false;
  for (const ch of requestedChains) {
    const authInfo = lookupTemplateAuthChain(templateContext, ch);
    const labelInfo = !authInfo ? lookupTemplateLabelChain(templateContext, ch) : undefined;
    const info = authInfo ?? labelInfo;
    if (info) {
      const useLabel = !authInfo && Boolean(labelInfo);
      const chainLabel = useLabel ? info.labelChain : info.authChain;
      if (!resolvedChains.some((c) => c.chainLabel === chainLabel)) {
        resolvedChains.push({
          chainLabel,
          summary: summarizePolymerChainResidues(info, useLabel),
        });
      }
    } else {
      missingChain = true;
    }
  }

  let availPart: string;
  if (!missingChain && resolvedChains.length === 1) {
    availPart = `available Chain ${resolvedChains[0].chainLabel} residues: ${resolvedChains[0].summary}`;
  } else if (!missingChain && resolvedChains.length > 1) {
    availPart = `available residues — ${resolvedChains
      .map((c) => `Chain ${c.chainLabel}: ${c.summary}`)
      .join('; ')}`;
  } else {
    const allPoly = [...templateContext.polymerByAuthChain.values()];
    const chainSummaries = allPoly.map(
      (c) => `Chain ${c.authChain} (${summarizePolymerChainResidues(c)})`,
    );
    const ligCodes = [...new Set(templateContext.ligands.map((l) => l.compId))];
    const parts = [
      ...(chainSummaries.length > 0 ? [`chains: ${chainSummaries.join('; ')}`] : []),
      ...(ligCodes.length > 0 ? [`ligands: ${ligCodes.join(', ')}`] : []),
    ];
    availPart =
      parts.length > 0 ? `available ${parts.join(' · ')}` : 'template contains 0 polymer residues';
  }

  throw new Error(`${headPart} — ${availPart}`);
}

function isContigPureLabelSpace(
  templateContext: Rfd4TemplateContext,
  contig: string,
): boolean {
  const parts = contig
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  let sawPreservedPolymer = false;
  for (const tok of parts) {
    if (tok === '/0' || /^\d+(?:-\d+)?[PRDN]?$/i.test(tok)) continue;
    if (/^[A-Za-z0-9_]+:\d+(?:-\d+)?[PRDN]?$/i.test(tok)) {
      const prefix = tok.slice(0, tok.indexOf(':'));
      if (
        !lookupTemplateAuthChain(templateContext, prefix) &&
        !lookupTemplateLabelChain(templateContext, prefix)
      ) {
        continue;
      }
    }
    const chainLig = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(tok);
    if (chainLig && !/^-?\d+[A-Za-z]?$/.test(chainLig[2])) {
      // `A:SAM` is author/comp shorthand, not raw RFD4 label-space syntax
      return false;
    }
    const parsed = /^[A-Za-z]/.test(tok) || tok.includes(':') ? parseResidueRangeTokens(tok) : [];
    if (parsed.length !== 1) return false;
    const pt = parsed[0];
    if (pt.startInsCode || pt.endInsCode) return false;
    const labelPoly = lookupTemplateLabelChain(templateContext, pt.chain);
    if (
      labelPoly &&
      labelPoly.residues.some((r) => r.labelSeqId >= pt.start && r.labelSeqId <= pt.end)
    ) {
      sawPreservedPolymer = true;
      continue;
    }
    const ligHit = templateContext.ligands.some(
      (l) => l.labelChain === pt.chain && l.resId >= pt.start && l.resId <= pt.end,
    );
    if (ligHit) continue;
    return false;
  }
  return sawPreservedPolymer;
}

function extractTopLevelContigFromCustomSpec(rawSpecText: string): string | null {
  const trimmed = rawSpecText.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith('{')) {
    try {
      const obj = JSON.parse(trimmed) as Record<string, unknown>;
      if (typeof obj?.contig === 'string') return obj.contig.trim();
      if (obj && typeof obj === 'object' && !('input' in obj) && !('contig' in obj)) {
        const firstKey = Object.keys(obj).find((k) => k !== 'global_args');
        const sub = firstKey ? (obj[firstKey] as Record<string, unknown> | undefined) : undefined;
        if (typeof sub?.contig === 'string') return sub.contig.trim();
      }
    } catch {
      /* fall through to line scan */
    }
  }
  for (const rawLine of trimmed.split(/\r?\n/)) {
    const noComment = rawLine.replace(/(?:^|\s)#.*$/, '').trim();
    const m = /^contig\s*:\s*(.+)$/.exec(noComment);
    if (!m) continue;
    let val = m[1].trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1).trim();
    }
    return val || null;
  }
  return null;
}

function remapRfd4ConditionsDict(
  conditions: Record<string, unknown>,
  templateContext: Rfd4TemplateContext,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [condName, valueMap] of Object.entries(conditions)) {
    if (!valueMap || typeof valueMap !== 'object' || Array.isArray(valueMap)) {
      out[condName] = valueMap;
      continue;
    }
    const remappedValueMap: Record<string, unknown> = {};
    for (const [valKey, entries] of Object.entries(valueMap as Record<string, unknown>)) {
      if (!Array.isArray(entries)) {
        remappedValueMap[valKey] = entries;
        continue;
      }
      remappedValueMap[valKey] = entries.map((entry) => {
        if (entry && typeof entry === 'object' && typeof (entry as { select?: unknown }).select === 'string') {
          const prevSel = (entry as { select: string }).select;
          return {
            ...(entry as Record<string, unknown>),
            select: templateContext.remapHbondSelection(prevSel),
          };
        }
        return entry;
      });
    }
    out[condName] = remappedValueMap;
  }
  return out;
}

/**
 * Validate and remap a parsed RFD4 `customSpecYaml` object against `templateContext` (#97).
 * Handles PDB -> split-CIF `1..N` renumbering and author-numbered mmCIF templates while
 * preserving specs already authored in native mmCIF `label_asym_id` / `label_seq_id` space.
 */
export function remapAndValidateRfd4CustomSpec(
  parsedSpec: Record<string, unknown>,
  templateContext: Rfd4TemplateContext,
  opts?: { isFromPdb?: boolean; templateLabel?: string | null; exampleId?: string },
): Record<string, unknown> {
  if (!parsedSpec || typeof parsedSpec !== 'object') return parsedSpec;

  let specObj: Record<string, unknown> = { ...parsedSpec };
  if (!('input' in specObj) && !('contig' in specObj) && !('conditions' in specObj)) {
    const globalArgs =
      specObj.global_args && typeof specObj.global_args === 'object'
        ? (specObj.global_args as Record<string, unknown>)
        : {};
    const exampleKeys = Object.keys(specObj).filter((k) => k !== 'global_args');
    if (exampleKeys.length > 0) {
      const chosenKey =
        opts?.exampleId && specObj[opts.exampleId] ? opts.exampleId : exampleKeys[0];
      const chosen = specObj[chosenKey];
      if (chosen && typeof chosen === 'object') {
        specObj = { ...globalArgs, ...(chosen as Record<string, unknown>) };
      }
    }
  }

  const rawContig = typeof specObj.contig === 'string' ? specObj.contig.trim() : '';
  const alreadyLabelSpace =
    !opts?.isFromPdb && Boolean(rawContig) && isContigPureLabelSpace(templateContext, rawContig);

  if (rawContig) {
    validateRfd4TemplateSelectors(templateContext, {
      contig: rawContig,
      templateLabel: opts?.templateLabel,
      allowLabelSpaceFallback: alreadyLabelSpace,
    });
    if (!alreadyLabelSpace) {
      specObj.contig = templateContext.remapContig(rawContig);
    }
  }

  if (
    !alreadyLabelSpace &&
    specObj.conditions &&
    typeof specObj.conditions === 'object' &&
    !Array.isArray(specObj.conditions)
  ) {
    specObj.conditions = remapRfd4ConditionsDict(
      specObj.conditions as Record<string, unknown>,
      templateContext,
    );
  }

  return specObj;
}

/**
 * Single-source-of-truth fail-fast template selector validator shared by `rfd4.worker.ts`
 * (pre-weight-download) and `validateRuntimeNodeInputs` in `bio-capabilities.ts` (Layer 2 preflight).
 */
export function validateRfd4InputsAgainstTemplate(
  params: Record<string, unknown>,
  template?: StructureArtifact,
  condArtifact?: ConditionSpecArtifact,
  fallbackLabel?: string | null,
): void {
  const mode = String(params.mode ?? 'unconditional');
  if (mode === 'preset') return;
  if (mode === 'conditioned' && condArtifact?.cifUrl?.startsWith('preset:')) return;

  const customSpecYaml = String(params.customSpecYaml ?? '').trim();
  if (mode === 'unconditional' && !customSpecYaml) return;

  const rawStructureText =
    mode === 'partial' || mode === 'inverse-folding'
      ? (template?.cifText || template?.pdbText)
      : (condArtifact?.inputCif || template?.cifText || template?.pdbText);
  if (!rawStructureText) return;

  const isCif = /^\s*(data_|loop_|_atom_site\.)/m.test(rawStructureText);
  const templateContext = extractRfd4TemplateContext(rawStructureText);
  const templateLabel = template?.label || condArtifact?.label || fallbackLabel || null;
  const omitContig = mode === 'partial' || mode === 'inverse-folding';

  if (customSpecYaml) {
    const yamlContig = omitContig ? null : extractTopLevelContigFromCustomSpec(customSpecYaml);
    if (yamlContig) {
      const alreadyLabelSpace = !isCif ? false : isContigPureLabelSpace(templateContext, yamlContig);
      validateRfd4TemplateSelectors(templateContext, {
        contig: yamlContig,
        templateLabel,
        allowLabelSpaceFallback: alreadyLabelSpace,
      });
    }
    return;
  }

  if (condArtifact) {
    if (condArtifact.intent) {
      validateRfd4TemplateSelectors(templateContext, {
        contig: omitContig ? null : condArtifact.intent.contig,
        motifSpan: condArtifact.intent.motifSpans,
        motifAtoms: condArtifact.intent.motifAtoms,
        hotspots: condArtifact.intent.hotspots,
        templateLabel,
      });
    } else if (!omitContig && condArtifact.contig) {
      validateRfd4TemplateSelectors(templateContext, {
        contig: condArtifact.contig,
        templateLabel,
        allowLabelSpaceFallback: isCif,
      });
    }
    return;
  }

  const inlineContig = String(params.contig ?? '').trim();
  const inlineMotif = String(params.motifSpan ?? '').trim();
  const inlineMotifAtoms = String(params.motifAtoms ?? '').trim();
  const inlineHotspots = String(params.hotspots ?? '').trim();

  validateRfd4TemplateSelectors(templateContext, {
    contig: omitContig || isBareLengthContig(inlineContig) ? null : inlineContig,
    motifSpan: inlineMotif || null,
    motifAtoms: inlineMotifAtoms || null,
    hotspots: inlineHotspots || null,
    templateLabel,
  });
}

export interface MotifLineage {
  /** 1-based flat residue indices for downstream ProteinMPNN / AntiFold locking. */
  fixedResidueIndices: number[];
  /** 0-based residue indices into `caCoords` corresponding 1:1 with `nativeMotifCoords`. */
  motifIndices: number[];
  /** Optional native Cα [x, y, z] coordinates aligned 1:1 with `motifIndices`. */
  nativeMotifCoords?: number[][];
}

/**
 * Shared pure motif-lineage builder preventing 0-based vs 1-based drift (`CT-01`):
 * - `fixedResidueIndices`: 1-based flat residue indices for ProteinMPNN / AntiFold.
 * - `motifIndices`: 0-based indices into `StructureArtifact.caCoords` for `designability-scorer`.
 * - `nativeMotifCoords`: native template Cα [x, y, z] coordinates aligned 1:1 with `motifIndices`.
 */
export function buildMotifLineage(
  zeroBasedTargetIndices: readonly number[],
  sourceIndices?: readonly number[],
  upCa?: ArrayLike<number>,
): MotifLineage {
  const motifIndices: number[] = [];
  const fixedResidueIndices: number[] = [];
  const canExtractCoords =
    Boolean(sourceIndices) &&
    Boolean(upCa) &&
    sourceIndices!.length === zeroBasedTargetIndices.length &&
    sourceIndices!.length > 0;
  const nativeMotifCoords: number[][] | undefined = canExtractCoords ? [] : undefined;

  for (let k = 0; k < zeroBasedTargetIndices.length; k++) {
    const tIdx = Math.trunc(zeroBasedTargetIndices[k]);
    if (!Number.isInteger(tIdx) || tIdx < 0) continue;
    motifIndices.push(tIdx);
    fixedResidueIndices.push(tIdx + 1);
    if (nativeMotifCoords && sourceIndices && upCa) {
      const sIdx = Math.trunc(sourceIndices[k]);
      const base = sIdx * 3;
      if (Number.isInteger(sIdx) && sIdx >= 0 && base + 2 < upCa.length) {
        nativeMotifCoords.push([
          Number(upCa[base] ?? 0),
          Number(upCa[base + 1] ?? 0),
          Number(upCa[base + 2] ?? 0),
        ]);
      } else {
        nativeMotifCoords.push([0, 0, 0]);
      }
    }
  }

  return {
    fixedResidueIndices,
    motifIndices,
    ...(nativeMotifCoords && nativeMotifCoords.length === motifIndices.length && motifIndices.length > 0
      ? { nativeMotifCoords }
      : {}),
  };
}

export interface ExtractedDesignMetadata {
  lockedChains?: string[];
  fixedResidueIndices?: number[];
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
  motifIndices?: number[];
  nativeMotifCoords?: number[][];
  nativeMotifAtomCoords?: MotifAtomTarget[];
}

/**
 * Derive output `lockedChains`, 1-based `fixedResidueIndices`, `chainRoles`, 0-based `motifIndices`,
 * `nativeMotifCoords`, and atom-level `nativeMotifAtomCoords` from an RFD4 generated structure + input batch.
 */
export function extractRfd4DesignMetadata(
  parsed: { nResidues: number; chainIds: string[]; resNums: number[] },
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  structureAtoms: Array<Record<string, any>> | undefined,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  batch: Record<string, any> | undefined,
): ExtractedDesignMetadata {
  if (parsed.nResidues === 0) return {};

  const condSeqRaw = batch?.condition_sequence_1_residue?.data ?? batch?.condition_sequence_1_residue;

  // Group non-hetero protein atom records by residue (`chain:resId:insCode` or `tok:tokenId`),
  // retaining only residue buckets with an observed CA atom so the list aligns 1:1 with `parsed.nResidues`
  // even when nucleic acid chains are present.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const residueGroups: Array<Array<Record<string, any>>> = [];
  if (Array.isArray(structureAtoms)) {
    const groupByKey = new Map<string, number>();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rawGroups: Array<Array<Record<string, any>>> = [];
    for (const rec of structureAtoms) {
      if (rec?.isHetero) continue;
      let key: string;
      if (rec.chainId !== undefined && rec.resId !== undefined) {
        key = `res:${rec.chainId}:${rec.resId}:${rec.insCode ?? ''}`;
      } else if (typeof rec.tokenId === 'number') {
        key = `tok:${rec.tokenId}`;
      } else {
        key = `seq:${rawGroups.length}`;
      }
      let idx = groupByKey.get(key);
      if (idx === undefined) {
        idx = rawGroups.length;
        groupByKey.set(key, idx);
        rawGroups.push([]);
      }
      rawGroups[idx].push(rec);
    }
    for (const grp of rawGroups) {
      if (grp.some((r) => r.atomName === 'CA')) {
        residueGroups.push(grp);
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const extractNativeCoord = (rec: Record<string, any> | undefined): [number, number, number] | null => {
    if (!rec) return null;
    const rawCoord =
      (Array.isArray(rec.inCoord) && rec.inCoord.length === 3 ? rec.inCoord : null) ??
      (Array.isArray(rec.intCrd) && rec.intCrd.length === 3 ? rec.intCrd : null) ??
      (Array.isArray(rec.crd) && rec.crd.length === 3 ? rec.crd : null);
    if (rawCoord && rawCoord.every((v: unknown) => typeof v === 'number' && Number.isFinite(v))) {
      return [Number(rawCoord[0]), Number(rawCoord[1]), Number(rawCoord[2])];
    }
    return null;
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const isHeavyAtomRec = (rec: Record<string, any>): boolean => {
    const el = String(rec.element ?? '').trim().toUpperCase();
    if (el === 'H' || el === 'D') return false;
    const nm = String(rec.atomName ?? '').trim().toUpperCase();
    if (!nm) return false;
    if (!el && /^\d*H/.test(nm)) return false;
    return true;
  };

  const uniqueChains: string[] = [];
  const chainStats = new Map<string, { total: number; seqFixed: number; coordFixed: number }>();
  const perResFlags: Array<{
    chain: string;
    isSeqFixed: boolean;
    isMotifCandidate: boolean;
    nativeCaCoord: [number, number, number] | null;
    atomTargets: MotifAtomTarget[];
  }> = [];

  const alignedGroups = residueGroups.length === parsed.nResidues ? residueGroups : null;

  for (let i = 0; i < parsed.nResidues; i++) {
    const chain = parsed.chainIds[i] || 'A';
    if (!chainStats.has(chain)) {
      chainStats.set(chain, { total: 0, seqFixed: 0, coordFixed: 0 });
      uniqueChains.push(chain);
    }
    const stat = chainStats.get(chain)!;
    stat.total++;

    const resAtoms = alignedGroups ? alignedGroups[i] : [];
    const caRec = resAtoms.find((r) => r.atomName === 'CA');

    let hasSeqCond = false;
    let isUnidx = false;
    let hasPosCoordGroup = false;
    let hasZeroCoordGroup = false;
    const atomTargets: MotifAtomTarget[] = [];

    for (const rec of resAtoms) {
      const tid = typeof rec.tokenId === 'number' ? rec.tokenId : -1;
      if (tid >= 0 && condSeqRaw && Number(condSeqRaw[tid]) > 0) {
        hasSeqCond = true;
      }
      if (rec.isUnindexedInsertion === true) {
        isUnidx = true;
      }
      const cg = typeof rec.coordGroup === 'number' ? rec.coordGroup : -1;
      if (cg > 0) {
        hasPosCoordGroup = true;
        if (isHeavyAtomRec(rec)) {
          const c = extractNativeCoord(rec);
          if (c) {
            atomTargets.push({
              resIdx: i,
              atomName: String(rec.atomName).trim(),
              x: c[0],
              y: c[1],
              z: c[2],
            });
          }
        }
      } else if (cg === 0) {
        hasZeroCoordGroup = true;
      }
    }

    // P0-1: Unindexed catalytic insertions always have fixed sequence identity for downstream MPNN
    const isSeqFixed = hasSeqCond || isUnidx;
    const isCoordFixed = hasPosCoordGroup || hasZeroCoordGroup || isUnidx;
    if (isSeqFixed) stat.seqFixed++;
    if (isCoordFixed) stat.coordFixed++;

    const nativeCaCoord = extractNativeCoord(caRec);
    const isMotifCandidate =
      hasPosCoordGroup || isUnidx || (hasZeroCoordGroup && nativeCaCoord !== null);

    perResFlags.push({
      chain,
      isSeqFixed,
      isMotifCandidate,
      nativeCaCoord,
      atomTargets,
    });
  }

  const lockedChains: string[] = [];
  const chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> = {};

  if (uniqueChains.length > 1) {
    const anyFullyFixed = uniqueChains.some((ch) => {
      const s = chainStats.get(ch)!;
      return s.total > 0 && s.seqFixed === s.total;
    });
    for (const ch of uniqueChains) {
      const s = chainStats.get(ch)!;
      if (s.total > 0 && s.seqFixed === s.total) {
        lockedChains.push(ch);
        chainRoles[ch] = 'target';
      } else if (anyFullyFixed && s.seqFixed === 0 && s.coordFixed === 0) {
        chainRoles[ch] = 'binder';
      } else {
        chainRoles[ch] = 'scaffold';
      }
    }
  } else if (uniqueChains.length === 1) {
    chainRoles[uniqueChains[0]] = 'scaffold';
  }

  const lockedSet = new Set(lockedChains);
  const fixedResidueIndices: number[] = [];
  const motifIndices: number[] = [];
  const nativeMotifCoords: number[][] = [];
  const nativeMotifAtomCoords: MotifAtomTarget[] = [];

  for (let i = 0; i < perResFlags.length; i++) {
    const f = perResFlags[i];
    if (lockedSet.has(f.chain)) continue;
    if (f.isSeqFixed) {
      fixedResidueIndices.push(i + 1); // 1-based output index (sequence-fixed + unindexed catalytic insertions)
    }
    if (f.isMotifCandidate) {
      motifIndices.push(i); // 0-based index into caCoords
      if (f.nativeCaCoord) {
        nativeMotifCoords.push(f.nativeCaCoord);
      }
      if (f.atomTargets.length > 0) {
        nativeMotifAtomCoords.push(...f.atomTargets);
      }
    }
  }

  return {
    ...(lockedChains.length > 0 ? { lockedChains } : {}),
    ...(fixedResidueIndices.length > 0 ? { fixedResidueIndices } : {}),
    ...(Object.keys(chainRoles).length > 0 ? { chainRoles } : {}),
    ...(motifIndices.length > 0 ? { motifIndices } : {}),
    ...(nativeMotifCoords.length > 0 && nativeMotifCoords.length === motifIndices.length
      ? { nativeMotifCoords }
      : {}),
    ...(nativeMotifAtomCoords.length > 0 ? { nativeMotifAtomCoords } : {}),
  };
}

/**
 * When Genie 3 runs with `predictSequence: false`, upstream falls back to unmasked denoiser
 * logits (`ddimRes.seqLogits`), corrupting locked target/motif sequences.
 * This helper emits `'A'` for unconditioned positions and preserves `gtRestype` wherever
 * `condSeqMask > 0.5`.
 */
export function resolveGenie3PolyAlaSequence(
  length: number,
  compiled?: Pick<Genie3CompiledSpec, 'condSeqMask' | 'gtRestype'>,
): string {
  let seq = '';
  for (let i = 0; i < length; i++) {
    if (compiled && compiled.condSeqMask[i] > 0.5) {
      let bestIdx = -1;
      let bestVal = 0;
      const base = i * 20;
      for (let c = 0; c < 20; c++) {
        const v = compiled.gtRestype[base + c];
        if (v > bestVal) {
          bestVal = v;
          bestIdx = c;
        }
      }
      seq += bestIdx >= 0 ? GENIE3_RESTYPE_1[bestIdx] : 'A';
    } else {
      seq += 'A';
    }
  }
  return seq;
}

/** Exact effective DDIM step count produced by Genie 3's `getSamplingSteps(nTimestep, nSampleStep)`. */
export function effectiveGenie3DdimSteps(nSampleStep: number, nTimestep = 1000): number {
  const stepSize = Math.max(1, Math.floor(nTimestep / Math.max(1, Math.round(nSampleStep))));
  return Math.floor((nTimestep - 1) / stepSize) + 1;
}

/**
 * Convert LocalFold's vendor per-chain TM objects (`chainPtm`, `chainIptm`, `chainPairIptm` keyed
 * by `"0"`/`"0|1"` for AF2 or `"1"`/`"1|2"` for AF3) into canonical `FoldedCandidate` fields.
 */
export function normalizeLocalFoldChainScores(
  raw: {
    ptm?: number | null;
    iptm?: number | null;
    multimerScore?: number | null;
    chainPtm?: Record<string, number> | null;
    chainIptm?: Record<string, number> | null;
    chainPairIptm?: Record<string, number> | null;
  },
  uniqueChainIds: readonly string[],
  baseIndex: 0 | 1,
): {
  chainPtm?: Record<string, number>;
  chainIptm?: Record<string, number>;
  chainPairIptm?: number[][];
  multimerScore?: number;
} {
  const nChains = uniqueChainIds.length;
  const idxToChain = (rawKey: string): { idx: number; label: string } | null => {
    const num = Number(rawKey);
    if (!Number.isInteger(num)) return null;
    const idx = num - baseIndex;
    if (idx < 0 || idx >= nChains) return null;
    return { idx, label: uniqueChainIds[idx] };
  };

  let chainPtm: Record<string, number> | undefined;
  if (raw.chainPtm && typeof raw.chainPtm === 'object') {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(raw.chainPtm)) {
      const m = idxToChain(k);
      if (m && typeof v === 'number' && Number.isFinite(v)) out[m.label] = v;
    }
    if (Object.keys(out).length > 0) chainPtm = out;
  }

  let chainIptm: Record<string, number> | undefined;
  if (raw.chainIptm && typeof raw.chainIptm === 'object') {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(raw.chainIptm)) {
      const m = idxToChain(k);
      if (m && typeof v === 'number' && Number.isFinite(v)) out[m.label] = v;
    }
    if (Object.keys(out).length > 0) chainIptm = out;
  }

  let chainPairIptm: number[][] | undefined;
  if (nChains > 1 && raw.chainPairIptm && typeof raw.chainPairIptm === 'object') {
    const mat: number[][] = Array.from({ length: nChains }, (_, i) =>
      Array.from({ length: nChains }, (_, j) => (i === j ? (chainPtm?.[uniqueChainIds[i]] ?? 0) : 0)),
    );
    let anyPair = false;
    for (const [k, v] of Object.entries(raw.chainPairIptm)) {
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      const parts = k.split('|');
      if (parts.length !== 2) continue;
      const a = idxToChain(parts[0]);
      const b = idxToChain(parts[1]);
      if (!a || !b) continue;
      mat[a.idx][b.idx] = v;
      mat[b.idx][a.idx] = v;
      anyPair = true;
    }
    if (anyPair) chainPairIptm = mat;
  }

  let multimerScore: number | undefined;
  if (typeof raw.multimerScore === 'number' && Number.isFinite(raw.multimerScore)) {
    multimerScore = raw.multimerScore;
  } else if (
    nChains > 1 &&
    typeof raw.iptm === 'number' &&
    Number.isFinite(raw.iptm) &&
    typeof raw.ptm === 'number' &&
    Number.isFinite(raw.ptm)
  ) {
    multimerScore = 0.8 * raw.iptm + 0.2 * raw.ptm;
  }

  return {
    ...(chainPtm ? { chainPtm } : {}),
    ...(chainIptm ? { chainIptm } : {}),
    ...(chainPairIptm ? { chainPairIptm } : {}),
    ...(multimerScore !== undefined ? { multimerScore } : {}),
  };
}

/** Check whether a ConditionSpec can be compiled and executed by Genie 3 (protein motif / PPI binder). */
export function checkGenie3ConditionCompatibility(
  cond: ConditionSpecArtifact | undefined,
): { supported: boolean; reason?: string } {
  if (!cond) return { supported: true };
  if (cond.cifUrl?.startsWith('preset:')) {
    return {
      supported: false,
      reason: `RFD4 built-in preset (${cond.cifUrl.slice(7)}) uses RFD4-specific context archives — switch to Custom/Benchmark conditioning with a PDB Input template for Genie 3, or run with RFD4-Proteina.`,
    };
  }
  const intent = cond.intent;
  if (!intent) {
    return {
      supported: false,
      reason: 'Raw RFD4 JSON overrides cannot run on Genie 3 — use RFD4-Proteina or standard Conditioning fields.',
    };
  }
  if (
    (intent.hbondMode && intent.hbondMode !== 'none') ||
    intent.hbondDonorSelection ||
    intent.hbondAcceptorSelection ||
    intent.rasaBuriedSelection ||
    intent.piStackSelection
  ) {
    return {
      supported: false,
      reason: 'Hydrogen-bond and all-atom biochemical constraints require RFD4-Proteina (Genie 3 is Cα backbone diffusion).',
    };
  }
  if (intent.cathCode) {
    return {
      supported: false,
      reason: 'CATH fold-code conditioning is supported by RFD4-Proteina only.',
    };
  }
  if (intent.unindexedMotif) {
    return {
      supported: false,
      reason: 'Unindexed motif placement requires RFD4-Proteina — uncheck "Unindexed Motif" (indexed motif scaffolding) to run on Genie 3.',
    };
  }
  return { supported: true };
}

/**
 * Build an automatic ConditionSpec when a user wires a template PDB directly into Genie 3
 * without an explicit Conditioning block:
 * - If the template came from a curated benchmark, applies its curated hotspots / motif spec
 *   (adapting binder length to Genie 3's Length slider).
 * - Otherwise, designs a de novo binder of `defaultLength` aa against the template's primary chain.
 */
export function buildDefaultGenie3ConditionFromTemplate(
  template: StructureArtifact,
  defaultLength: number,
): ConditionSpecArtifact {
  const clampedLen = Math.max(20, Math.min(256, Math.round(defaultLength || 80)));

  if (template.benchmarkId) {
    const bench = getBenchmarkById(template.benchmarkId);
    if (bench) {
      if (!bench.supportsGenie3) {
        throw new Error(
          `${bench.shortLabel} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than Genie 3.`,
        );
      }
      const cp = bench.conditionParams;
      let contig = cp.contig;
      if (bench.category === 'binder' && contig.includes('/0')) {
        // Respect Genie 3's Length slider for the designed binder chain while keeping the curated target crop
        const parts = contig.split(/[,\s]+/).filter(Boolean);
        const updated = parts.map((tok) => (/^\d+(?:-\d+)?$/.test(tok) ? `${clampedLen}-${clampedLen}` : tok));
        contig = updated.join(',');
      }
      const intent = buildCanonicalConditionIntent({
        contig,
        lockChain: cp.lockChain ?? '',
        hotspots: cp.hotspots ?? '',
        motifSpan: cp.motifSpan ?? '',
        unindexedMotif: false,
        benchmarkId: bench.id,
        pdbId: bench.pdbId,
      });
      return {
        kind: 'ConditionSpec',
        id: `auto-g3-${bench.id}-L${clampedLen}`,
        label: bench.shortLabel,
        contig: intent.contig,
        conditions: {},
        intent,
        inputCif: template.cifText ?? template.pdbText,
        specMode: true,
        summary: `${bench.shortLabel} (auto)`,
      };
    }
  }

  // Generic template PDB: design a de novo binder against the template's first protein chain
  const firstChain = template.chainIds?.[0] || 'A';
  const intent = buildCanonicalConditionIntent({
    contig: `/0,${clampedLen}-${clampedLen}`,
    lockChain: firstChain,
  });
  return {
    kind: 'ConditionSpec',
    id: `auto-g3-binder-${template.id}-L${clampedLen}`,
    label: `Binder vs ${template.label}`,
    contig: intent.contig,
    conditions: {},
    intent,
    inputCif: template.cifText ?? template.pdbText,
    specMode: true,
    summary: `binder ${clampedLen} aa vs chain ${firstChain}`,
  };
}

export interface Genie3CompiledSpec {
  mode: 'unconditional' | 'binder' | 'motif';
  length: number;
  asymId: Int32Array;
  symId: Int32Array;
  entityId: Int32Array;
  residueIndex: Int32Array;
  condGroup: Int32Array;
  condSeqMask: Float32Array;
  condStructMask: Float32Array;
  condInterfaceMask: Float32Array;
  gtRestype: Float32Array;
  gtAtomPositions: Float32Array;
  chainLengths: number[];
  /** Explicit correspondence linking each selected motif/target residue back to its original source residue. */
  selectedSourceResidues: SelectedMotifSourceResidue[];
  summary: string;
}

function parseLengthRangeMid(tok: string): number | null {
  const m = /^(\d+)(?:-(\d+))?$/.exec(tok.trim());
  if (!m) return null;
  const lo = parseInt(m[1], 10);
  const hi = m[2] ? parseInt(m[2], 10) : lo;
  return Math.max(1, Math.round((lo + hi) / 2));
}

function writeRestypeOneHot(gtRestype: Float32Array, tokenIdx: number, aa1: string): void {
  const idx = GENIE3_RESTYPE_1.indexOf(aa1.toUpperCase());
  if (idx >= 0) {
    gtRestype[tokenIdx * 20 + idx] = 1;
  }
}

/**
 * Compile a canonical ConditionSpecArtifact + upstream template PDB/mmCIF into the exact
 * typed arrays expected by Genie 3's `createFeaturesFromSpec(...)`.
 */
export function compileConditionToGenie3Spec(
  cond: ConditionSpecArtifact,
  template: StructureArtifact | undefined,
  defaultLength: number,
): Genie3CompiledSpec {
  const compat = checkGenie3ConditionCompatibility(cond);
  if (!compat.supported) {
    throw new Error(compat.reason ?? 'Conditioning spec is not supported by Genie 3');
  }

  const intent = cond.intent!;
  const rawStructure = template?.cifText ?? template?.pdbText ?? cond.inputCif ?? '';

  // Pure length contig (no lockChain, no motifSpans, no preserved contig residues): unconditional monomer
  const hasLock = Boolean(intent.lockChain);
  const hasMotif = Boolean(
    (intent.motifSpans && intent.motifSpans.length > 0) || contigHasPreservedResidues(intent.contig),
  );
  if (!hasLock && !hasMotif) {
    const len = parseLengthRangeMid(intent.contig) ?? defaultLength;
    const clamped = Math.max(30, Math.min(256, len));
    const residueIndex = new Int32Array(clamped);
    for (let i = 0; i < clamped; i++) residueIndex[i] = i + 1;
    return {
      mode: 'unconditional',
      length: clamped,
      asymId: new Int32Array(clamped),
      symId: new Int32Array(clamped),
      entityId: new Int32Array(clamped),
      residueIndex,
      condGroup: new Int32Array(clamped),
      condSeqMask: new Float32Array(clamped),
      condStructMask: new Float32Array(clamped),
      condInterfaceMask: new Float32Array(clamped),
      gtRestype: new Float32Array(clamped * 20),
      gtAtomPositions: new Float32Array(clamped * 3),
      chainLengths: [clamped],
      selectedSourceResidues: [],
      summary: `unconditional L=${clamped}`,
    };
  }

  if (!rawStructure.trim()) {
    throw new Error(
      'Genie 3 motif/binder conditioning requires a template 3D structure — wire a PDB Input block into Conditioning (template) or Genie 3 (template).',
    );
  }

  const parsed = parseStructureText(rawStructure);
  if (parsed.nResidues < 3) {
    throw new Error('Template structure must contain at least 3 protein residues with Cα atoms.');
  }

  // Index template residues preserving full source identity (multi-char chains + insertion codes)
  interface TplRes {
    templateIndex: number;
    chain: string;
    labelChainId?: string;
    resNum: number;
    labelSeqId?: number;
    insCode: string;
    resName: string;
    projectedChainId: string;
    aa1: string;
    x: number;
    y: number;
    z: number;
  }
  const tplList: TplRes[] = [];
  const exactChains = new Set<string>();
  for (let i = 0; i < parsed.nResidues; i++) {
    const src = parsed.sourceResidues[i];
    const chain = src?.authChainId ?? parsed.chainIds[i] ?? 'A';
    exactChains.add(chain);
    const entry: TplRes = {
      templateIndex: i,
      chain,
      ...(src?.labelChainId ? { labelChainId: src.labelChainId } : {}),
      resNum: src?.resNum ?? parsed.resNums[i] ?? i + 1,
      ...(src?.labelSeqId !== undefined ? { labelSeqId: src.labelSeqId } : {}),
      insCode: src?.insCode ?? parsed.insCodes[i] ?? '',
      resName: src?.resName ?? (AA1_TO_AA3[parsed.sequence[i]?.toUpperCase() ?? 'X'] ?? 'UNK'),
      projectedChainId: src?.projectedChainId ?? parsed.projectedChainIds[i] ?? chain[0] ?? 'A',
      aa1: parsed.sequence[i] || 'A',
      x: parsed.caCoords[i * 3 + 0],
      y: parsed.caCoords[i * 3 + 1],
      z: parsed.caCoords[i * 3 + 2],
    };
    tplList.push(entry);
  }

  const selectTemplateResidues = (ranges: readonly ResidueRangeToken[]): TplRes[] => {
    const out: TplRes[] = [];
    for (const r of ranges) {
      for (const t of tplList) {
        if (
          chainMatchesExactOrFallback(t.chain, r.chain, exactChains) &&
          matchesResNumInsBounds(t.resNum, t.insCode, r.start, r.end, r.startInsCode, r.endInsCode)
        ) {
          out.push(t);
        }
      }
    }
    return out;
  };

  const toSelectedSource = (t: TplRes, outputIndex: number): SelectedMotifSourceResidue => ({
    outputIndex,
    templateIndex: t.templateIndex,
    authChainId: t.chain,
    ...(t.labelChainId ? { labelChainId: t.labelChainId } : {}),
    resNum: t.resNum,
    ...(t.labelSeqId !== undefined ? { labelSeqId: t.labelSeqId } : {}),
    insCode: t.insCode,
    resName: t.resName,
    aa1: t.aa1,
    projectedChainId: t.projectedChainId,
  });

  // ── Mode 1: PPI Binder Design (Genie 3 binder-first convention: chain 0 = binder, chain 1 = fixed target) ──
  if (hasLock) {
    const lockChain = intent.lockChain!.trim();
    const parts = intent.contig.split(/[,\s]+/).filter(Boolean);
    let binderLen = defaultLength;
    const targetRanges: ResidueRangeToken[] = [];

    for (const part of parts) {
      if (part === '/0') continue;
      const rangeTokens = parseResidueRangeTokens(part);
      if (rangeTokens.length > 0 && (/^[A-Za-z]/.test(part) || part.includes(':'))) {
        targetRanges.push(...rangeTokens);
      } else {
        const mid = parseLengthRangeMid(part);
        if (mid !== null) binderLen = mid;
      }
    }

    let targetRes: TplRes[] = [];
    if (targetRanges.length > 0) {
      targetRes = selectTemplateResidues(targetRanges);
    } else {
      targetRes = tplList.filter((t) => chainMatchesExactOrFallback(t.chain, lockChain, exactChains));
    }

    if (targetRes.length < 3) {
      throw new Error(
        `Could not find target residues for chain "${lockChain}" in the wired template structure.`,
      );
    }

    const hotspotRanges = parseResidueRangeTokens((intent.hotspots ?? []).join(', '));

    const clampedBinder = Math.max(20, Math.min(256, binderLen));
    const totalLen = clampedBinder + targetRes.length;
    if (totalLen > 384) {
      throw new Error(
        `Total binder + target length (${totalLen} aa) exceeds Genie 3 browser WebGPU limit (384 aa). Crop the target range in the contig.`,
      );
    }

    const asymId = new Int32Array(totalLen);
    const symId = new Int32Array(totalLen);
    const entityId = new Int32Array(totalLen);
    const residueIndex = new Int32Array(totalLen);
    const condGroup = new Int32Array(totalLen);
    const condSeqMask = new Float32Array(totalLen);
    const condStructMask = new Float32Array(totalLen);
    const condInterfaceMask = new Float32Array(totalLen);
    const gtRestype = new Float32Array(totalLen * 20);
    const gtAtomPositions = new Float32Array(totalLen * 3);

    // Compute target centroid so fixed coordinates sit cleanly at the origin
    let cx = 0, cy = 0, cz = 0;
    for (const t of targetRes) {
      cx += t.x; cy += t.y; cz += t.z;
    }
    cx /= targetRes.length; cy /= targetRes.length; cz /= targetRes.length;

    // Chain 0: designed binder (unconditioned)
    for (let i = 0; i < clampedBinder; i++) {
      asymId[i] = 0;
      entityId[i] = 0;
      residueIndex[i] = i + 1;
    }

    // Chain 1: fixed target with hotspots
    let hotspotMatches = 0;
    const selectedSourceResidues: SelectedMotifSourceResidue[] = [];
    for (let j = 0; j < targetRes.length; j++) {
      const idx = clampedBinder + j;
      const t = targetRes[j];
      asymId[idx] = 1;
      entityId[idx] = 1;
      residueIndex[idx] = j + 1;
      condGroup[idx] = 1;
      condStructMask[idx] = 1;
      condSeqMask[idx] = 1;
      const isHotspot = hotspotRanges.some(
        (h) =>
          chainMatchesExactOrFallback(t.chain, h.chain, exactChains) &&
          matchesResNumInsBounds(t.resNum, t.insCode, h.start, h.end, h.startInsCode, h.endInsCode),
      );
      if (isHotspot) {
        condInterfaceMask[idx] = 1;
        hotspotMatches++;
      }
      writeRestypeOneHot(gtRestype, idx, t.aa1);
      gtAtomPositions[idx * 3 + 0] = t.x - cx;
      gtAtomPositions[idx * 3 + 1] = t.y - cy;
      gtAtomPositions[idx * 3 + 2] = t.z - cz;
      selectedSourceResidues.push(toSelectedSource(t, idx));
    }

    return {
      mode: 'binder',
      length: totalLen,
      asymId,
      symId,
      entityId,
      residueIndex,
      condGroup,
      condSeqMask,
      condStructMask,
      condInterfaceMask,
      gtRestype,
      gtAtomPositions,
      chainLengths: [clampedBinder, targetRes.length],
      selectedSourceResidues,
      summary: `binder ${clampedBinder} aa + target ${targetRes.length} aa (${hotspotMatches} hotspots)`,
    };
  }

  // ── Mode 2: Indexed Motif Scaffolding (single chain with fixed motif segments) ──
  const contigParts = intent.contig.split(/[,\s]+/).filter((s) => s && s !== '/0');
  type Segment = { kind: 'scaffold'; len: number } | { kind: 'motif'; residues: TplRes[] };
  const segments: Segment[] = [];
  let hasExplicitMotifInContig = false;

  for (const part of contigParts) {
    if (/^[A-Za-z]/.test(part) || part.includes(':')) {
      const ranges = parseResidueRangeTokens(part);
      if (ranges.length === 0) {
        throw new Error(`Invalid Genie 3 motif contig token "${part}".`);
      }
      const res = selectTemplateResidues(ranges);
      if (res.length === 0) {
        throw new Error(
          `Genie 3 motif contig span "${part}" matched 0 residues in the wired template structure.`,
        );
      }
      hasExplicitMotifInContig = true;
      segments.push({ kind: 'motif', residues: res });
    } else {
      const mid = parseLengthRangeMid(part);
      if (mid !== null) segments.push({ kind: 'scaffold', len: mid });
    }
  }

  // If contig didn't explicitly include the motif spans (e.g. contig="80-80" + motifSpan="A16-35"),
  // build balanced scaffold flanking segments around the motif spans!
  if (!hasExplicitMotifInContig) {
    const motifGroups = parseResidueRangeTokens((intent.motifSpans ?? []).join(', '));
    const motifBlocks: TplRes[][] = [];
    const unmatchedSpans: string[] = [];
    for (const r of motifGroups) {
      const res = selectTemplateResidues([r]);
      if (res.length > 0) {
        motifBlocks.push(res);
      } else {
        unmatchedSpans.push(`${r.chain}${r.start}${r.startInsCode ?? ''}${r.end !== r.start || r.endInsCode !== r.startInsCode ? `-${r.end}${r.endInsCode ?? ''}` : ''}`);
      }
    }
    if (unmatchedSpans.length > 0) {
      throw new Error(
        `Genie 3 motif span(s) matched 0 residues in the wired template structure: ${unmatchedSpans.join(', ')}.`,
      );
    }
    if (motifBlocks.length === 0) {
      throw new Error(
        `Could not locate motif residues (${(intent.motifSpans ?? []).join(', ')}) in the wired template structure.`,
      );
    }
    const targetTotal = parseLengthRangeMid(intent.contig) ?? defaultLength;
    const motifTotal = motifBlocks.reduce((s, b) => s + b.length, 0);
    const remaining = Math.max(motifBlocks.length + 1, targetTotal - motifTotal);
    const gapSlots = motifBlocks.length + 1;
    const baseGap = Math.max(4, Math.floor(remaining / gapSlots));
    segments.length = 0;
    for (let i = 0; i < motifBlocks.length; i++) {
      segments.push({ kind: 'scaffold', len: baseGap });
      segments.push({ kind: 'motif', residues: motifBlocks[i] });
    }
    segments.push({ kind: 'scaffold', len: Math.max(4, remaining - baseGap * motifBlocks.length) });
  }

  const totalLen = segments.reduce(
    (s, seg) => s + (seg.kind === 'scaffold' ? seg.len : seg.residues.length),
    0,
  );
  if (totalLen < 10 || totalLen > 384) {
    throw new Error(`Motif scaffold length (${totalLen} aa) must be between 10 and 384 aa for Genie 3.`);
  }

  const motifResAll = segments.flatMap((s) => (s.kind === 'motif' ? s.residues : []));
  if (motifResAll.length === 0) {
    throw new Error('No matching motif residues found in the wired template structure.');
  }

  let cx = 0, cy = 0, cz = 0;
  for (const m of motifResAll) {
    cx += m.x; cy += m.y; cz += m.z;
  }
  cx /= motifResAll.length; cy /= motifResAll.length; cz /= motifResAll.length;

  const asymId = new Int32Array(totalLen);
  const symId = new Int32Array(totalLen);
  const entityId = new Int32Array(totalLen);
  const residueIndex = new Int32Array(totalLen);
  const condGroup = new Int32Array(totalLen);
  const condSeqMask = new Float32Array(totalLen);
  const condStructMask = new Float32Array(totalLen);
  const condInterfaceMask = new Float32Array(totalLen);
  const gtRestype = new Float32Array(totalLen * 20);
  const gtAtomPositions = new Float32Array(totalLen * 3);
  const selectedSourceResidues: SelectedMotifSourceResidue[] = [];

  let cursor = 0;
  for (const seg of segments) {
    if (seg.kind === 'scaffold') {
      for (let k = 0; k < seg.len; k++) {
        residueIndex[cursor] = cursor + 1;
        cursor++;
      }
    } else {
      for (const m of seg.residues) {
        residueIndex[cursor] = cursor + 1;
        condGroup[cursor] = 1;
        condStructMask[cursor] = 1;
        condSeqMask[cursor] = 1;
        writeRestypeOneHot(gtRestype, cursor, m.aa1);
        gtAtomPositions[cursor * 3 + 0] = m.x - cx;
        gtAtomPositions[cursor * 3 + 1] = m.y - cy;
        gtAtomPositions[cursor * 3 + 2] = m.z - cz;
        selectedSourceResidues.push(toSelectedSource(m, cursor));
        cursor++;
      }
    }
  }

  return {
    mode: 'motif',
    length: totalLen,
    asymId,
    symId,
    entityId,
    residueIndex,
    condGroup,
    condSeqMask,
    condStructMask,
    condInterfaceMask,
    gtRestype,
    gtAtomPositions,
    chainLengths: [totalLen],
    selectedSourceResidues,
    summary: `motif scaffold L=${totalLen} (${motifResAll.length} fixed motif aa)`,
  };
}

// ---------------------------------------------------------------------------
// Unified pure synchronous input resolvers shared by GPU workers & Input Preview
// ---------------------------------------------------------------------------

function fnv1aHex(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

export type Rfd4CompiledKind =
  | 'preset'
  | 'spec'
  | 'custom-yaml'
  | 'template-spec'
  | 'template-custom-yaml';

export interface Rfd4CompiledInput {
  kind: Rfd4CompiledKind;
  mode:
    | 'unconditional'
    | 'binder'
    | 'motif'
    | 'conditioned'
    | 'preset'
    | 'partial'
    | 'inverse-folding';
  length: number;
  nsteps: number;
  recycleSchedule: 'none' | 'low' | 'medium' | 'high';
  earlyStop: boolean;
  seed: number;
  partialFraction: number;
  activePresetId: string | null;
  needEncoder: boolean;
  inverseFolding: boolean;
  effectivePartialFraction: number | null;
  runLabel: string;
  contig: string | null;
  conditions: Record<string, unknown>;
  customSpecYaml: string;
  tags: string[];
  intent?: CanonicalConditionIntent;
  selectedSourceResidues?: SelectedMotifSourceResidue[];
  templateContext?: Rfd4TemplateContext;
  usesTemplate: boolean;
}

/**
 * Pure synchronous RFD4-Proteina mode-branching & conditioning compiler.
 * Single source of truth shared by `rfd4.worker.ts` and `input-preview.ts`.
 */
export function resolveRfd4CompiledInput(
  params: Record<string, unknown>,
  template?: StructureArtifact,
  condArtifact?: ConditionSpecArtifact,
  templateContextOverride?: Rfd4TemplateContext,
): Rfd4CompiledInput {
  const mode = String(params.mode ?? 'unconditional');
  if (
    mode !== 'unconditional' &&
    mode !== 'binder' &&
    mode !== 'motif' &&
    mode !== 'conditioned' &&
    mode !== 'preset' &&
    mode !== 'partial' &&
    mode !== 'inverse-folding'
  ) {
    throw new Error(`Unknown RFD4 mode "${mode}"`);
  }

  const length = Math.max(30, Math.min(256, Number(params.length ?? 70)));
  const nsteps = Math.max(10, Math.min(120, Number(params.nsteps ?? 60)));
  const recycleSchedule = (params.recycleSchedule ?? 'low') as 'none' | 'low' | 'medium' | 'high';
  const requestedEarlyStop = Boolean(params.earlyStop);
  const earlyStop = mode === 'inverse-folding' ? false : requestedEarlyStop;
  const seed = Number(params.seed ?? 42);
  const customSpecYaml = String(params.customSpecYaml ?? '').trim();
  const partialFraction = Math.max(0.05, Math.min(0.95, Number(params.partialFraction ?? 0.5)));
  const inlineContig = String(params.contig ?? '').trim();
  const inlineLock = String(params.lockChain ?? '').trim();
  const inlineHotspots = String(params.hotspots ?? '').trim();
  const inlineMotif = String(params.motifSpan ?? '').trim();
  const inlineMotifAtoms = String(params.motifAtoms ?? '').trim();
  const inlineUnidx = Boolean(params.unindexedMotif);
  const inlineHbondMode = String(params.hbondMode ?? 'none');
  const inlineHbondSel = String(params.hbondSelection ?? '').trim();
  const inlineHbondDonorSel = String(params.hbondDonorSelection ?? '').trim();
  const inlineHbondAcceptorSel = String(params.hbondAcceptorSelection ?? '').trim();
  const inlineRasaBuriedSel = String(params.rasaBuriedSelection ?? '').trim();
  const inlinePiStackSel = String(params.piStackSelection ?? '').trim();
  const inlineCath = String(params.cathCode ?? '').trim();

  const hasTemplateStructure = Boolean(
    template?.cifText || template?.pdbText || condArtifact?.inputCif || condArtifact?.cifUrl,
  );
  if (
    (mode === 'motif' || mode === 'binder' || mode === 'partial' || mode === 'inverse-folding') &&
    !hasTemplateStructure
  ) {
    throw new Error(
      `RFD4 "${mode}" mode requires a wired structure input (e.g. from PDB Input).`,
    );
  }

  const activePresetId =
    mode === 'preset'
      ? String(params.presetId ?? 'custom_cath_monomer')
      : mode === 'conditioned' && condArtifact?.cifUrl?.startsWith('preset:')
        ? condArtifact.cifUrl.slice(7)
        : null;

  const needEncoder =
    mode === 'partial' ||
    mode === 'inverse-folding' ||
    activePresetId === 'partial_diffusion_1prw' ||
    activePresetId === 'inverse_folding_1prw' ||
    (mode === 'conditioned' &&
      (Boolean(condArtifact?.inverseFolding) ||
        (condArtifact?.partialFraction != null && condArtifact.partialFraction > 0)));

  const rawTemplateText =
    mode === 'partial' || mode === 'inverse-folding'
      ? (template?.cifText || template?.pdbText)
      : (condArtifact?.inputCif || template?.cifText || template?.pdbText);

  const ensureTemplateContext = (): Rfd4TemplateContext | undefined => {
    if (templateContextOverride) return templateContextOverride;
    if (!rawTemplateText) return undefined;
    return extractRfd4TemplateContext(rawTemplateText);
  };

  const base = {
    mode,
    length,
    nsteps,
    recycleSchedule,
    earlyStop,
    seed,
    partialFraction,
    activePresetId,
    needEncoder,
    inverseFolding: mode === 'inverse-folding',
    effectivePartialFraction: mode === 'partial' ? partialFraction : null,
    customSpecYaml,
  } as const;

  if (mode === 'preset') {
    return {
      ...base,
      kind: 'preset',
      runLabel: `RFD4 · ${activePresetId!}`,
      contig: null,
      conditions: {},
      tags: [`preset:${activePresetId!}`],
      usesTemplate: false,
    };
  }

  if (mode === 'binder' || mode === 'motif' || mode === 'conditioned') {
    if (condArtifact) {
      if (activePresetId) {
        const effPartial =
          condArtifact.partialFraction != null && condArtifact.partialFraction > 0
            ? condArtifact.partialFraction
            : null;
        const effInvFold = Boolean(condArtifact.inverseFolding);
        return {
          ...base,
          kind: 'preset',
          earlyStop: effInvFold ? false : earlyStop,
          inverseFolding: effInvFold,
          effectivePartialFraction: effPartial,
          runLabel: `RFD4 · ${activePresetId}`,
          contig: null,
          conditions: {},
          tags: [`preset:${activePresetId}`],
          usesTemplate: false,
        };
      }

      const templateContext = ensureTemplateContext();
      const effPartial =
        condArtifact.partialFraction != null && condArtifact.partialFraction > 0
          ? condArtifact.partialFraction
          : null;
      const effInvFold = Boolean(condArtifact.inverseFolding);
      const compiled = condArtifact.intent
        ? compileConditionToRfd4(condArtifact.intent, templateContext)
        : {
            contig: condArtifact.contig,
            conditions: condArtifact.conditions ?? {},
            tags: [condArtifact.contig],
          };
      return {
        ...base,
        kind: 'spec',
        earlyStop: effInvFold ? false : earlyStop,
        inverseFolding: effInvFold,
        effectivePartialFraction: effPartial,
        runLabel: `RFD4 · ${condArtifact.label}`,
        contig: compiled.contig,
        conditions: compiled.conditions,
        tags: compiled.tags,
        ...(condArtifact.intent ? { intent: condArtifact.intent } : {}),
        ...(compiled.selectedSourceResidues
          ? { selectedSourceResidues: compiled.selectedSourceResidues }
          : {}),
        ...(templateContext ? { templateContext } : {}),
        usesTemplate: Boolean(templateContext || condArtifact.cifUrl),
      };
    }

    if (customSpecYaml) {
      const templateContext = ensureTemplateContext();
      return {
        ...base,
        kind: 'custom-yaml',
        runLabel: 'RFD4 · custom spec',
        contig: null,
        conditions: {},
        tags: ['custom-spec'],
        ...(templateContext ? { templateContext } : {}),
        usesTemplate: Boolean(templateContext),
      };
    }

    // Inline conditioning directly from RFD4-Proteina node parameters
    const templateContext = ensureTemplateContext();
    const defaultChain = template?.chainIds?.[0] || 'A';
    const effectiveLock = mode === 'binder' ? inlineLock || defaultChain : inlineLock;

    let effectiveContig = inlineContig;
    let synthesizedContig = false;
    if (mode === 'binder' && isBareLengthContig(effectiveContig)) {
      if (!templateContext) {
        throw new Error(
          'RFD4 binder mode requires a wired target structure (e.g. from PDB Input).',
        );
      }
      const synth = templateContext.synthesizeBinderContig(effectiveLock, length);
      if (!synth) {
        throw new Error(
          `Could not synthesize binder contig for target chain "${effectiveLock}" from wired structure.`,
        );
      }
      effectiveContig = synth;
      synthesizedContig = true;
    } else if (!effectiveContig) {
      effectiveContig = `${length}-${length}`;
    }

    const intent = buildCanonicalConditionIntent({
      contig: effectiveContig,
      lockChain: effectiveLock,
      hotspots: inlineHotspots,
      motifSpan: inlineMotif,
      motifAtoms: inlineMotifAtoms,
      unindexedMotif: inlineUnidx,
      hbondMode: inlineHbondMode,
      hbondSelection: inlineHbondSel,
      hbondDonorSelection: inlineHbondDonorSel,
      hbondAcceptorSelection: inlineHbondAcceptorSel,
      rasaBuriedSelection: inlineRasaBuriedSel,
      piStackSelection: inlinePiStackSel,
      cathCode: inlineCath,
    });
    const compiled = compileConditionToRfd4(intent, templateContext, {
      contigAlreadyLabelSpace: synthesizedContig,
    });
    return {
      ...base,
      kind: 'spec',
      runLabel: `RFD4 · ${compiled.tags.slice(0, 2).join(' · ')}`,
      contig: compiled.contig,
      conditions: compiled.conditions,
      tags: compiled.tags,
      intent,
      ...(compiled.selectedSourceResidues
        ? { selectedSourceResidues: compiled.selectedSourceResidues }
        : {}),
      ...(templateContext ? { templateContext } : {}),
      usesTemplate: Boolean(templateContext),
    };
  }

  if (mode === 'partial' || mode === 'inverse-folding') {
    const templateContext = ensureTemplateContext();
    if (!templateContext) {
      throw new Error(
        `RFD4 "${mode}" mode requires a wired structure input (e.g. from PDB Input)`,
      );
    }
    const runLabel =
      mode === 'partial' ? `RFD4 partial (frac=${partialFraction})` : 'RFD4 inverse-fold';
    const hasInlineCond =
      Boolean(inlineLock) ||
      Boolean(inlineHotspots) ||
      Boolean(inlineMotif) ||
      Boolean(inlineMotifAtoms) ||
      (inlineHbondMode !== 'none' && Boolean(inlineHbondSel)) ||
      Boolean(inlineHbondDonorSel) ||
      Boolean(inlineHbondAcceptorSel) ||
      Boolean(inlineRasaBuriedSel) ||
      Boolean(inlinePiStackSel) ||
      Boolean(inlineCath);

    if (customSpecYaml) {
      return {
        ...base,
        kind: 'template-custom-yaml',
        runLabel,
        contig: null,
        conditions: {},
        tags: ['input:template', 'custom-spec'],
        templateContext,
        usesTemplate: true,
      };
    }

    if (condArtifact?.intent || hasInlineCond) {
      const intent =
        condArtifact?.intent ??
        buildCanonicalConditionIntent({
          contig: inlineContig || '70-70',
          lockChain: inlineLock,
          hotspots: inlineHotspots,
          motifSpan: inlineMotif,
          motifAtoms: inlineMotifAtoms,
          unindexedMotif: false,
          hbondMode: inlineHbondMode,
          hbondSelection: inlineHbondSel,
          hbondDonorSelection: inlineHbondDonorSel,
          hbondAcceptorSelection: inlineHbondAcceptorSel,
          rasaBuriedSelection: inlineRasaBuriedSel,
          piStackSelection: inlinePiStackSel,
          cathCode: inlineCath,
        });
      const compiled = compileConditionToRfd4(intent, templateContext, {
        omitContig: true,
      });
      return {
        ...base,
        kind: 'template-spec',
        runLabel,
        contig: null,
        conditions: compiled.conditions,
        tags: ['input:template', ...compiled.tags.slice(1)],
        intent,
        ...(compiled.selectedSourceResidues
          ? { selectedSourceResidues: compiled.selectedSourceResidues }
          : {}),
        templateContext,
        usesTemplate: true,
      };
    }

    return {
      ...base,
      kind: 'template-spec',
      runLabel,
      contig: null,
      conditions: {},
      tags: ['input:template'],
      templateContext,
      usesTemplate: true,
    };
  }

  if (customSpecYaml) {
    const templateContext = ensureTemplateContext();
    return {
      ...base,
      kind: 'custom-yaml',
      runLabel: 'RFD4 custom spec',
      contig: null,
      conditions: {},
      tags: ['custom-spec'],
      ...(templateContext ? { templateContext } : {}),
      usesTemplate: Boolean(templateContext),
    };
  }

  if (inlineCath) {
    const intent = buildCanonicalConditionIntent({
      contig: `${length}-${length}`,
      cathCode: inlineCath,
    });
    const compiled = compileConditionToRfd4(intent);
    return {
      ...base,
      kind: 'spec',
      runLabel: `RFD4 · CATH ${inlineCath}`,
      contig: compiled.contig,
      conditions: compiled.conditions,
      tags: compiled.tags,
      intent,
      usesTemplate: false,
    };
  }

  return {
    ...base,
    kind: 'spec',
    runLabel: `RFD4 (L=${length})`,
    contig: `${length}-${length}`,
    conditions: {},
    tags: [`${length}-${length}`],
    usesTemplate: false,
  };
}

export interface Genie3CompiledInput {
  mode: 'unconditional' | 'binder' | 'motif';
  taskMode: string;
  length: number;
  nSampleStep: number;
  effectiveSteps: number;
  directionScale: number;
  noiseScale: number;
  predictSequence: boolean;
  seed: number;
  effectiveCond?: ConditionSpecArtifact;
  compiledSpec?: Genie3CompiledSpec;
  totalLength: number;
  chainLengths: number[];
  summary: string;
}

/**
 * Pure synchronous Genie 3 mode-branching, validation, and feature-spec compiler.
 * Single source of truth shared by `genie3.worker.ts` and `input-preview.ts`.
 */
export function resolveGenie3CompiledInput(
  params: Record<string, unknown>,
  template?: StructureArtifact,
  condArtifact?: ConditionSpecArtifact,
): Genie3CompiledInput {
  const length = Math.max(30, Math.min(256, Math.round(Number(params.length ?? 80))));
  const nSampleStep = Math.max(10, Math.min(250, Math.round(Number(params.nSampleStep ?? 100))));
  const directionScale = Math.max(0.0, Math.min(1.5, Number(params.directionScale ?? 0.8)));
  const noiseScale = Math.max(0, Math.min(1.5, Number(params.noiseScale ?? 1.0)));
  const predictSequence = params.predictSequence !== false;
  const seed = Math.max(0, Math.round(Number(params.seed ?? 42)));
  const taskMode = String(params.taskMode ?? 'unconditional');
  const inlineContig = String(params.contig ?? '').trim();
  const inlineLock = String(params.lockChain ?? '').trim();
  const inlineHotspots = String(params.hotspots ?? '').trim();
  const inlineMotif = String(params.motifSpan ?? '').trim();

  // Fail fast BEFORE downloading 45 MB weights if required motif/binder inputs are missing
  if (!condArtifact && taskMode === 'motif' && !inlineMotif && !inlineContig) {
    throw new Error(
      'Genie 3 motif mode requires a motifSpan (e.g. A16-35) or contig specifying motif residues.',
    );
  }
  if (
    taskMode !== 'unconditional' &&
    (taskMode === 'motif' || taskMode === 'binder' || Boolean(inlineLock) || Boolean(inlineMotif)) &&
    !template?.cifText &&
    !template?.pdbText &&
    !condArtifact?.inputCif
  ) {
    throw new Error(
      `Genie 3 "${taskMode}" mode requires a wired structure input (e.g. from PDB Input).`,
    );
  }

  // Guard against explicitly incompatible all-atom benchmarks or non-protein (<3 Cα) templates.
  // Unified Primitive 2 policy: structures with >=3 protein Cα residues + uncropped HETATMs/DNA
  // compile cleanly because Genie 3 ignores non-protein records cleanly.
  if (template && taskMode !== 'unconditional') {
    const bench = template.benchmarkId ? getBenchmarkById(template.benchmarkId) : undefined;
    const inferredPdbId =
      bench?.pdbId ??
      (/\b([0-9][A-Za-z0-9]{3})\b/.exec(template.label ?? '')?.[1]?.toUpperCase() || undefined);
    const policy = evaluateProteinOnlyStructurePolicy({
      modelName: 'Genie 3',
      benchmark: bench,
      pdbId: inferredPdbId,
      proteinResidueCount: template.length,
      ligandNames: template.diagnostics?.ligandNames,
      nucleicAcidChains: template.diagnostics?.nucleicAcidChains,
      hasLigands: template.diagnostics?.hasLigands,
      hasNucleicAcids: template.diagnostics?.hasNucleicAcids,
    });
    if (policy.action === 'hard-block') {
      throw new Error(policy.message);
    }
  }

  let effectiveCond: ConditionSpecArtifact | undefined =
    taskMode === 'unconditional' ? condArtifact : condArtifact;
  if (
    taskMode !== 'unconditional' &&
    !effectiveCond &&
    (taskMode === 'binder' || taskMode === 'motif' || inlineLock || inlineMotif)
  ) {
    // Ignore stale binder-only lockChain when taskMode is explicitly 'motif'
    const isBinder = taskMode === 'binder' || (taskMode !== 'motif' && Boolean(inlineLock));
    const defaultChain = template?.chainIds?.[0] || 'A';
    const resolvedLock = isBinder ? inlineLock || defaultChain : '';
    const resolvedContig =
      inlineContig || (isBinder ? `/0,${length}-${length}` : `${length}-${length}`);
    const intent = buildCanonicalConditionIntent({
      contig: resolvedContig,
      lockChain: resolvedLock,
      hotspots: isBinder ? inlineHotspots : '',
      motifSpan: inlineMotif,
    });
    effectiveCond = {
      kind: 'ConditionSpec',
      id: `inline-g3-${fnv1aHex(JSON.stringify({ intent, templateId: template?.id }))}`,
      label: isBinder ? `Binder (${resolvedLock})` : 'Motif Scaffold',
      contig: intent.contig,
      conditions: {},
      intent,
      inputCif: template?.cifText ?? template?.pdbText,
      specMode: true,
      summary: isBinder ? `binder vs ${resolvedLock}` : `motif ${inlineMotif}`,
    };
  } else if (taskMode !== 'unconditional' && !effectiveCond && template) {
    effectiveCond = buildDefaultGenie3ConditionFromTemplate(template, length);
  }

  const compiledSpec = effectiveCond
    ? compileConditionToGenie3Spec(effectiveCond, template, length)
    : undefined;
  if (
    taskMode === 'motif' &&
    (!compiledSpec || compiledSpec.mode !== 'motif' || compiledSpec.selectedSourceResidues.length === 0)
  ) {
    throw new Error(
      'Genie 3 motif mode requires at least 1 selected motif residue (provide motifSpan or a contig containing template residue spans like 15-15,A16-35,15-15).',
    );
  }
  const effectiveSteps = effectiveGenie3DdimSteps(nSampleStep);

  return {
    mode: compiledSpec?.mode ?? 'unconditional',
    taskMode,
    length,
    nSampleStep,
    effectiveSteps,
    directionScale,
    noiseScale,
    predictSequence,
    seed,
    ...(effectiveCond ? { effectiveCond } : {}),
    ...(compiledSpec ? { compiledSpec } : {}),
    totalLength: compiledSpec?.length ?? length,
    chainLengths: compiledSpec?.chainLengths ?? [length],
    summary: compiledSpec?.summary ?? `unconditional L=${length}`,
  };
}

// ---------------------------------------------------------------------------
// Shared LocalFold Effective Config Resolver (CT-05: 1:1 Worker <-> Preview)
// ---------------------------------------------------------------------------

export const LOCALFOLD_AF2_MODELS: ReadonlySet<string> = new Set([
  'monomer',
  'monomer-2',
  'monomer-3',
  'monomer-4',
  'monomer-5',
]);

export const LOCALFOLD_AF2_MULTIMER_MODELS: ReadonlySet<string> = new Set([
  'multimer',
  'multimer-2',
  'multimer-3',
  'multimer-4',
  'multimer-5',
]);

export const LOCALFOLD_ESMFOLD2_MODELS: ReadonlySet<string> = new Set([
  'ef2-fast-300m',
  'ef2-fast-600m',
]);

export const LOCALFOLD_AF3_MODELS: ReadonlySet<string> = new Set([
  'rosettafold3',
  'boltz2',
  'openbind0',
  'protenix2',
  'opendde',
  'mol1',
  'intellifold2',
  'af3',
]);

export interface LocalFoldEffectiveConfigInput {
  modelKey?: string;
  params?: Record<string, unknown>;
  /** Max protein residue length across scheduled candidate(s) (triggers 64-step floor when >2048). */
  maxResidueLength?: number;
  /** Optional explicit override for holo context presence (e.g. already-parsed ligands/DNA/RNA). */
  hasHoloContext?: boolean;
  /** Optional wired template StructureArtifact from `localfold-af2.template` port. */
  templateInput?: StructureArtifact | null;
}

export interface LocalFoldEffectiveConfig {
  modelKey: string;
  isAf2Mono: boolean;
  isAf2Multi: boolean;
  isAf3: boolean;
  isEsmfold2: boolean;
  recycles: number;
  tolerance: number;
  diffusionSteps: number;
  msaMode: MsaMode;
  msaA3mText: string;
  ligandCcdCodes: string[];
  dnaSeqs: string[];
  rnaSeqs: string[];
  hasHoloContext: boolean;
  requestedEsmfold2Steps: number;
  effectiveEsmfold2Steps: number;
  esmfold2StepsAdjusted: boolean;
  templateFamily: LocalFoldTemplateFamily;
  isTemplateCapable: boolean;
  templateSpanChains: boolean;
  templateSource: ResolvedLocalFoldTemplateSource | null;
  templateSourceError?: string;
}

/**
 * Resolve ESMFold2 effective diffusion steps matching upstream `esmfold2StepsFor`:
 * holo/atomised complexes (ligands/DNA/RNA) or sequences >2048 aa floor at 64 steps.
 */
export function resolveEsmfold2EffectiveSteps(
  requestedSteps: unknown,
  opts?: { hasHoloContext?: boolean; residueLength?: number },
): number {
  const parsed = Number(requestedSteps ?? 15);
  const base = Number.isFinite(parsed) ? Math.round(parsed) : 15;
  const needs64Floor = Boolean(opts?.hasHoloContext) || (opts?.residueLength ?? 0) > 2048;
  return needs64Floor ? Math.max(base, 64) : base;
}

/**
 * Pure single-source-of-truth LocalFold effective config resolver (`CT-05`).
 * Shared by `localfold.worker.ts`, `localfold-esmfold2.ts`, and `input-preview.ts`
 * so `af2-multimer` tolerance (`0.5` Å), holo parsing, and `effectiveEsmfold2Steps` (`64` on
 * holo or `>2048` aa) are resolved identically BEFORE `runHash`, `FoldedStructure.id`, and preview.
 */
export function resolveLocalFoldEffectiveConfig(
  input: LocalFoldEffectiveConfigInput,
): LocalFoldEffectiveConfig {
  const p = input.params ?? {};
  const modelKey = String(input.modelKey ?? p.model ?? 'monomer').trim();
  const isAf2Mono = LOCALFOLD_AF2_MODELS.has(modelKey);
  const isAf2Multi = LOCALFOLD_AF2_MULTIMER_MODELS.has(modelKey);
  const isAf3 = LOCALFOLD_AF3_MODELS.has(modelKey);
  const isEsmfold2 = LOCALFOLD_ESMFOLD2_MODELS.has(modelKey);

  const maxRecycles = modelKey === 'mol1' || isAf2Multi ? 20 : 10;
  const parsedRecycles = Number(p.recycles ?? 1);
  const recycles = Math.max(
    0,
    Math.min(maxRecycles, Number.isFinite(parsedRecycles) ? Math.round(parsedRecycles) : 1),
  );

  const defaultTol = isAf2Multi ? 0.5 : 0;
  const rawTol = p.tolerance;
  const tolerance =
    rawTol === undefined || rawTol === null || rawTol === ''
      ? defaultTol
      : Number.isFinite(Number(rawTol))
        ? Math.max(0, Math.min(5, Number(rawTol)))
        : defaultTol;

  const parsedDiff = Number(p.diffusionSteps ?? 25);
  const diffusionSteps = Math.max(
    10,
    Math.min(200, Number.isFinite(parsedDiff) ? Math.round(parsedDiff) : 25),
  );

  const rawMsaMode = isEsmfold2 ? 'single' : String(p.msaMode ?? 'single').trim();
  const msaMode: MsaMode =
    rawMsaMode === 'mmseqs2' || rawMsaMode === 'target-only' || rawMsaMode === 'a3m'
      ? rawMsaMode
      : 'single';
  const msaA3mText = typeof p.msaA3mText === 'string' ? p.msaA3mText : '';

  const ligandCcdCodes: string[] = (
    Array.isArray(p.ligandCcdCodes)
      ? (p.ligandCcdCodes as unknown[]).flatMap((s) => String(s ?? '').split(/[,;\s]+/))
      : String(p.ligandCcdCodes ?? '').split(/[,;\s]+/)
  )
    .map((s) => s.trim().toUpperCase())
    .filter((s) => s.length > 0 && /[A-Z0-9]/.test(s));
  const dnaSeqs: string[] = Array.isArray(p.dnaSequences)
    ? (p.dnaSequences as string[])
        .map((s) => String(s).replace(/[^ACGT]/gi, '').toUpperCase())
        .filter(Boolean)
    : String(p.dnaSequences ?? '')
        .split(/[:,;\s]+/)
        .map((s) => s.replace(/[^ACGT]/gi, '').toUpperCase())
        .filter(Boolean);
  const rnaSeqs: string[] = Array.isArray(p.rnaSequences)
    ? (p.rnaSequences as string[])
        .map((s) => String(s).replace(/[^ACGU]/gi, '').toUpperCase())
        .filter(Boolean)
    : String(p.rnaSequences ?? '')
        .split(/[:,;\s]+/)
        .map((s) => s.replace(/[^ACGU]/gi, '').toUpperCase())
        .filter(Boolean);

  const hasHoloContext =
    input.hasHoloContext ??
    (ligandCcdCodes.length > 0 || dnaSeqs.length > 0 || rnaSeqs.length > 0);
  const parsedEf2 = Number(p.esmfold2Steps ?? 15);
  const requestedEsmfold2Steps = Number.isFinite(parsedEf2) ? Math.round(parsedEf2) : 15;
  const effectiveEsmfold2Steps = resolveEsmfold2EffectiveSteps(requestedEsmfold2Steps, {
    hasHoloContext,
    residueLength: input.maxResidueLength,
  });

  const templateFamily = localFoldTemplateFamily(modelKey);
  const isTemplateCapable = isLocalFoldTemplateCapable(modelKey);
  const templateSpanChains =
    templateFamily === 'af2-multimer' || templateFamily === 'af3'
      ? p.templateSpanChains !== false
      : true;
  let templateSource: ResolvedLocalFoldTemplateSource | null = null;
  let templateSourceError: string | undefined;
  try {
    templateSource = resolveLocalFoldTemplateSource({
      templateInput: input.templateInput,
      params: p,
    });
  } catch (err) {
    templateSourceError = err instanceof Error ? err.message : String(err);
  }

  return {
    modelKey,
    isAf2Mono,
    isAf2Multi,
    isAf3,
    isEsmfold2,
    recycles,
    tolerance,
    diffusionSteps,
    msaMode,
    msaA3mText,
    ligandCcdCodes,
    dnaSeqs,
    rnaSeqs,
    hasHoloContext,
    requestedEsmfold2Steps,
    effectiveEsmfold2Steps,
    esmfold2StepsAdjusted: isEsmfold2 && effectiveEsmfold2Steps !== requestedEsmfold2Steps,
    templateFamily,
    isTemplateCapable,
    templateSpanChains,
    templateSource,
    ...(templateSourceError !== undefined ? { templateSourceError } : {}),
  };
}
