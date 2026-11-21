// Model-agnostic conditioning compiler:
// 1. Builds a canonical CanonicalConditionIntent from block parameters.
// 2. Compiles canonical intent -> RFD4-Proteina `sel(...)` / `C_*` dict (decoupled from scheduler).
// 3. Compiles canonical intent + template structure -> Genie 3 `createFeaturesFromSpec(...)` batch
//    for both protein motif scaffolding and PPI binder design!

import { parseStructureText } from '$lib/protein-ops/pdb';
import { getBenchmarkById } from './benchmarks';
import type {
  CanonicalConditionIntent,
  ConditionSpecArtifact,
  StructureArtifact,
} from './types';

export interface ResidueRangeToken {
  chain: string;
  start: number;
  end: number;
}

const GENIE3_RESTYPE_1 = 'ARNDCQEGHILKMFPSTWYV';
const WATER_COMP_IDS = new Set(['HOH', 'DOD', 'WAT', 'H2O']);
const DNA_COMP_IDS = new Set(['DA', 'DC', 'DG', 'DT', 'DI', 'DU']);
const RNA_COMP_IDS = new Set(['A', 'C', 'G', 'U', 'I']);

export interface Rfd4PolymerResidueEntry {
  authChain: string;
  authSeqId: number;
  labelChain: string;
  labelSeqId: number;
  compId: string;
  isNucleic: boolean;
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
  let seenAtom = false;

  for (const line of lines) {
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
    const insCode = line.slice(26, 27).trim() || '.';
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
      const authKey = `${authChain}:${resolvedAuthSeq}`;
      const labelKey = `${labelChain}:${labelSeq}`;
      if (!byLabelRes.has(labelKey)) {
        const entry: Rfd4PolymerResidueEntry = {
          authChain,
          authSeqId: resolvedAuthSeq,
          labelChain,
          labelSeqId: labelSeq,
          compId: comp,
          isNucleic,
        };
        byLabelRes.set(labelKey, entry);
        if (!byAuthRes.has(authKey)) byAuthRes.set(authKey, entry);

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

  const nucleicChains = [...polymerByLabelChain.values()].filter((c) => c.isNucleic);

  const remapPolymerChain = (chain: string): string => {
    const ch = chain.trim().toUpperCase();
    const byAuth = polymerByAuthChain.get(ch);
    if (byAuth) return byAuth.labelChain;
    const byLabel = polymerByLabelChain.get(ch);
    if (byLabel) return byLabel.labelChain;
    return ch;
  };

  const remapPolymerRange = (
    chain: string,
    start: number,
    end: number,
  ): { labelChain: string; start: number; end: number } => {
    const ch = chain.trim().toUpperCase();
    const authInfo = polymerByAuthChain.get(ch);
    if (authInfo) {
      const hits = authInfo.residues.filter((r) => r.authSeqId >= start && r.authSeqId <= end);
      if (hits.length > 0) {
        let minL = Infinity;
        let maxL = -Infinity;
        for (const h of hits) {
          if (h.labelSeqId < minL) minL = h.labelSeqId;
          if (h.labelSeqId > maxL) maxL = h.labelSeqId;
        }
        return { labelChain: authInfo.labelChain, start: minL, end: maxL };
      }
    }
    const labelInfo = polymerByLabelChain.get(ch);
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
    return { labelChain: remapPolymerChain(ch), start, end };
  };

  const remapResidueList = (tokens: string[]): string[] => {
    const out: string[] = [];
    for (const r of parseResidueRangeTokens(tokens.join(', '))) {
      const mapped = remapPolymerRange(r.chain, r.start, r.end);
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
      // Expandable segment (e.g. "70-70", "5-20", "h1:16-16P")
      if (/^\d+(?:-\d+)?[PRDN]?$/i.test(tok) || /^[A-Za-z0-9_]+:\d+(?:-\d+)?[PRDN]?$/i.test(tok)) {
        out.push(tok);
        continue;
      }
      // Explicit ligand selector like "A:SAM" or "A:ZN"
      const chainLig = /^([A-Za-z]+):([A-Za-z0-9]{1,5})$/.exec(tok);
      if (chainLig && !/^\d+$/.test(chainLig[2])) {
        const ch = chainLig[1].toUpperCase();
        const comp = chainLig[2].toUpperCase();
        const matches = ligands.filter(
          (l) => l.compId === comp && (l.authChain === ch || l.labelChain === ch),
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
      // Preserved residue/range token like "A17-132" or "A102" or "B296"
      const preserved = /^([A-Za-z]+)(\d+)(?:-(\d+))?$/.exec(tok);
      if (preserved) {
        const ch = preserved[1].toUpperCase();
        const start = parseInt(preserved[2], 10);
        const end = preserved[3] ? parseInt(preserved[3], 10) : start;
        // Check polymer match first
        const authPoly = polymerByAuthChain.get(ch);
        const labelPoly = polymerByLabelChain.get(ch);
        const hasAuthHit = authPoly?.residues.some((r) => r.authSeqId >= start && r.authSeqId <= end);
        const hasLabelHit = labelPoly?.residues.some((r) => r.labelSeqId >= start && r.labelSeqId <= end);
        if (hasAuthHit || hasLabelHit) {
          const m = remapPolymerRange(ch, start, end);
          out.push(m.start === m.end ? `${m.labelChain}${m.start}` : `${m.labelChain}${m.start}-${m.end}`);
          continue;
        }
        // Check ligand match by authChain/labelChain + resId
        const ligHit = ligands.find(
          (l) => (l.labelChain === ch || l.authChain === ch) && l.resId >= start && l.resId <= end,
        );
        if (ligHit) {
          out.push(`${ligHit.labelChain}${ligHit.resId}`);
          continue;
        }
        out.push(tok);
        continue;
      }
      // Whole polymer chain letter like "B" or "C"
      if (/^[A-Za-z]$/.test(tok)) {
        const ch = tok.toUpperCase();
        const info = polymerByAuthChain.get(ch) ?? polymerByLabelChain.get(ch);
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
        const byAuth = ligands.filter(
          (l) =>
            l.compId === compPart &&
            (chainPart === '*' || l.authChain === chainPart.toUpperCase() || l.labelChain === chainPart.toUpperCase()),
        );
        const matches = byAuth.length > 0 ? byAuth : ligands.filter((l) => l.compId === compPart);
        if (matches.length > 0) {
          const uniqueChains = [...new Set(matches.map((m) => m.labelChain))];
          parts[0] = uniqueChains.length === 1 ? uniqueChains[0] : `[${uniqueChains.join(',')}]`;
          return parts.join('/');
        }
      }

      if (chainPart && chainPart !== '*' && /^[A-Za-z]+$/.test(chainPart)) {
        const origChain = chainPart.toUpperCase();
        parts[0] = remapPolymerChain(origChain);
        if (resPart && resPart !== '*') {
          const inner = resPart.startsWith('[') && resPart.endsWith(']') ? resPart.slice(1, -1) : resPart;
          const remappedNums: string[] = [];
          for (const item of inner.split(',').map((s) => s.trim()).filter(Boolean)) {
            const m = /^(\d+)(?:-(\d+))?$/.exec(item);
            if (!m) {
              remappedNums.push(item);
              continue;
            }
            const lo = parseInt(m[1], 10);
            const hi = m[2] ? parseInt(m[2], 10) : lo;
            const mapped = remapPolymerRange(origChain, lo, hi);
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
      return trimmed.replace(/sel\(\s*(['"])([^'"]+)\1\s*\)/g, (_m, q: string, inner: string) => {
        const remapped = inner
          .split(',')
          .map((p) => remapPath(p.trim()))
          .join(',');
        return `sel(${q}${remapped}${q})`;
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
      const ch = lockChain.trim().toUpperCase();
      target = polymerByAuthChain.get(ch) ?? polymerByLabelChain.get(ch);
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
    .some((tok) => /^[A-Za-z]+\d+(?:-\d+)?$/.test(tok));
}

function preservedChainsInContig(contig: string): Set<string> {
  const chains = new Set<string>();
  for (const raw of contig.split(',')) {
    const m = /^([A-Za-z]+)\d+(?:-\d+)?$/.exec(raw.trim());
    if (m) chains.add(m[1].toUpperCase());
  }
  return chains;
}

/** Parse comma/whitespace-separated tokens like "A54, A56, A16-35" into structured ranges. */
export function parseResidueRangeTokens(raw: string): ResidueRangeToken[] {
  const out: ResidueRangeToken[] = [];
  for (const tok of raw.split(/[,\s]+/).filter(Boolean)) {
    const m = /^([A-Za-z])?:?(\d+)(?:-(\d+))?$/.exec(tok.trim());
    if (!m) continue;
    const chain = (m[1] || 'A').toUpperCase();
    const start = parseInt(m[2], 10);
    const end = m[3] ? parseInt(m[3], 10) : start;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) continue;
    out.push({ chain, start, end });
  }
  return out;
}

/** Extract a clean, model-agnostic CanonicalConditionIntent from node params. */
export function buildCanonicalConditionIntent(
  params: Record<string, unknown>,
): CanonicalConditionIntent {
  const contig = String(params.contig ?? '70-70').trim() || '70-70';
  const lockChain = String(params.lockChain ?? '').trim().toUpperCase().slice(0, 1) || undefined;
  const hotspots = String(params.hotspots ?? '')
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const motifSpans = String(params.motifSpan ?? '')
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const unindexedMotif = Boolean(params.unindexedMotif);
  const hbondMode = (String(params.hbondMode ?? 'none') as CanonicalConditionIntent['hbondMode']) || 'none';
  const hbondSelection = String(params.hbondSelection ?? '').trim() || undefined;
  const cathCode = String(params.cathCode ?? '').trim() || undefined;
  const benchmarkId = String(params.benchmarkId ?? '').trim() || undefined;
  const pdbId = String(params.pdbId ?? '').trim().toUpperCase() || undefined;

  return {
    contig,
    ...(lockChain ? { lockChain } : {}),
    ...(hotspots.length > 0 ? { hotspots } : {}),
    ...(motifSpans.length > 0 ? { motifSpans } : {}),
    ...(unindexedMotif ? { unindexedMotif: true } : {}),
    ...(hbondMode && hbondMode !== 'none' ? { hbondMode } : {}),
    ...(hbondSelection ? { hbondSelection } : {}),
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
  opts?: { omitContig?: boolean },
): {
  contig: string;
  conditions: Record<string, unknown>;
  tags: string[];
} {
  let contig = intent.contig.trim() || '70-70';
  if (templateContext && !opts?.omitContig) {
    contig = templateContext.remapContig(contig);
  }

  const effectiveLockChain = intent.lockChain
    ? templateContext
      ? templateContext.remapPolymerChain(intent.lockChain)
      : intent.lockChain[0].toUpperCase()
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
      const spanTokens = parsedSpans.map((r) =>
        r.start === r.end ? `${r.chain}${r.start}` : `${r.chain}${r.start}-${r.end}`,
      );
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

  // Inject preserved template ligands/ions and nucleic acid chains into contig
  const injectedLigands: Rfd4LigandEntry[] = [];
  const injectedNucleic: Rfd4PolymerChainInfo[] = [];
  if (templateContext) {
    const existingChains = preservedChainsInContig(contig);
    for (const lig of templateContext.ligands) {
      if (!existingChains.has(lig.labelChain.toUpperCase())) {
        if (!opts?.omitContig) {
          contig = `${contig},/0,${lig.labelChain}${lig.resId}`;
        }
        existingChains.add(lig.labelChain.toUpperCase());
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
      list.push(r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`);
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
    const byChain = new Map<string, string[]>();
    for (const r of parseResidueRangeTokens(effectiveMotifSpans.join(', '))) {
      const list = byChain.get(r.chain) ?? [];
      list.push(r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`);
      byChain.set(r.chain, list);
    }
    const exprs = [...byChain.entries()].map(([ch, nums]) => `sel('${ch}/*/[${nums.join(',')}]/*')`);
    if (exprs.length > 0) {
      const sel = exprs.join(' | ');
      conditions.C_CRDGRP = { 1: [{ select: sel }] };
      hasMotifCrdGrp = true;
      appendSelect('C_SEQ', sel);
      if (intent.unindexedMotif) {
        conditions.C_IDX = { false: [{ select: sel, group: 'motif' }] };
      }
      tags.push(
        intent.unindexedMotif
          ? `unidx-motif:${effectiveMotifSpans.join(', ')}`
          : `motif:${effectiveMotifSpans.join(', ')}`,
      );
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

  if (intent.hbondMode && intent.hbondMode !== 'none' && intent.hbondSelection) {
    const remappedSel = templateContext
      ? templateContext.remapHbondSelection(intent.hbondSelection)
      : intent.hbondSelection;
    const expr = remappedSel.includes('sel(') ? remappedSel : `sel('${remappedSel}')`;
    conditions.C_HB = { [intent.hbondMode]: [{ select: expr }] };
    tags.push(`HB:${intent.hbondMode}`);
  }

  if (intent.cathCode) {
    conditions.C_CATH = { [intent.cathCode]: [{ select: "sel('*/*/*/*')" }] };
    tags.push(`CATH:${intent.cathCode}`);
  }

  return { contig, conditions, tags };
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

export interface ExtractedDesignMetadata {
  lockedChains?: string[];
  fixedResidueIndices?: number[];
  chainRoles?: Record<string, 'target' | 'binder' | 'scaffold'>;
  motifIndices?: number[];
  nativeMotifCoords?: number[][];
}

/**
 * Derive output `lockedChains`, 1-based `fixedResidueIndices`, `chainRoles`, 0-based `motifIndices`,
 * and `nativeMotifCoords` from an RFD4 generated structure + input batch.
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
  const caRecords: Array<Record<string, unknown>> = [];
  if (Array.isArray(structureAtoms)) {
    for (const rec of structureAtoms) {
      if (!rec.isHetero && rec.atomName === 'CA') {
        caRecords.push(rec);
      }
    }
  }

  const uniqueChains: string[] = [];
  const chainStats = new Map<string, { total: number; fixed: number }>();
  const perResFlags: Array<{
    chain: string;
    isFixed: boolean;
    isMotifCandidate: boolean;
    nativeCoord: number[] | null;
  }> = [];

  for (let i = 0; i < parsed.nResidues; i++) {
    const chain = parsed.chainIds[i] || 'A';
    if (!chainStats.has(chain)) {
      chainStats.set(chain, { total: 0, fixed: 0 });
      uniqueChains.push(chain);
    }
    const stat = chainStats.get(chain)!;
    stat.total++;

    const rec = caRecords.length === parsed.nResidues ? caRecords[i] : undefined;
    const tid = typeof rec?.tokenId === 'number' ? rec.tokenId : -1;
    const hasSeqCond = tid >= 0 && condSeqRaw && Number(condSeqRaw[tid]) > 0;
    const coordGroup = typeof rec?.coordGroup === 'number' ? rec.coordGroup : -1;
    const isUnidx = Boolean(rec?.isUnindexedInsertion);
    const isFixed = hasSeqCond || coordGroup >= 0 || isUnidx;
    if (isFixed) stat.fixed++;

    const rawCoord =
      (Array.isArray(rec?.inCoord) && rec.inCoord.length === 3 ? rec.inCoord : null) ??
      (Array.isArray(rec?.crd) && rec.crd.length === 3 ? rec.crd : null) ??
      (Array.isArray(rec?.intCrd) && rec.intCrd.length === 3 ? rec.intCrd : null);
    const nativeCoord =
      rawCoord && rawCoord.every((v: unknown) => typeof v === 'number' && Number.isFinite(v))
        ? [Number(rawCoord[0]), Number(rawCoord[1]), Number(rawCoord[2])]
        : null;

    perResFlags.push({
      chain,
      isFixed,
      isMotifCandidate: coordGroup > 0 || isUnidx || (coordGroup === 0 && nativeCoord !== null),
      nativeCoord,
    });
  }

  const lockedChains: string[] = [];
  const chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> = {};

  if (uniqueChains.length > 1) {
    const anyFullyFixed = uniqueChains.some((ch) => {
      const s = chainStats.get(ch)!;
      return s.total > 0 && s.fixed === s.total;
    });
    for (const ch of uniqueChains) {
      const s = chainStats.get(ch)!;
      if (s.total > 0 && s.fixed === s.total) {
        lockedChains.push(ch);
        chainRoles[ch] = 'target';
      } else if (anyFullyFixed && s.fixed === 0) {
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

  for (let i = 0; i < perResFlags.length; i++) {
    const f = perResFlags[i];
    if (lockedSet.has(f.chain)) continue;
    if (f.isFixed) {
      fixedResidueIndices.push(i + 1); // 1-based output index
    }
    if (f.isMotifCandidate && f.nativeCoord) {
      motifIndices.push(i); // 0-based index into caCoords
      nativeMotifCoords.push(f.nativeCoord);
    }
  }

  return {
    ...(lockedChains.length > 0 ? { lockedChains } : {}),
    ...(fixedResidueIndices.length > 0 ? { fixedResidueIndices } : {}),
    ...(Object.keys(chainRoles).length > 0 ? { chainRoles } : {}),
    ...(motifIndices.length > 0 ? { motifIndices, nativeMotifCoords } : {}),
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
  if (intent.hbondMode && intent.hbondMode !== 'none') {
    return {
      supported: false,
      reason: 'Hydrogen-bond atom constraints require RFD4-Proteina (Genie 3 is Cα backbone diffusion).',
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
  const firstChain = (template.chainIds?.[0] || 'A').toUpperCase();
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

  // Pure length contig (no lockChain, no motifSpans): unconditional monomer of requested length
  const hasLock = Boolean(intent.lockChain);
  const hasMotif = Boolean(intent.motifSpans && intent.motifSpans.length > 0);
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

  // Index template residues by chain + residue number for exact canonical RCSB lookup
  interface TplRes {
    chain: string;
    resNum: number;
    aa1: string;
    x: number;
    y: number;
    z: number;
  }
  const tplList: TplRes[] = [];
  const tplMap = new Map<string, TplRes>();
  for (let i = 0; i < parsed.nResidues; i++) {
    const entry: TplRes = {
      chain: parsed.chainIds[i] || 'A',
      resNum: parsed.resNums[i] ?? i + 1,
      aa1: parsed.sequence[i] || 'A',
      x: parsed.caCoords[i * 3 + 0],
      y: parsed.caCoords[i * 3 + 1],
      z: parsed.caCoords[i * 3 + 2],
    };
    tplList.push(entry);
    tplMap.set(`${entry.chain}:${entry.resNum}`, entry);
  }

  // ── Mode 1: PPI Binder Design (Genie 3 binder-first convention: chain 0 = binder, chain 1 = fixed target) ──
  if (hasLock) {
    const lockChain = intent.lockChain!.toUpperCase();
    const parts = intent.contig.split(/[,\s]+/).filter(Boolean);
    let binderLen = defaultLength;
    const targetRanges: ResidueRangeToken[] = [];

    for (const part of parts) {
      if (part === '/0') continue;
      const rangeTokens = parseResidueRangeTokens(part);
      if (rangeTokens.length > 0 && /^[A-Za-z]/.test(part)) {
        targetRanges.push(...rangeTokens);
      } else {
        const mid = parseLengthRangeMid(part);
        if (mid !== null) binderLen = mid;
      }
    }

    let targetRes: TplRes[] = [];
    if (targetRanges.length > 0) {
      for (const r of targetRanges) {
        for (const t of tplList) {
          if (t.chain === r.chain && t.resNum >= r.start && t.resNum <= r.end) {
            targetRes.push(t);
          }
        }
      }
    } else {
      targetRes = tplList.filter((t) => t.chain === lockChain);
    }

    if (targetRes.length < 3) {
      throw new Error(
        `Could not find target residues for chain "${lockChain}" in the wired template structure.`,
      );
    }

    const hotspotSet = new Set<string>();
    for (const h of parseResidueRangeTokens((intent.hotspots ?? []).join(', '))) {
      for (let rn = h.start; rn <= h.end; rn++) {
        hotspotSet.add(`${h.chain}:${rn}`);
      }
    }

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
    for (let j = 0; j < targetRes.length; j++) {
      const idx = clampedBinder + j;
      const t = targetRes[j];
      asymId[idx] = 1;
      entityId[idx] = 1;
      residueIndex[idx] = j + 1;
      condGroup[idx] = 1;
      condStructMask[idx] = 1;
      condSeqMask[idx] = 1;
      if (hotspotSet.has(`${t.chain}:${t.resNum}`)) {
        condInterfaceMask[idx] = 1;
        hotspotMatches++;
      }
      writeRestypeOneHot(gtRestype, idx, t.aa1);
      gtAtomPositions[idx * 3 + 0] = t.x - cx;
      gtAtomPositions[idx * 3 + 1] = t.y - cy;
      gtAtomPositions[idx * 3 + 2] = t.z - cz;
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
      summary: `binder ${clampedBinder} aa + target ${targetRes.length} aa (${hotspotMatches} hotspots)`,
    };
  }

  // ── Mode 2: Indexed Motif Scaffolding (single chain with fixed motif segments) ──
  const contigParts = intent.contig.split(/[,\s]+/).filter((s) => s && s !== '/0');
  type Segment = { kind: 'scaffold'; len: number } | { kind: 'motif'; residues: TplRes[] };
  const segments: Segment[] = [];
  let hasExplicitMotifInContig = false;

  for (const part of contigParts) {
    if (/^[A-Za-z]/.test(part)) {
      const ranges = parseResidueRangeTokens(part);
      const res: TplRes[] = [];
      for (const r of ranges) {
        for (let rn = r.start; rn <= r.end; rn++) {
          const hit = tplMap.get(`${r.chain}:${rn}`);
          if (hit) res.push(hit);
        }
      }
      if (res.length > 0) {
        hasExplicitMotifInContig = true;
        segments.push({ kind: 'motif', residues: res });
      }
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
    for (const r of motifGroups) {
      const res: TplRes[] = [];
      for (let rn = r.start; rn <= r.end; rn++) {
        const hit = tplMap.get(`${r.chain}:${rn}`);
        if (hit) res.push(hit);
      }
      if (res.length > 0) motifBlocks.push(res);
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
    summary: `motif scaffold L=${totalLen} (${motifResAll.length} fixed motif aa)`,
  };
}
