import type { EdgeStructurePreview, ProteinChainTrace, ProteinLigandTrace } from './raster';
import type { OgStructureTarget } from './og-graph';

const WATER_CODES = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'TIP3']);
const STANDARD_AA = new Set([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE',
]);

interface SimpleCropFilter {
  chains: Set<string>;
  ranges: Array<{ chain: string; start: number; end: number }>;
  ligands: Array<{ chain?: string; compId: string }>;
}

function parseSimpleCrop(spec: string): SimpleCropFilter | null {
  const tokens = spec
    .trim()
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (tokens.length === 0) return null;

  const chains = new Set<string>();
  const ranges: Array<{ chain: string; start: number; end: number }> = [];
  const ligands: Array<{ chain?: string; compId: string }> = [];

  for (const tok of tokens) {
    if (tok.includes(':')) {
      const rm = /^([A-Za-z0-9_]+):(-?\d+)[A-Za-z]?(?:-(-?\d+)[A-Za-z]?)?$/.exec(tok);
      if (rm) {
        const s = parseInt(rm[2], 10);
        const e = rm[3] !== undefined ? parseInt(rm[3], 10) : s;
        ranges.push({ chain: rm[1], start: Math.min(s, e), end: Math.max(s, e) });
        continue;
      }
      const lm = /^([A-Za-z0-9_]+):([A-Za-z0-9]{1,5})$/.exec(tok);
      if (lm) {
        ligands.push({ chain: lm[1], compId: lm[2].toUpperCase() });
        continue;
      }
    }
    if (/^[A-Za-z0-9]$/.test(tok)) {
      chains.add(tok);
      continue;
    }
    const bareR = /^([A-Za-z]+)?(-?\d+)[A-Za-z]?(?:-(-?\d+)[A-Za-z]?)?$/.exec(tok);
    if (bareR) {
      const ch = bareR[1] || 'A';
      const s = parseInt(bareR[2], 10);
      const e = bareR[3] !== undefined ? parseInt(bareR[3], 10) : s;
      ranges.push({ chain: ch, start: Math.min(s, e), end: Math.max(s, e) });
      if (!tok.includes('-') && /^[A-Za-z][A-Za-z0-9]{1,4}$/.test(tok)) {
        ligands.push({ compId: tok.toUpperCase() });
      }
      continue;
    }
    if (/^[A-Za-z0-9]{2,5}$/.test(tok) && /[A-Za-z]/.test(tok)) {
      ligands.push({ compId: tok.toUpperCase() });
      chains.add(tok);
    }
  }
  return { chains, ranges, ligands };
}

function matchesPolymer(filter: SimpleCropFilter | null, chain: string, resNum: number): boolean {
  if (!filter) return true;
  if (filter.chains.has(chain)) return true;
  for (const r of filter.ranges) {
    if (r.chain === chain && resNum >= r.start && resNum <= r.end) return true;
  }
  return false;
}

function matchesLigand(filter: SimpleCropFilter | null, chain: string, comp: string): boolean {
  const u = comp.toUpperCase();
  if (WATER_CODES.has(u)) return false;
  if (!filter) return !STANDARD_AA.has(u);
  for (const l of filter.ligands) {
    if (l.compId === u && (!l.chain || l.chain === chain)) return true;
  }
  return false;
}

/** Fast PDB Cα + ligand parser for edge preview rendering. */
export function parsePdbForOgPreview(
  pdbText: string,
  title: string,
  cropSegments: string,
  isPlddt = false,
): EdgeStructurePreview | null {
  const filter = parseSimpleCrop(cropSegments);
  const chainMap = new Map<string, ProteinChainTrace>();
  const ligMap = new Map<string, ProteinLigandTrace>();
  const seenRes = new Set<string>();
  let residueCount = 0;

  const lines = pdbText.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('ENDMDL') && residueCount > 0) break;
    const isAtom = line.startsWith('ATOM  ');
    const isHet = !isAtom && line.startsWith('HETATM');
    if (!isAtom && !isHet) continue;

    const alt = line[16] ?? ' ';
    if (alt !== ' ' && alt !== 'A' && alt !== '1') continue;
    const atomName = line.slice(12, 16).trim();
    const comp = line.slice(17, 20).trim().toUpperCase();
    const chain = (line[21] ?? 'A').trim() || 'A';
    const resStr = line.slice(22, 27).trim();
    const resNum = parseInt(resStr, 10);
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    if ((isAtom || comp === 'MSE') && atomName === 'CA') {
      if (!matchesPolymer(filter, chain, Number.isFinite(resNum) ? resNum : 0)) continue;
      const rKey = `${chain}:${resStr}`;
      if (seenRes.has(rKey)) continue;
      seenRes.add(rKey);
      let tr = chainMap.get(chain);
      if (!tr) {
        tr = { id: chain, points: [], ...(isPlddt ? { plddt: [] } : {}) };
        chainMap.set(chain, tr);
      }
      tr.points.push([x, y, z]);
      if (isPlddt) {
        const bf = parseFloat(line.slice(60, 66));
        tr.plddt!.push(Number.isFinite(bf) ? bf : 80);
      }
      residueCount++;
    } else if (isHet && matchesLigand(filter, chain, comp)) {
      if (atomName.startsWith('H')) continue;
      const lKey = `${chain}:${comp}:${resStr}`;
      let lt = ligMap.get(lKey);
      if (!lt) {
        lt = { name: comp, chain, atoms: [] };
        ligMap.set(lKey, lt);
      }
      if (lt.atoms.length < 64) {
        lt.atoms.push([x, y, z]);
      }
    }
  }

  // Keep up to 12 chains & 6 ligand instances for crisp visual clarity
  const chains = [...chainMap.values()].filter((c) => c.points.length >= 2).slice(0, 12);
  const ligands = [...ligMap.values()].filter((l) => l.atoms.length > 0).slice(0, 6);
  if (chains.length === 0 && ligands.length === 0) return null;

  const cropNote = cropSegments.trim() ? ` · ${cropSegments.trim()}` : '';
  const chainSummary =
    chains.length === 1 ? `Chain ${chains[0].id}` : `${chains.length} chains`;
  return {
    title,
    subtitle: `${chainSummary} · L=${residueCount} aa${cropNote}`,
    chains,
    ligands,
    residueCount,
    isPlddt,
  };
}

/** Fast mmCIF Cα + ligand fallback parser for PDB entries without legacy .pdb files. */
export function parseCifForOgPreview(
  cifText: string,
  title: string,
  cropSegments: string,
  isPlddt = false,
): EdgeStructurePreview | null {
  const filter = parseSimpleCrop(cropSegments);
  const lines = cifText.split('\n');
  const cols: string[] = [];
  let inAtomSite = false;
  let iGroup = -1,
    iAlt = -1,
    iAtom = -1,
    iComp = -1,
    iAuthAsym = -1,
    iLabelAsym = -1,
    iAuthSeq = -1,
    iLabelSeq = -1,
    iX = -1,
    iY = -1,
    iZ = -1,
    iB = -1,
    iModel = -1;
  let firstModel = '';
  const chainMap = new Map<string, ProteinChainTrace>();
  const ligMap = new Map<string, ProteinLigandTrace>();
  const seenRes = new Set<string>();
  let residueCount = 0;

  for (let i = 0; i < lines.length; i++) {
    const tr = lines[i].trim();
    if (tr.startsWith('_atom_site.')) {
      inAtomSite = true;
      cols.push(tr.slice('_atom_site.'.length));
      continue;
    }
    if (!inAtomSite) continue;
    if (!tr || tr.startsWith('#') || tr.startsWith('loop_') || tr.startsWith('_')) {
      if (cols.length > 0) break;
      continue;
    }
    if (iX < 0) {
      iGroup = cols.indexOf('group_PDB');
      iAlt = cols.indexOf('label_alt_id');
      iAtom = cols.indexOf('label_atom_id');
      if (iAtom < 0) iAtom = cols.indexOf('auth_atom_id');
      iComp = cols.indexOf('label_comp_id');
      if (iComp < 0) iComp = cols.indexOf('auth_comp_id');
      iAuthAsym = cols.indexOf('auth_asym_id');
      iLabelAsym = cols.indexOf('label_asym_id');
      iAuthSeq = cols.indexOf('auth_seq_id');
      iLabelSeq = cols.indexOf('label_seq_id');
      iX = cols.indexOf('Cartn_x');
      iY = cols.indexOf('Cartn_y');
      iZ = cols.indexOf('Cartn_z');
      iB = cols.indexOf('B_iso_or_equiv');
      iModel = cols.indexOf('pdbx_PDB_model_num');
      if (iX < 0 || iY < 0 || iZ < 0) return null;
    }
    const toks = tr.split(/\s+/);
    if (toks.length < cols.length) continue;
    if (iModel >= 0) {
      const m = toks[iModel];
      if (!firstModel) firstModel = m;
      else if (m !== firstModel) continue;
    }
    const alt = iAlt >= 0 ? toks[iAlt] : '.';
    if (alt !== '.' && alt !== '?' && alt !== 'A' && alt !== '1') continue;

    const grp = iGroup >= 0 ? toks[iGroup].toUpperCase() : 'ATOM';
    const atomName = (iAtom >= 0 ? toks[iAtom] : '').replace(/^['"]|['"]$/g, '');
    const comp = (iComp >= 0 ? toks[iComp] : '').toUpperCase();
    const chain =
      (iAuthAsym >= 0 && toks[iAuthAsym] !== '.' && toks[iAuthAsym] !== '?'
        ? toks[iAuthAsym]
        : iLabelAsym >= 0
          ? toks[iLabelAsym]
          : 'A') || 'A';
    const seqStr =
      iAuthSeq >= 0 && toks[iAuthSeq] !== '.' && toks[iAuthSeq] !== '?'
        ? toks[iAuthSeq]
        : iLabelSeq >= 0
          ? toks[iLabelSeq]
          : '0';
    const resNum = parseInt(seqStr, 10);
    const x = parseFloat(toks[iX]);
    const y = parseFloat(toks[iY]);
    const z = parseFloat(toks[iZ]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    if ((grp === 'ATOM' || comp === 'MSE') && atomName === 'CA') {
      if (!matchesPolymer(filter, chain, Number.isFinite(resNum) ? resNum : 0)) continue;
      const rKey = `${chain}:${seqStr}`;
      if (seenRes.has(rKey)) continue;
      seenRes.add(rKey);
      let trChain = chainMap.get(chain);
      if (!trChain) {
        trChain = { id: chain, points: [], ...(isPlddt ? { plddt: [] } : {}) };
        chainMap.set(chain, trChain);
      }
      trChain.points.push([x, y, z]);
      if (isPlddt) {
        const bf = iB >= 0 ? parseFloat(toks[iB]) : 80;
        trChain.plddt!.push(Number.isFinite(bf) ? bf : 80);
      }
      residueCount++;
    } else if (grp === 'HETATM' && matchesLigand(filter, chain, comp)) {
      if (atomName.startsWith('H')) continue;
      const lKey = `${chain}:${comp}:${seqStr}`;
      let lt = ligMap.get(lKey);
      if (!lt) {
        lt = { name: comp, chain, atoms: [] };
        ligMap.set(lKey, lt);
      }
      if (lt.atoms.length < 64) lt.atoms.push([x, y, z]);
    }
  }

  const chains = [...chainMap.values()].filter((c) => c.points.length >= 2).slice(0, 12);
  const ligands = [...ligMap.values()].filter((l) => l.atoms.length > 0).slice(0, 6);
  if (chains.length === 0 && ligands.length === 0) return null;

  const cropNote = cropSegments.trim() ? ` · ${cropSegments.trim()}` : '';
  const chainSummary =
    chains.length === 1 ? `Chain ${chains[0].id}` : `${chains.length} chains`;
  return {
    title,
    subtitle: `${chainSummary} · L=${residueCount} aa${cropNote}`,
    chains,
    ligands,
    residueCount,
    isPlddt,
  };
}

async function fetchTextWithTimeout(
  url: string,
  timeoutMs: number,
  maxBytes: number,
  fetchImpl: typeof fetch,
): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, {
      signal: controller.signal,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
      },
    });
    if (!res.ok) return null;
    const cl = Number(res.headers.get('content-length') ?? '0');
    if (cl > maxBytes) return null;
    const text = await res.text();
    if (text.length > maxBytes) return null;
    return text;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch and parse a public RCSB PDB or AlphaFold DB target at the edge for OG 3D preview.
 * Fails open to `null` on timeout, size limit, or network errors so OG image generation never fails.
 */
export async function fetchEdgeStructurePreview(
  target: OgStructureTarget,
  fetchImpl: typeof fetch = fetch,
): Promise<EdgeStructurePreview | null> {
  const TIMEOUT_MS = 3500;
  const MAX_BYTES = 4_500_000;

  if (target.kind === 'rcsb') {
    const id = target.pdbId.toUpperCase();
    const pdbText = await fetchTextWithTimeout(
      `https://files.rcsb.org/download/${id}.pdb`,
      TIMEOUT_MS,
      MAX_BYTES,
      fetchImpl,
    );
    if (pdbText) {
      return parsePdbForOgPreview(pdbText, `RCSB ${id}`, target.cropSegments, false);
    }
    const cifText = await fetchTextWithTimeout(
      `https://files.rcsb.org/download/${id}.cif`,
      TIMEOUT_MS,
      MAX_BYTES,
      fetchImpl,
    );
    if (cifText) {
      return parseCifForOgPreview(cifText, `RCSB ${id}`, target.cropSegments, false);
    }
    return null;
  }

  if (target.kind === 'afdb') {
    const acc = target.uniprotId.toUpperCase();
    const metaJson = await fetchTextWithTimeout(
      `https://alphafold.ebi.ac.uk/api/prediction/${acc}`,
      TIMEOUT_MS,
      250_000,
      fetchImpl,
    );
    let pdbUrl = `https://alphafold.ebi.ac.uk/files/AF-${acc}-F1-model_v6.pdb`;
    let geneLabel = `AFDB ${acc}`;
    if (metaJson) {
      try {
        const arr = JSON.parse(metaJson);
        const rec = Array.isArray(arr) ? arr[0] : arr;
        if (rec?.pdbUrl && typeof rec.pdbUrl === 'string') pdbUrl = rec.pdbUrl;
        if (rec?.gene && typeof rec.gene === 'string') geneLabel = `AFDB ${rec.gene} (${acc})`;
      } catch {
        /* use fallback url */
      }
    }
    const pdbText = await fetchTextWithTimeout(pdbUrl, TIMEOUT_MS, MAX_BYTES, fetchImpl);
    if (!pdbText) return null;
    return parsePdbForOgPreview(pdbText, geneLabel, target.cropSegments, true);
  }

  return null;
}
