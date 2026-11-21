// Fetch 3D structures directly from RCSB PDB (with PDBe fallback) by 4-char PDB ID.

import { ResultMemo } from '../engine/result-memo';
import { parseStructureText } from './pdb';

export interface FetchedPdbStructure {
  pdbId: string;
  format: 'mmcif' | 'pdb';
  filename: string;
  text: string;
  sourceUrl: string;
  nResidues: number;
  chainCount: number;
  ligandNames: string[];
  nucleicAcidChains: string[];
}

export const QUICK_PDB_EXAMPLES: { id: string; name: string }[] = [
  { id: '1PRW', name: 'Calmodulin' },
  { id: '5O45', name: 'PD-L1' },
  { id: '1L2Y', name: 'Trp-cage' },
  { id: '4JHW', name: 'RSV-F' },
];

const memoryCache = new ResultMemo<FetchedPdbStructure>(16 * 1024 * 1024, 16);
const pending = new Map<string, Promise<FetchedPdbStructure>>();

/** Normalize user-entered PDB IDs (e.g. " 1prw.cif " -> "1PRW", "pdb_00001prw" -> "PDB_00001PRW"). */
export function normalizePdbId(raw: string): string {
  const cleaned = raw
    .trim()
    .replace(/\.(cif|mmcif|pdb|ent)$/i, '')
    .trim()
    .toUpperCase();
  return cleaned;
}

/** Standard 4-char RCSB PDB ID (digit + 3 alphanumerics) or extended 12-char `PDB_0000XXXX` ID. */
export function isValidPdbId(raw: string): boolean {
  const id = normalizePdbId(raw);
  return /^[0-9][A-Z0-9]{3}$/.test(id) || /^PDB_[0-9A-Z]{8}$/.test(id);
}

export function clearRcsbCache(): void {
  memoryCache.clear();
  pending.clear();
}

/**
 * Fetch an mmCIF (preferred) or PDB structure from RCSB PDB with automatic fallback
 * to RCSB `.pdb` and PDBe `_updated.cif`.
 */
export async function fetchStructureByPdbId(
  rawId: string,
  fetchImpl: typeof fetch = fetch,
): Promise<FetchedPdbStructure> {
  const pdbId = normalizePdbId(rawId);
  if (!isValidPdbId(pdbId)) {
    throw new Error(`Invalid PDB ID "${rawId}" — enter a 4-character RCSB code (e.g. 1PRW, 5O45).`);
  }

  const cached = memoryCache.get(pdbId);
  if (cached) return cached;

  const inFlight = pending.get(pdbId);
  if (inFlight) return inFlight;

  const request = fetchStructure(pdbId, fetchImpl);
  pending.set(pdbId, request);
  try {
    const result = await request;
    if (pending.get(pdbId) === request) memoryCache.set(pdbId, result);
    return result;
  } finally {
    // A cache clear may have started a newer request for this ID.
    if (pending.get(pdbId) === request) pending.delete(pdbId);
  }
}

async function fetchStructure(pdbId: string, fetchImpl: typeof fetch): Promise<FetchedPdbStructure> {
  const candidates: Array<{ url: string; format: 'mmcif' | 'pdb'; filename: string }> = [
    {
      url: `https://files.rcsb.org/download/${encodeURIComponent(pdbId)}.cif`,
      format: 'mmcif',
      filename: `${pdbId}.cif`,
    },
    {
      url: `https://files.rcsb.org/download/${encodeURIComponent(pdbId)}.pdb`,
      format: 'pdb',
      filename: `${pdbId}.pdb`,
    },
    {
      url: `https://www.ebi.ac.uk/pdbe/entry-files/download/${encodeURIComponent(pdbId.toLowerCase())}_updated.cif`,
      format: 'mmcif',
      filename: `${pdbId}.cif`,
    },
  ];

  let lastStatus = '';
  for (const cand of candidates) {
    try {
      const resp = await fetchImpl(cand.url, { mode: 'cors' });
      if (!resp.ok) {
        lastStatus = `HTTP ${resp.status}`;
        continue;
      }
      const text = await resp.text();
      if (!text || text.trim().length < 32) {
        lastStatus = 'empty response';
        continue;
      }
      const parsed = parseStructureText(text);
      if (
        parsed.nResidues < 3 &&
        !parsed.diagnostics.hasLigands &&
        !parsed.diagnostics.hasNucleicAcids
      ) {
        lastStatus = 'no protein, nucleic-acid, or ligand atoms found';
        continue;
      }
      return {
        pdbId,
        format: parsed.format,
        filename: cand.filename,
        text,
        sourceUrl: cand.url,
        nResidues: parsed.nResidues,
        chainCount: parsed.diagnostics.chainCount,
        ligandNames: parsed.diagnostics.ligandNames,
        nucleicAcidChains: parsed.diagnostics.nucleicAcidChains,
      };
    } catch (err) {
      lastStatus = err instanceof Error ? err.message : String(err);
    }
  }

  throw new Error(
    `Could not fetch PDB entry "${pdbId}" from RCSB/PDBe (${lastStatus || 'network error'}). Check the ID or upload the file manually.`,
  );
}
