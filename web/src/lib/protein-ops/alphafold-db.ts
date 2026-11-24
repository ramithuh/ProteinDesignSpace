// Fetch predicted 3D protein structures directly from the public AlphaFold Protein Structure
// Database (EMBL-EBI / Google DeepMind, CC-BY-4.0) by UniProt accession.

import { ResultMemo } from '../engine/result-memo';
import { parseStructureText } from './pdb';

export interface FetchedAlphaFoldStructure {
  uniprotId: string;
  entryId: string;
  version: number | null;
  format: 'mmcif' | 'pdb';
  filename: string;
  text: string;
  sourceUrl: string;
  nResidues: number;
  chainCount: number;
  ligandNames: string[];
  nucleicAcidChains: string[];
  gene?: string;
  proteinName?: string;
  organism?: string;
  meanPlddt?: number;
}

export interface AlphaFoldDbExample {
  id: string;
  name: string;
  organism: string;
  length: number;
}

export const QUICK_ALPHAFOLD_DB_EXAMPLES: AlphaFoldDbExample[] = [
  { id: 'P69905', name: 'Hemoglobin α', organism: 'Human', length: 142 },
  { id: 'P00698', name: 'Lysozyme', organism: 'Chicken', length: 147 },
  { id: 'Q9NZQ7', name: 'PD-L1', organism: 'Human', length: 290 },
  { id: 'P04637', name: 'p53', organism: 'Human', length: 393 },
];

export const DEFAULT_ALPHAFOLD_DB_UNIPROT_ID = QUICK_ALPHAFOLD_DB_EXAMPLES[0].id;

const memoryCache = new ResultMemo<FetchedAlphaFoldStructure>(16 * 1024 * 1024, 16);
const pending = new Map<string, Promise<FetchedAlphaFoldStructure>>();

const UNIPROT_ACCESSION_RE =
  /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$/;

/**
 * Normalize user-entered UniProt accessions or pasted AlphaFold DB IDs/URLs
 * (e.g. "p04637", "AF-P04637-F1-model_v6.cif", "https://alphafold.ebi.ac.uk/entry/P04637" -> "P04637").
 */
export function normalizeUniProtId(raw: string): string {
  let s = raw.trim();
  if (!s) return '';
  const urlMatch = s.match(/alphafold\.ebi\.ac\.uk\/(?:entry|files)\/([^/?#]+)/i);
  if (urlMatch) {
    s = urlMatch[1];
  }
  s = s.replace(/\.(bcif|cif|mmcif|pdb|ent)$/i, '').trim();
  const afMatch = s.match(/^AF-([A-Za-z0-9]+(?:-\d+)?)-F\d+(?:-model_v\d+)?$/i);
  if (afMatch) {
    s = afMatch[1];
  }
  return s.toUpperCase();
}

/** Validate a canonical 6- or 10-character UniProtKB accession (with optional isoform suffix). */
export function isValidUniProtId(raw: string): boolean {
  const id = normalizeUniProtId(raw);
  return UNIPROT_ACCESSION_RE.test(id);
}

export function clearAlphaFoldDbCache(): void {
  memoryCache.clear();
  pending.clear();
}

const DEFAULT_FETCH_TIMEOUT_MS = 15_000;

interface AlphaFoldApiEntry {
  entryId?: string;
  uniprotAccession?: string;
  latestVersion?: number;
  cifUrl?: string;
  pdbUrl?: string;
  gene?: string;
  uniprotDescription?: string;
  organismScientificName?: string;
  globalMetricValue?: number;
}

async function fetchWithTimeout(
  url: string,
  fetchImpl: typeof fetch,
  timeoutMs: number,
): Promise<{ resp: Response | null; error: string }> {
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  let timedOut = false;
  const timer = controller
    ? setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs)
    : null;
  try {
    const resp = await fetchImpl(url, {
      mode: 'cors',
      ...(controller ? { signal: controller.signal } : {}),
    });
    return { resp, error: '' };
  } catch (err) {
    return {
      resp: null,
      error: timedOut
        ? `timed out after ${(timeoutMs / 1000).toFixed(0)}s`
        : err instanceof Error
          ? err.message
          : String(err),
    };
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

/**
 * Fetch an AlphaFold DB predicted structure by UniProt accession.
 * Queries the public EMBL-EBI AlphaFold DB prediction API to resolve the latest model
 * version & metadata, then downloads the mmCIF (falling back to PDB and direct v6/v4 URLs).
 */
export async function fetchStructureByUniProtId(
  rawId: string,
  fetchImpl: typeof fetch = fetch,
  timeoutMs = DEFAULT_FETCH_TIMEOUT_MS,
): Promise<FetchedAlphaFoldStructure> {
  const uniprotId = normalizeUniProtId(rawId);
  if (!isValidUniProtId(uniprotId)) {
    throw new Error(
      `Invalid UniProt accession "${rawId}" — enter a UniProt ID (e.g. P69905, P04637, Q9NZQ7).`,
    );
  }

  const cached = memoryCache.get(uniprotId);
  if (cached) return cached;

  const inFlight = pending.get(uniprotId);
  if (inFlight) return inFlight;

  const request = fetchAlphaFoldStructure(uniprotId, fetchImpl, timeoutMs);
  pending.set(uniprotId, request);
  try {
    const result = await request;
    if (pending.get(uniprotId) === request) memoryCache.set(uniprotId, result);
    return result;
  } finally {
    if (pending.get(uniprotId) === request) pending.delete(uniprotId);
  }
}

async function fetchAlphaFoldStructure(
  uniprotId: string,
  fetchImpl: typeof fetch,
  timeoutMs = DEFAULT_FETCH_TIMEOUT_MS,
): Promise<FetchedAlphaFoldStructure> {
  const effectiveTimeoutMs =
    typeof timeoutMs === 'number' && Number.isFinite(timeoutMs) && timeoutMs > 0
      ? timeoutMs
      : DEFAULT_FETCH_TIMEOUT_MS;

  let meta: AlphaFoldApiEntry | null = null;
  let apiNotFound = false;

  const apiUrl = `https://alphafold.ebi.ac.uk/api/prediction/${encodeURIComponent(uniprotId)}`;
  const apiRes = await fetchWithTimeout(apiUrl, fetchImpl, effectiveTimeoutMs);
  if (apiRes.resp) {
    if (apiRes.resp.status === 404) {
      apiNotFound = true;
    } else if (apiRes.resp.ok) {
      try {
        const json = (await apiRes.resp.json()) as unknown;
        if (Array.isArray(json) && json.length > 0) {
          const entries = json as AlphaFoldApiEntry[];
          meta =
            entries.find(
              (e) =>
                String(e.entryId ?? '').toUpperCase() === `AF-${uniprotId}-F1` ||
                String(e.uniprotAccession ?? '').toUpperCase() === uniprotId,
            ) ?? entries[0];
        }
      } catch {
        // Fall through to direct file URL candidates
      }
    }
  }

  const canonicalEntryId = String(meta?.entryId || `AF-${uniprotId}-F1`);
  const latestVersion =
    typeof meta?.latestVersion === 'number' && Number.isFinite(meta.latestVersion)
      ? meta.latestVersion
      : null;

  const candidates: Array<{ url: string; format: 'mmcif' | 'pdb'; filename: string; version: number | null }> = [];
  const seenUrls = new Set<string>();
  const pushCandidate = (
    url: string | undefined,
    format: 'mmcif' | 'pdb',
    filename: string,
    version: number | null,
  ) => {
    if (!url || seenUrls.has(url)) return;
    seenUrls.add(url);
    candidates.push({ url, format, filename, version });
  };

  if (meta?.cifUrl) {
    const vTag = latestVersion ? `-model_v${latestVersion}` : '';
    pushCandidate(meta.cifUrl, 'mmcif', `${canonicalEntryId}${vTag}.cif`, latestVersion);
  }
  if (meta?.pdbUrl) {
    const vTag = latestVersion ? `-model_v${latestVersion}` : '';
    pushCandidate(meta.pdbUrl, 'pdb', `${canonicalEntryId}${vTag}.pdb`, latestVersion);
  }

  // Fallback direct URLs if the metadata endpoint did not return URLs (unless 404 confirmed absent)
  if (!apiNotFound || candidates.length === 0) {
    for (const ver of [6, 4]) {
      pushCandidate(
        `https://alphafold.ebi.ac.uk/files/AF-${encodeURIComponent(uniprotId)}-F1-model_v${ver}.cif`,
        'mmcif',
        `AF-${uniprotId}-F1-model_v${ver}.cif`,
        ver,
      );
    }
    pushCandidate(
      `https://alphafold.ebi.ac.uk/files/AF-${encodeURIComponent(uniprotId)}-F1-model_v4.pdb`,
      'pdb',
      `AF-${uniprotId}-F1-model_v4.pdb`,
      4,
    );
  }

  let lastStatus = apiNotFound ? 'HTTP 404' : apiRes.error;
  for (const cand of candidates) {
    const { resp, error } = await fetchWithTimeout(cand.url, fetchImpl, effectiveTimeoutMs);
    if (!resp) {
      lastStatus = error || 'network error';
      continue;
    }
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
      lastStatus = 'no protein atoms found';
      continue;
    }
    const meanPlddt =
      typeof meta?.globalMetricValue === 'number' && Number.isFinite(meta.globalMetricValue)
        ? meta.globalMetricValue
        : undefined;
    return {
      uniprotId,
      entryId: canonicalEntryId,
      version: cand.version,
      format: parsed.format,
      filename: cand.filename,
      text,
      sourceUrl: cand.url,
      nResidues: parsed.nResidues,
      chainCount: parsed.diagnostics.chainCount,
      ligandNames: parsed.diagnostics.ligandNames,
      nucleicAcidChains: parsed.diagnostics.nucleicAcidChains,
      ...(meta?.gene ? { gene: String(meta.gene).trim() } : {}),
      ...(meta?.uniprotDescription ? { proteinName: String(meta.uniprotDescription).trim() } : {}),
      ...(meta?.organismScientificName
        ? { organism: String(meta.organismScientificName).trim() }
        : {}),
      ...(meanPlddt !== undefined ? { meanPlddt } : {}),
    };
  }

  throw new Error(
    `Could not fetch AlphaFold DB entry for UniProt "${uniprotId}" (${lastStatus || 'network error'}). Check the accession (e.g. P69905, P04637) or upload the file manually.`,
  );
}
