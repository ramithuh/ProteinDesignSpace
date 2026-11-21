<script lang="ts">
  import { onDestroy } from 'svelte';
  import {
    ADAPTER_REGISTRY,
    LOCALFOLD_MODEL_OPTIONS,
    MPNN_CHECKPOINTS_BY_FAMILY,
    MPNN_VARIANT_OPTIONS,
    RFD4_BENCHMARK_OPTIONS,
  } from '$lib/engine/registry';
  import { getActiveModeSpec, getVisibleParamFields } from '$lib/engine/param-schema';
  import { STAGE_COLORS } from '$lib/engine/graph-utils';
  import {
    NUMERIC_SPECS,
    computeUploadDigest,
    isValidResidueToken,
    isValidSha256Hex,
    validatePdbId,
    validateSequenceText,
    validateStructureText,
    type NumericFieldSpec,
  } from '$lib/engine/param-validation';
  import {
    QUICK_PDB_EXAMPLES,
    fetchStructureByPdbId,
    normalizePdbId,
  } from '$lib/protein-ops/rcsb';
  import { parseStructureText, type ParsedStructure } from '$lib/protein-ops/pdb';
  import { resolveMotifSourceResidues, type SelectedMotifSourceResidue } from '$lib/engine/conditioning';
  import { BENCHMARK_TARGETS, getBenchmarkById } from '$lib/engine/benchmarks';
  import { MAX_DESIGNS } from '$lib/engine/work-plan';
  import NumericInput from './NumericInput.svelte';
  import SchemaParamFields from './SchemaParamFields.svelte';
  import type {
    CanonicalArtifact,
    MsaMode,
    NodeProgressEvent,
    SequenceSet,
    StructureArtifact,
    WorkflowNode,
  } from '$lib/engine/types';

  interface Props {
    node: WorkflowNode;
    disabled?: boolean;
    embedded?: boolean;
    nodeOutput?: Record<string, CanonicalArtifact>;
    nodeState?: NodeProgressEvent;
    upstreamStructure?: StructureArtifact | null;
    upstreamSequences?: SequenceSet | null;
    hasStructureInput?: boolean;
    siblingGeneratorIds?: string[];
    onUpdate: (nodeId: string, params: Record<string, unknown>) => void;
    onUpdateLabel?: (nodeId: string, label: string) => void;
    onSyncTaskToSiblings?: (nodeId: string) => void;
    onRunUpTo?: (nodeId: string) => void;
    onOpenStudio?: () => void;
  }

  let {
    node,
    disabled = false,
    embedded = false,
    nodeOutput = undefined,
    nodeState = undefined,
    upstreamStructure = null,
    upstreamSequences = null,
    hasStructureInput = false,
    siblingGeneratorIds = [],
    onUpdate,
    onUpdateLabel = undefined,
    onSyncTaskToSiblings = undefined,
    onRunUpTo = undefined,
    onOpenStudio = undefined,
  }: Props = $props();

  const NUM_DESIGNS_SPEC: NumericFieldSpec = {
    min: 1,
    max: MAX_DESIGNS,
    step: 1,
    integer: true,
    defaultValue: 1,
  };
  const MAX_TOTAL_FOLDS_SPEC: NumericFieldSpec = {
    min: 1,
    max: 64,
    step: 1,
    integer: true,
    defaultValue: 16,
  };

  function getStructureBackboneCount(struct: StructureArtifact | null | undefined): number {
    if (!struct) return 0;
    const raw = struct as StructureArtifact & {
      collection?: { items?: unknown[]; members?: unknown[] };
      structureCollection?: { items?: unknown[]; members?: unknown[] };
    };
    const items =
      raw.collection?.items ??
      raw.collection?.members ??
      raw.structureCollection?.items ??
      raw.structureCollection?.members;
    if (Array.isArray(items) && items.length > 0) {
      return items.length;
    }
    const repTotal = struct.representativeSelection?.totalMembers;
    if (typeof repTotal === 'number' && Number.isFinite(repTotal) && repTotal > 1) {
      return Math.round(repTotal);
    }
    return 1;
  }

  function countDistinctBackbonesFromItems(
    items: readonly { sourceStructureId?: string; designKey?: string }[] | undefined,
  ): number {
    if (!items || items.length === 0) return 0;
    const keys = new Set<string>();
    for (const item of items) {
      const k = (item.designKey ?? item.sourceStructureId ?? '').trim();
      if (k) keys.add(k);
    }
    return keys.size;
  }

  let upstreamBackboneCount = $derived.by((): number => {
    const fromStruct = getStructureBackboneCount(upstreamStructure);
    if (fromStruct > 1) return fromStruct;
    const seqOut = nodeOutput?.sequences?.kind === 'SequenceSet' ? nodeOutput.sequences : null;
    if (seqOut) {
      const fromSeqOut = countDistinctBackbonesFromItems(
        seqOut.candidates as Array<{ sourceStructureId?: string; designKey?: string }>,
      );
      if (fromSeqOut > 1) return fromSeqOut;
    }
    return fromStruct;
  });

  let lfMultiDesignInfo = $derived.by((): { backboneCount: number; sequenceCount: number | null } | null => {
    if (node.adapterId !== 'localfold-af2') return null;
    if (upstreamSequences && upstreamSequences.candidates.length > 0) {
      const nBackbones = countDistinctBackbonesFromItems(
        upstreamSequences.candidates as Array<{ sourceStructureId?: string; designKey?: string }>,
      );
      if (nBackbones > 1) {
        return { backboneCount: nBackbones, sequenceCount: upstreamSequences.candidates.length };
      }
    }
    const foldOut = nodeOutput?.folds?.kind === 'FoldedStructure' ? nodeOutput.folds : null;
    if (foldOut && foldOut.folds.length > 0) {
      const nBackbones = countDistinctBackbonesFromItems(
        foldOut.folds as Array<{ sourceStructureId?: string; designKey?: string }>,
      );
      if (nBackbones > 1) {
        return { backboneCount: nBackbones, sequenceCount: null };
      }
    }
    const fromStruct = getStructureBackboneCount(upstreamStructure);
    if (fromStruct > 1) {
      return { backboneCount: fromStruct, sequenceCount: null };
    }
    return null;
  });

  let liveStructure = $derived<StructureArtifact | null>(
    nodeOutput?.structure?.kind === 'StructureArtifact' ? nodeOutput.structure : null,
  );

  let manifest = $derived(ADAPTER_REGISTRY[node.adapterId]);
  let stageColor = $derived(STAGE_COLORS[manifest?.stage ?? 'source'] ?? '#64748b');

  // Lightweight cache for parsed upstream template structure so typing motif/hotspot tokens
  // never re-parses large mmCIF/PDB texts on every keystroke.
  let cachedTemplateKey = '';
  let cachedParsedTemplate: ParsedStructure | null = null;
  function getParsedUpstreamTemplate(struct: StructureArtifact | null): ParsedStructure | null {
    if (!struct) return null;
    const raw = struct.cifText ?? struct.pdbText ?? '';
    if (!raw.trim()) return null;
    const key = `${struct.id}:${raw.length}`;
    if (key === cachedTemplateKey) return cachedParsedTemplate;
    try {
      cachedParsedTemplate = parseStructureText(raw);
      cachedTemplateKey = key;
    } catch {
      cachedParsedTemplate = null;
      cachedTemplateKey = key;
    }
    return cachedParsedTemplate;
  }

  function formatResidueId(r: SelectedMotifSourceResidue): string {
    return `${r.authChainId}${r.resNum}${r.insCode || ''}`;
  }

  function summarizeSourceResidues(list: SelectedMotifSourceResidue[]): string {
    if (list.length === 0) return '';
    if (list.length <= 5) {
      return list.map(formatResidueId).join(', ');
    }
    const first = formatResidueId(list[0]);
    const second = formatResidueId(list[1]);
    const last = formatResidueId(list[list.length - 1]);
    return `${first}, ${second}, … ${last}`;
  }

  let conditioningSourceSummary = $derived.by((): {
    active: boolean;
    kindLabel: string;
    count: number;
    preview: string;
    hasTemplate: boolean;
  } | null => {
    const p = node.params ?? {};
    let mode = '';
    if (node.adapterId === 'rfd4-proteina') {
      mode = String(p.mode ?? 'unconditional');
    } else if (node.adapterId === 'genie3') {
      mode = String(p.taskMode ?? 'unconditional');
    } else {
      return null;
    }
    if (mode !== 'motif' && mode !== 'binder') return null;

    let spec = '';
    let kindLabel = 'motif';
    if (mode === 'motif') {
      spec = String(p.motifSpan ?? '').trim() || String(p.contig ?? '').trim();
      kindLabel = 'motif';
    } else {
      const hot = String(p.hotspots ?? '').trim();
      if (hot) {
        spec = hot;
        kindLabel = 'hotspot';
      } else {
        const contig = String(p.contig ?? '').trim();
        const beforeBreak = contig.includes('/0') ? contig.split('/0')[0].trim() : '';
        spec = beforeBreak;
        kindLabel = 'target';
      }
    }
    if (!spec) return null;

    const parsed = getParsedUpstreamTemplate(upstreamStructure);
    if (!parsed || parsed.nResidues === 0) {
      return { active: true, kindLabel, count: 0, preview: '', hasTemplate: false };
    }
    const resolved = resolveMotifSourceResidues(parsed, spec);
    return {
      active: true,
      kindLabel,
      count: resolved.length,
      preview: summarizeSourceResidues(resolved),
      hasTemplate: true,
    };
  });

  let selectedLocalFoldMeta = $derived.by(() => {
    if (node.adapterId !== 'localfold-af2') return null;
    const modelId = String(node.params?.model ?? 'monomer');
    return LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
  });

  let rfd4NeedsEncoder = $derived.by((): boolean => {
    if (node.adapterId !== 'rfd4-proteina') return false;
    const mode = String(node.params?.mode ?? 'unconditional');
    return (
      mode === 'partial' ||
      mode === 'inverse-folding' ||
      (mode === 'preset' &&
        Boolean(
          RFD4_BENCHMARK_OPTIONS.find((o) => o.id === String(node.params?.presetId ?? ''))
            ?.needsEncoder,
        ))
    );
  });

  let titleDraft = $state('');
  $effect(() => {
    void node.id;
    titleDraft = node.label ?? '';
  });

  // Debounced textarea drafts for large `pdbText`, `sequence`, and `msaA3mText` inputs (200 ms)
  let pdbTextDraft = $state('');
  let sequenceDraft = $state('');
  let msaA3mDraft = $state('');
  let pendingTextCommit: {
    timer: ReturnType<typeof setTimeout>;
    nodeId: string;
    key: 'pdbText' | 'sequence' | 'msaA3mText';
    value: string;
    baseParams: Record<string, unknown>;
  } | null = null;

  function flushPendingTextCommit() {
    if (!pendingTextCommit) return;
    const pending = pendingTextCommit;
    clearTimeout(pending.timer);
    pendingTextCommit = null;
    onUpdate(pending.nodeId, { ...pending.baseParams, [pending.key]: pending.value });
  }

  function scheduleTextParam(key: 'pdbText' | 'sequence' | 'msaA3mText', value: string) {
    if (key === 'pdbText') pdbTextDraft = value;
    else if (key === 'sequence') sequenceDraft = value;
    else msaA3mDraft = value;
    const targetNodeId = node.id;
    const baseParams = { ...node.params };
    if (pendingTextCommit) {
      if (pendingTextCommit.nodeId !== targetNodeId || pendingTextCommit.key !== key) {
        flushPendingTextCommit();
      } else {
        clearTimeout(pendingTextCommit.timer);
      }
    }
    const timer = setTimeout(() => {
      if (pendingTextCommit?.timer === timer) {
        pendingTextCommit = null;
      }
      onUpdate(targetNodeId, { ...baseParams, [key]: value });
    }, 200);
    pendingTextCommit = { timer, nodeId: targetNodeId, key, value, baseParams };
  }

  $effect(() => {
    const id = node.id;
    const extPdb = String(node.params.pdbText ?? '');
    const extSeq = String(node.params.sequence ?? '');
    const extA3m = String(node.params.msaA3mText ?? '');
    if (!pendingTextCommit || pendingTextCommit.nodeId !== id || pendingTextCommit.key !== 'pdbText') {
      pdbTextDraft = extPdb;
    }
    if (!pendingTextCommit || pendingTextCommit.nodeId !== id || pendingTextCommit.key !== 'sequence') {
      sequenceDraft = extSeq;
    }
    if (!pendingTextCommit || pendingTextCommit.nodeId !== id || pendingTextCommit.key !== 'msaA3mText') {
      msaA3mDraft = extA3m;
    }
  });

  onDestroy(() => {
    flushPendingTextCommit();
  });

  const SCORER_MOTIF_RMSD_SPEC: NumericFieldSpec = {
    min: 0.1,
    max: 10,
    step: 0.1,
    integer: false,
    defaultValue: 1.0,
    unit: 'Å',
  };
  const SCORER_IPAE_SPEC: NumericFieldSpec = {
    min: 1,
    max: 31.75,
    step: 0.5,
    integer: false,
    defaultValue: 10.0,
    unit: 'Å',
  };
  const SCORER_IPSAE_SPEC: NumericFieldSpec = {
    min: 0,
    max: 1,
    step: 0.05,
    integer: false,
    defaultValue: 0,
  };

  function localFoldMaxRecycles(modelId: string, family: string): number {
    return modelId === 'mol1' || family === 'af2-multimer' ? 20 : 10;
  }

  function localFoldDefaultRecyclesHint(modelId: string, family: string): string {
    if (family === 'af2-multimer') return 'Upstream default: 20 (max 20)';
    if (modelId === 'mol1') return 'Upstream default: 10 (max 20)';
    if (family === 'af3') return 'Upstream default: 10 (max 10)';
    return 'Upstream default: 3 (max 10)';
  }

  function formatLocalFoldMsaBadge(modeRaw: unknown): string {
    const mode = String(modeRaw ?? 'single') as MsaMode;
    if (mode === 'mmseqs2') return 'MMseqs2 MSA';
    if (mode === 'target-only') return 'Target-only MSA';
    if (mode === 'a3m') return 'Custom .a3m MSA';
    return 'Single-sequence (no MSA)';
  }

  function parseA3mSequenceSummary(raw: string): { seqCount: number; queryLength: number } {
    const trimmed = raw.trim();
    if (!trimmed) return { seqCount: 0, queryLength: 0 };
    const lines = trimmed.split(/\r?\n/);
    let seqCount = 0;
    let firstSeq = '';
    let curSeq = '';
    let seenHeader = false;
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#')) continue;
      if (line.startsWith('>')) {
        if (seenHeader && curSeq.length > 0) {
          seqCount++;
          if (!firstSeq) firstSeq = curSeq;
        }
        seenHeader = true;
        curSeq = '';
      } else {
        const clean = line.replace(/\s+/g, '');
        if (seenHeader) {
          curSeq += clean;
        } else {
          seqCount++;
          if (!firstSeq) firstSeq = clean;
        }
      }
    }
    if (seenHeader && curSeq.length > 0) {
      seqCount++;
      if (!firstSeq) firstSeq = curSeq;
    }
    const queryLength = firstSeq.replace(/[a-z.-]/g, '').length;
    return { seqCount, queryLength };
  }

  async function handleA3mFileUpload(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    flushPendingTextCommit();
    const targetNodeId = node.id;
    const baseParams = { ...node.params };
    try {
      const text = await file.text();
      msaA3mDraft = text;
      onUpdate(targetNodeId, { ...baseParams, msaMode: 'a3m', msaA3mText: text });
    } finally {
      input.value = '';
    }
  }

  function isAutoLocalFoldLabel(label: string | undefined): { auto: boolean; suffix: string } {
    const raw = (label ?? '').trim();
    if (!raw || raw === 'LocalFold') return { auto: true, suffix: '' };
    const parenMatch = raw.match(/^(.*?)\s*(\((?:MPNN|Native|AF2|RF3)\))$/);
    const base = parenMatch ? parenMatch[1].trim() : raw;
    const suffix = parenMatch ? ` ${parenMatch[2]}` : '';
    if (
      base === 'LocalFold' ||
      /^LocalFold\s*\([^)]+\)$/i.test(raw) ||
      LOCALFOLD_MODEL_OPTIONS.some((m) => m.shortLabel === base || m.label === base)
    ) {
      return { auto: true, suffix: /^LocalFold\s*\(/i.test(raw) && !parenMatch ? '' : suffix };
    }
    return { auto: false, suffix: '' };
  }

  function isAutoMpnnLabel(label: string | undefined): boolean {
    const raw = (label ?? '').trim();
    if (!raw) return true;
    return (
      raw === 'ProteinMPNN' ||
      MPNN_VARIANT_OPTIONS.some((v) => v.shortLabel === raw || v.label === raw)
    );
  }

  function isAutoPdbLabel(label: string | undefined, prevFilename?: string): boolean {
    const raw = (label ?? '').trim();
    if (!raw) return true;
    if (
      raw === 'PDB Input' ||
      raw === 'Load Structure (PDB / mmCIF)' ||
      raw === 'Demo 3-Helix (53 aa)' ||
      /^PDB:\s/i.test(raw) ||
      /^RCSB\s+[A-Z0-9]{4}$/i.test(raw) ||
      /^Custom:\s/i.test(raw)
    ) {
      return true;
    }
    if (prevFilename && raw === prevFilename.trim()) return true;
    return false;
  }

  let activeModelBadge = $derived.by((): string => {
    const p = node.params ?? {};
    switch (node.adapterId) {
      case 'localfold-af2':
        return `${selectedLocalFoldMeta?.label ?? LOCALFOLD_MODEL_OPTIONS[0].label} · ${formatLocalFoldMsaBadge(p.msaMode)}`;
      case 'rfd4-proteina':
        return rfd4NeedsEncoder
          ? 'RFD4-Proteina · Conditioned + structure encoder'
          : 'RFD4-Proteina · All-atom co-design';
      case 'genie3':
        return 'Genie 3 · Backbone diffusion';
      case 'web-protein-mpnn': {
        const famMeta =
          MPNN_VARIANT_OPTIONS.find((v) => v.family === p.family) ?? MPNN_VARIANT_OPTIONS[0];
        const ckpt = String(p.modelName ?? 'v_48_020');
        return `${famMeta.label} · ${ckpt}`;
      }
      case 'designability-scorer':
        return 'Kabsch alignment · scRMSD / scTM';
      case 'pdb-input':
        return 'PDB / mmCIF structure input';
      case 'sequence-input':
        return 'FASTA sequence input';
      default: {
        const activeMode = getActiveModeSpec(manifest, p);
        return activeMode
          ? `${manifest?.shortName ?? node.adapterId} · ${activeMode.label}`
          : (manifest?.shortName ?? node.adapterId);
      }
    }
  });

  function setParam(key: string, value: unknown) {
    const next: Record<string, unknown> = { ...node.params, [key]: value };
    if (
      (node.adapterId === 'rfd4-proteina' || node.adapterId === 'genie3') &&
      ['mode', 'taskMode', 'contig', 'lockChain', 'hotspots', 'motifSpan', 'unindexedMotif', 'hbondMode', 'hbondSelection', 'cathCode', 'presetId'].includes(key)
    ) {
      next.userEditedTask = true;
    }
    onUpdate(node.id, next);
  }

  /**
   * Update ONLY numeric binder length token(s) after `/0`, leaving internal target linker
   * spans before `/0` untouched (e.g. `A1-50,5-5,A60-120,/0,70-70` -> `A1-50,5-5,A60-120,/0,90-90`).
   */
  function updateBinderLengthInContig(
    prevContig: string,
    newLen: number,
    clearBareLength: boolean,
  ): string {
    const trimmed = prevContig.trim();
    if (!trimmed) return '';
    if (trimmed.includes('/0')) {
      const tokens = trimmed.split(/[,\s]+/).filter(Boolean);
      const breakIdx = tokens.lastIndexOf('/0');
      if (breakIdx >= 0) {
        return tokens
          .map((tok, idx) =>
            idx > breakIdx && /^\d+(?:-\d+)?$/.test(tok) ? `${newLen}-${newLen}` : tok,
          )
          .join(',');
      }
      const slashIdx = trimmed.lastIndexOf('/0');
      const before = trimmed.slice(0, slashIdx + 2);
      const afterTokens = trimmed
        .slice(slashIdx + 2)
        .split(/[,\s]+/)
        .filter(Boolean)
        .map((tok) => (/^\d+(?:-\d+)?$/.test(tok) ? `${newLen}-${newLen}` : tok));
      return afterTokens.length > 0 ? `${before},${afterTokens.join(',')}` : trimmed;
    }
    if (clearBareLength && /^\d+(?:-\d+)?$/.test(trimmed)) {
      return '';
    }
    return trimmed;
  }

  function parseResidueTokens(raw: unknown): string[] {
    return String(raw ?? '')
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function removeResidueTokenAt(key: string, idx: number) {
    const tokens = parseResidueTokens(node.params[key]);
    tokens.splice(idx, 1);
    setParam(key, tokens.join(', '));
  }

  // Per-node PDB fetch & upload status so two `pdb-input` nodes never bleed busy/error/summary state
  interface PdbNodeStatus {
    busy: boolean;
    error: string | null;
    summary: string | null;
  }
  let pdbFetchStateByNode = $state<Record<string, PdbNodeStatus>>({});
  let curPdbStatus = $derived<PdbNodeStatus>(
    pdbFetchStateByNode[node.id] ?? { busy: false, error: null, summary: null },
  );

  function patchPdbStatus(targetNodeId: string, patch: Partial<PdbNodeStatus>) {
    const prev = pdbFetchStateByNode[targetNodeId] ?? { busy: false, error: null, summary: null };
    pdbFetchStateByNode = {
      ...pdbFetchStateByNode,
      [targetNodeId]: { ...prev, ...patch },
    };
  }

  async function handleFetchPdbId(overrideId?: string, benchmarkIdOverride?: string) {
    const rawId = overrideId ?? String(node.params.pdbId ?? '');
    const normId = normalizePdbId(rawId);
    const check = validatePdbId(normId);
    const targetNodeId = node.id;
    if (check.status === 'error') {
      patchPdbStatus(targetNodeId, { error: check.message ?? 'Invalid PDB ID' });
      return;
    }
    const baseParams = { ...node.params };
    const prevNormId = normalizePdbId(String(baseParams.pdbId ?? ''));
    const shouldClearCrop = overrideId !== undefined || normId !== prevNormId;
    const prevLabel = node.label;
    const prevFilename = String(baseParams.filename ?? '');
    const nextBenchId =
      benchmarkIdOverride !== undefined
        ? benchmarkIdOverride
        : overrideId
          ? ''
          : String(baseParams.benchmarkId ?? '');
    patchPdbStatus(targetNodeId, { busy: true, error: null, summary: null });
    try {
      const res = await fetchStructureByPdbId(normId);
      if (isAutoPdbLabel(prevLabel, prevFilename)) {
        const quickEx = QUICK_PDB_EXAMPLES.find((ex) => ex.id === res.pdbId);
        onUpdateLabel?.(
          targetNodeId,
          quickEx ? `PDB: ${quickEx.name} (${res.pdbId})` : `RCSB ${res.pdbId}`,
        );
      }
      onUpdate(targetNodeId, {
        ...baseParams,
        mode: 'pdb-id',
        pdbId: res.pdbId,
        fetchedPdbId: res.pdbId,
        filename: res.filename,
        pdbText: res.text,
        benchmarkId: nextBenchId,
        ...(shouldClearCrop ? { cropSegments: '' } : {}),
      });
      const extras = [
        res.ligandNames.length > 0 ? `ligands: ${res.ligandNames.join(', ')}` : '',
        res.nucleicAcidChains.length > 0 ? `NA chains: ${res.nucleicAcidChains.join(', ')}` : '',
      ].filter(Boolean);
      patchPdbStatus(targetNodeId, {
        summary:
          `Loaded ${res.filename} · ${res.nResidues} aa · ${res.chainCount} chain${res.chainCount === 1 ? '' : 's'}` +
          (extras.length > 0 ? ` · ${extras.join(' · ')}` : ''),
      });
    } catch (err) {
      patchPdbStatus(targetNodeId, {
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      patchPdbStatus(targetNodeId, { busy: false });
    }
  }

  function clearMissingUploadExpectation() {
    flushPendingTextCommit();
    const next: Record<string, unknown> = { ...node.params };
    next.missingUpload = false;
    delete next.expectedPdbSha256;
    delete next.expectedFilename;
    delete next.missingUploadFilename;
    delete next.missingUploadSha256;
    patchPdbStatus(node.id, { error: null, summary: 'Cleared expected upload digest.' });
    onUpdate(node.id, next);
  }

  async function handleFileUpload(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    flushPendingTextCommit();
    const targetNodeId = node.id;
    const baseParams = { ...node.params };
    const prevLabel = node.label;
    const prevFilename = String(baseParams.filename ?? '');
    const expectedShaRaw = String(
      baseParams.expectedPdbSha256 ?? baseParams.missingUploadSha256 ?? '',
    )
      .trim()
      .toLowerCase();
    try {
      const text = await file.text();
      if (isAutoPdbLabel(prevLabel, prevFilename)) {
        onUpdateLabel?.(targetNodeId, `PDB: ${file.name}`);
      }
      const nextParams: Record<string, unknown> = {
        ...baseParams,
        mode: 'custom',
        filename: file.name,
        pdbText: text,
        cropSegments: '',
        benchmarkId: '',
      };
      if (expectedShaRaw && isValidSha256Hex(expectedShaRaw)) {
        const actualSha = computeUploadDigest(text);
        if (actualSha === expectedShaRaw) {
          nextParams.missingUpload = false;
          delete nextParams.expectedPdbSha256;
          delete nextParams.expectedFilename;
          delete nextParams.missingUploadFilename;
          delete nextParams.missingUploadSha256;
          patchPdbStatus(targetNodeId, {
            error: null,
            summary: `Verified ${file.name} (SHA-256 ${actualSha.slice(0, 12)}…)`,
          });
        } else {
          nextParams.missingUpload = true;
          patchPdbStatus(targetNodeId, {
            error: `Digest mismatch: uploaded ${file.name} (${actualSha.slice(0, 12)}…) does not match expected ${expectedShaRaw.slice(0, 12)}…. Re-upload the matching file or click Clear expectation.`,
            summary: null,
          });
        }
      } else {
        nextParams.missingUpload = false;
        delete nextParams.expectedPdbSha256;
        delete nextParams.expectedFilename;
        delete nextParams.missingUploadFilename;
        delete nextParams.missingUploadSha256;
        patchPdbStatus(targetNodeId, { error: null, summary: `Loaded ${file.name}` });
      }
      pdbTextDraft = text;
      onUpdate(targetNodeId, nextParams);
    } finally {
      input.value = '';
    }
  }
</script>

<div class="inspector" class:embedded>
  <div class="head">
    <div class="title-wrap">
      <span class="stage-dot" style="background: {stageColor}"></span>
      {#if onUpdateLabel}
        <input
          type="text"
          class="title-input"
          aria-label="Block title"
          title="Click to rename this block"
          {disabled}
          placeholder={manifest?.shortName || node.adapterId}
          value={titleDraft}
          oninput={(e) => {
            titleDraft = e.currentTarget.value;
            onUpdateLabel(node.id, titleDraft);
          }}
          onblur={() => {
            const trimmed = titleDraft.trim();
            if (trimmed !== titleDraft) {
              titleDraft = trimmed;
              onUpdateLabel(node.id, trimmed);
            }
          }}
          onkeydown={(e) => {
            if (e.key === 'Enter') {
              e.currentTarget.blur();
            }
          }}
        />
      {:else}
        <h3>{node.label || manifest?.displayName || node.adapterId}</h3>
      {/if}
      <span class="sub-kind" title="Underlying model / engine">{activeModelBadge}</span>
    </div>
    {#if (!embedded && onOpenStudio) || onRunUpTo}
      <div class="head-actions">
        {#if !embedded && onOpenStudio}
          <button
            type="button"
            class="studio-open-btn"
            onclick={() => {
              flushPendingTextCommit();
              onOpenStudio();
            }}
            title="Inspect what the model sees & paint residue constraints in Studio (Space)"
          >
            <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <rect x="1.8" y="2.4" width="12.4" height="11.2" rx="2" stroke="currentColor" stroke-width="1.4" />
              <path d="M6.2 2.6V13.4" stroke="currentColor" stroke-width="1.3" />
              <path d="M8.6 6H11.8M8.6 8.5H11.8M8.6 11H10.6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
            </svg>
            Inspect Studio
          </button>
        {/if}
        {#if onRunUpTo}
          <button
            type="button"
            class="run-here-btn"
            {disabled}
            onclick={() => {
              flushPendingTextCommit();
              onRunUpTo(node.id);
            }}
            title="Execute only upstream blocks up to this block (memoized for full runs)"
          >
            ▶ Run to here
          </button>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Primary Scientific Parameters -->
  <div class="fields">
    {#if node.adapterId === 'pdb-input'}
      <label class="wide">
        Source
        <select
          {disabled}
          value={String(node.params.mode ?? 'demo-bundle')}
          onchange={(e) => {
            flushPendingTextCommit();
            const nextMode = e.currentTarget.value;
            patchPdbStatus(node.id, { error: null, summary: null });
            if (isAutoPdbLabel(node.label, String(node.params.filename ?? ''))) {
              if (nextMode === 'demo-bundle') {
                onUpdateLabel?.(node.id, 'PDB Input');
              } else if (nextMode === 'pdb-id') {
                const norm = normalizePdbId(String(node.params.pdbId ?? ''));
                onUpdateLabel?.(
                  node.id,
                  validatePdbId(norm).status === 'ok' ? `RCSB ${norm}` : 'PDB Input',
                );
              } else if (nextMode === 'custom') {
                const fn = String(node.params.filename ?? '').trim();
                onUpdateLabel?.(node.id, fn && fn !== 'demo_3helix_53aa.pdb' ? `PDB: ${fn}` : 'PDB Input');
              }
            }
            onUpdate(node.id, {
              ...node.params,
              mode: nextMode,
              ...(nextMode === 'demo-bundle' ? { cropSegments: '', benchmarkId: '' } : {}),
            });
          }}
        >
          <option value="demo-bundle">Built-in 53 aa 3-helix bundle demo</option>
          <option value="pdb-id">Fetch from RCSB PDB ID (e.g. 1PRW, 5O45)</option>
          <option value="custom">Upload or paste PDB / mmCIF</option>
        </select>
      </label>

      {#if node.params.mode === 'pdb-id'}
        {@const pdbIdVal = String(node.params.pdbId ?? '')}
        {@const pdbIdCheck = validatePdbId(pdbIdVal)}
        {@const isAutoFetching = nodeState?.status === 'loading'}
        {@const isBusy = curPdbStatus.busy || isAutoFetching}
        {@const isAlreadyLoaded =
          pdbIdCheck.status === 'ok' &&
          (Boolean(liveStructure) ||
            (normalizePdbId(String(node.params.fetchedPdbId ?? '')) === normalizePdbId(pdbIdVal) &&
              String(node.params.pdbText ?? '').trim().length > 0))}
        <div class="full pdb-fetch-wrap">
          <span class="field-lbl">RCSB PDB ID</span>
          <div class="pdb-fetch-row">
            <input
              type="text"
              class:input-error={pdbIdVal.trim().length > 0 && pdbIdCheck.status === 'error'}
              disabled={disabled || isBusy}
              placeholder="e.g. 1PRW, 5O45, 4JHW"
              maxlength="12"
              value={pdbIdVal}
              oninput={(e) => {
                const nextPdbId = e.currentTarget.value.toUpperCase();
                patchPdbStatus(node.id, { error: null, summary: null });
                const norm = normalizePdbId(nextPdbId);
                if (isAutoPdbLabel(node.label, String(node.params.filename ?? ''))) {
                  onUpdateLabel?.(
                    node.id,
                    validatePdbId(norm).status === 'ok' ? `RCSB ${norm}` : 'PDB Input',
                  );
                }
                onUpdate(node.id, {
                  ...node.params,
                  pdbId: nextPdbId,
                  benchmarkId: '',
                  cropSegments: '',
                });
              }}
              onkeydown={(e) => {
                if (e.key === 'Enter' && pdbIdCheck.status === 'ok' && !disabled && !isBusy) {
                  e.preventDefault();
                  void handleFetchPdbId();
                }
              }}
            />
            <button
              type="button"
              class="fetch-btn"
              disabled={disabled || isBusy || pdbIdCheck.status === 'error'}
              onclick={() => void handleFetchPdbId()}
            >
              {isBusy ? 'Fetching…' : isAlreadyLoaded ? '↻ Refetch' : '↓ Fetch'}
            </button>
          </div>
          <div class="quick-chips">
            <span class="quick-lbl">Examples:</span>
            {#each QUICK_PDB_EXAMPLES as ex (ex.id)}
              <button
                type="button"
                class="quick-chip"
                disabled={disabled || isBusy}
                onclick={() => void handleFetchPdbId(ex.id, '')}
                title={`Fetch ${ex.id} (${ex.name}) from RCSB PDB`}
              >
                {ex.id} · {ex.name}
              </button>
            {/each}
          </div>
          <label class="bench-pick">
            <span>Or pick a curated benchmark target</span>
            <select
              disabled={disabled || isBusy}
              value={String(node.params.benchmarkId ?? '')}
              onchange={(e) => {
                const benchId = e.currentTarget.value;
                const bench = getBenchmarkById(benchId);
                if (!bench) return;
                patchPdbStatus(node.id, { error: null, summary: null });
                if (isAutoPdbLabel(node.label, String(node.params.filename ?? ''))) {
                  onUpdateLabel?.(node.id, `PDB: ${bench.name} (${bench.pdbId})`);
                }
                onUpdate(node.id, {
                  ...node.params,
                  mode: 'pdb-id',
                  pdbId: bench.pdbId,
                  fetchedPdbId: '',
                  filename: `${bench.pdbId}.cif`,
                  pdbText: '',
                  benchmarkId: bench.id,
                  cropSegments: bench.cropSegments ?? '',
                });
              }}
            >
              <option value="">Select a curated benchmark PDB…</option>
              <optgroup label="PPI Binders (RFdiffusion / Genie 3 / BindCraft)">
                {#each BENCHMARK_TARGETS.filter((b) => b.category === 'binder') as b (b.id)}
                  <option value={b.id}>{b.pdbId} — {b.name} (~{b.approxTokens} tok)</option>
                {/each}
              </optgroup>
              <optgroup label="Motif Scaffolding (RFdiffusion / MotifBench)">
                {#each BENCHMARK_TARGETS.filter((b) => b.category === 'motif') as b (b.id)}
                  <option value={b.id}>{b.pdbId} — {b.name} (~{b.approxTokens} tok)</option>
                {/each}
              </optgroup>
              <optgroup label="AME Enzymes, Ligands & DNA (RFD4 All-Atom)">
                {#each BENCHMARK_TARGETS.filter((b) => b.category === 'enzyme-ligand') as b (b.id)}
                  <option value={b.id}>{b.pdbId} — {b.name} (~{b.approxTokens} tok)</option>
                {/each}
              </optgroup>
            </select>
          </label>
          {#if curPdbStatus.error}
            <span class="field-hint err">{curPdbStatus.error}</span>
          {:else if nodeState?.status === 'error' && nodeState.error}
            <span class="field-hint err">{nodeState.error}</span>
          {:else if isBusy}
            <span class="field-hint">Fetching {normalizePdbId(pdbIdVal) || 'structure'} from RCSB PDB for 3D preview…</span>
          {:else if curPdbStatus.summary}
            <span class="field-hint ok">✓ {curPdbStatus.summary}</span>
          {:else if liveStructure}
            <span class="field-hint ok">
              ✓ Loaded {liveStructure.label} · {liveStructure.diagnostics?.chainCount ?? 1} chain{(liveStructure.diagnostics?.chainCount ?? 1) === 1 ? '' : 's'} (previewed in 3D)
            </span>
          {:else if isAlreadyLoaded}
            <span class="field-hint ok">✓ Ready ({String(node.params.filename || `${normalizePdbId(pdbIdVal)}.cif`)} loaded)</span>
          {:else if pdbIdCheck.status === 'error'}
            <span class="field-hint" class:err={pdbIdVal.trim().length > 0}>{pdbIdCheck.message}</span>
          {:else}
            <span class="field-hint">Enter a 4-character PDB ID to auto-preview in 3D.</span>
          {/if}
        </div>
      {:else if node.params.mode === 'custom'}
        {@const pdbCheck = validateStructureText(pdbTextDraft)}
        {@const expectedFilename = String(
          node.params.expectedFilename ?? node.params.missingUploadFilename ?? node.params.filename ?? '',
        ).trim()}
        {@const expectedSha = String(
          node.params.expectedPdbSha256 ?? node.params.missingUploadSha256 ?? '',
        ).trim()}
        {@const showMissingUploadCard = Boolean(node.params.missingUpload) || Boolean(expectedSha)}
        {#if showMissingUploadCard}
          <div class="missing-upload-card full" role="alert">
            <div class="mu-head">
              <span class="mu-title">⚠ Missing Uploaded File</span>
              {#if expectedSha}
                <code class="mu-sha" title={`Expected SHA-256: ${expectedSha}`}>
                  SHA-256 {expectedSha.slice(0, 12)}…
                </code>
              {/if}
            </div>
            <p class="mu-body">
              This shared recipe expects custom structure
              {#if expectedFilename}
                <code class="filename-wrap">{expectedFilename}</code>
              {:else}
                an omitted local PDB/mmCIF upload
              {/if}.
              Re-upload the matching file to verify integrity, or clear the expectation to use a different input.
            </p>
            <div class="mu-actions">
              <label class="mu-upload-btn" class:is-disabled={disabled}>
                <span>Re-upload matching file</span>
                <input
                  type="file"
                  accept=".pdb,.cif,.mmcif,.ent,.txt"
                  hidden
                  {disabled}
                  onchange={handleFileUpload}
                />
              </label>
              <button
                type="button"
                class="mu-clear-btn"
                {disabled}
                onclick={clearMissingUploadExpectation}
              >
                Clear expectation
              </button>
            </div>
          </div>
        {/if}
        <label class="file-row full">
          <span>Upload .pdb / .cif</span>
          <input type="file" accept=".pdb,.cif,.mmcif,.ent,.txt" {disabled} onchange={handleFileUpload} />
        </label>
        <label class="full">
          <span class="filename-wrap">PDB / mmCIF text ({String(node.params.filename || 'custom.pdb')})</span>
          <textarea
            rows="4"
            class:input-error={pdbCheck.status === 'error'}
            {disabled}
            placeholder="Paste ATOM lines or mmCIF _atom_site block…"
            value={pdbTextDraft}
            oninput={(e) => scheduleTextParam('pdbText', e.currentTarget.value)}
            onblur={flushPendingTextCommit}
          ></textarea>
          {#if curPdbStatus.error}
            <span class="field-hint err">{curPdbStatus.error}</span>
          {:else if nodeState?.status === 'error' && nodeState.error}
            <span class="field-hint err">{nodeState.error}</span>
          {:else if curPdbStatus.summary}
            <span class="field-hint ok">✓ {curPdbStatus.summary}</span>
          {/if}
          {#if pdbCheck.message}
            <span class="field-hint" class:err={pdbCheck.status === 'error'} class:ok={pdbCheck.status === 'ok'}>
              {pdbCheck.message}
            </span>
          {/if}
        </label>
      {/if}

      {#if node.params.mode === 'pdb-id' || node.params.mode === 'custom'}
        {@const cropVal = String(node.params.cropSegments ?? '')}
        {@const droppedCropLigands = (() => {
          const trimCrop = cropVal.trim();
          const rawTxt = String(node.params.pdbText ?? '').trim();
          if (!trimCrop || !rawTxt) return [] as string[];
          try {
            const fullLigs = parseStructureText(rawTxt).diagnostics.ligandNames;
            if (fullLigs.length === 0) return [] as string[];
            const keptLigs = new Set(
              (liveStructure?.diagnostics?.ligandNames ?? []).map((l) => l.toUpperCase()),
            );
            return fullLigs.filter((l) => !keptLigs.has(l.toUpperCase()));
          } catch {
            return [] as string[];
          }
        })()}
        <div class="full pdb-fetch-wrap">
          <span class="field-lbl">Segment / Chain Crop (optional)</span>
          <div class="pdb-fetch-row">
            <input
              type="text"
              id="pdb-crop-segments"
              name="cropSegments"
              {disabled}
              placeholder="Full structure (or e.g. A, A:BEN, A:CA)"
              value={cropVal}
              oninput={(e) => setParam('cropSegments', e.currentTarget.value)}
            />
            {#if cropVal.trim().length > 0}
              <button
                type="button"
                class="fetch-btn"
                {disabled}
                onclick={() => setParam('cropSegments', '')}
                title="Clear crop and show the full RCSB structure"
              >
                Full PDB
              </button>
            {/if}
          </div>
          {#if droppedCropLigands.length > 0}
            <div class="crop-lig-warn" role="status">
              <span>
                ⚠ Crop omits ligands/ions: <b>{droppedCropLigands.join(', ')}</b> (downstream LigandMPNN/LASErMPNN will run apo).
              </span>
              <button
                type="button"
                class="fetch-btn"
                {disabled}
                onclick={() =>
                  setParam(
                    'cropSegments',
                    `${cropVal.trim()}, ${droppedCropLigands.join(', ')}`,
                  )}
                title="Append omitted ligands/ions to the crop"
              >
                + Keep {droppedCropLigands.join(', ')}
              </button>
            </div>
          {/if}
          <span class="field-hint">
            Extracts exact chains or residue spans while preserving original PDB residue numbers; include ligand codes (e.g. <code>A, BEN, CA</code>) to retain holo ligands.
          </span>
        </div>
      {/if}

    {:else if node.adapterId === 'sequence-input'}
      {@const seqCheck = validateSequenceText(sequenceDraft)}
      <label class="full">
        Amino acid sequence (plain or multi-FASTA)
        <textarea
          rows="5"
          class:input-error={seqCheck.status === 'error'}
          class:input-warn={seqCheck.status === 'warn'}
          {disabled}
          placeholder=">seq1&#10;MKTAYIAKQRQISFVK..."
          value={sequenceDraft}
          oninput={(e) => scheduleTextParam('sequence', e.currentTarget.value)}
          onblur={flushPendingTextCommit}
        ></textarea>
        <span
          class="field-hint"
          class:err={seqCheck.status === 'error'}
          class:warn={seqCheck.status === 'warn'}
          class:ok={seqCheck.status === 'ok'}
        >
          {seqCheck.message}
        </span>
      </label>

      <details class="adv-details full">
        <summary>Advanced parameters</summary>
        <div class="adv-fields">
          <label class="wide">
            Label
            <input
              type="text"
              {disabled}
              value={String(node.params.label ?? 'custom-seq-1')}
              oninput={(e) => setParam('label', e.currentTarget.value)}
            />
          </label>
        </div>
      </details>

    {:else if node.adapterId === 'rfd4-proteina'}
      {@const rfdMode = String(node.params.mode ?? 'unconditional')}
      {@const rfdNumDesigns = Math.max(1, Math.min(MAX_DESIGNS, Math.round(Number(node.params.numDesigns ?? 1)) || 1))}
      {@const hotspotTokens = parseResidueTokens(node.params.hotspots)}
      {@const motifTokens = parseResidueTokens(node.params.motifSpan)}
      {@const hasCustomContig =
        (rfdMode === 'motif' || rfdMode === 'binder') && String(node.params.contig ?? '').trim().length > 0}
      <label class="full">
        Design Task
        <select
          {disabled}
          value={rfdMode}
          onchange={(e) => {
            const nextMode = e.currentTarget.value;
            const prevContig = String(node.params.contig ?? '').trim();
            if (nextMode === 'binder' && /^\d+(?:-\d+)?$/.test(prevContig)) {
              onUpdate(node.id, {
                ...node.params,
                mode: nextMode,
                contig: '',
                userEditedTask: true,
              });
              return;
            }
            setParam('mode', nextMode);
          }}
        >
          <option value="unconditional">Unconditional Monomer</option>
          <option value="motif">Motif Scaffolding (fixed residues)</option>
          <option value="binder">PPI Binder (target + hotspots)</option>
          <option value="partial">Partial Diffusion (refine structure)</option>
          <option value="inverse-folding">Inverse Folding (design sequence)</option>
          <option value="preset">All-Atom / Built-In Task Preset</option>
        </select>
      </label>

      {#if siblingGeneratorIds.length > 0 && onSyncTaskToSiblings}
        <div class="full">
          <button
            type="button"
            class="sync-sib-btn"
            {disabled}
            onclick={() => onSyncTaskToSiblings(node.id)}
            title="Copy shared task settings (mode, contig, hotspots, motifs) to sibling generator(s) on the same structure"
          >
            ⇄ Apply task to sibling generator{siblingGeneratorIds.length > 1 ? 's' : ''} ({siblingGeneratorIds.length})
          </button>
        </div>
      {/if}

      {#if rfdMode === 'unconditional'}
        <label>
          Length (aa)
          <NumericInput
            {disabled}
            ariaLabel="Length (aa)"
            spec={NUMERIC_SPECS['rfd4-proteina'].length}
            value={Number(node.params.length ?? 70)}
            onChange={(v) => setParam('length', v)}
          />
        </label>
        <label>
          CATH Fold (optional)
          <input
            type="text"
            {disabled}
            placeholder="e.g. 3.40.50"
            value={String(node.params.cathCode ?? '')}
            oninput={(e) => setParam('cathCode', e.currentTarget.value)}
          />
        </label>
      {:else if rfdMode === 'motif'}
        <div class="chip-field full">
          <label>
            Fixed Motif Spans (click 3D residues or enter e.g. A16-35, A52-71)
            <input
              type="text"
              {disabled}
              placeholder="A16-35, A52-71"
              value={String(node.params.motifSpan ?? '')}
              oninput={(e) => setParam('motifSpan', e.currentTarget.value)}
            />
          </label>
          {#if motifTokens.length > 0}
            <div class="res-chips">
              {#each motifTokens as tok, i (`rfd_motif_${tok}_${i}`)}
                {@const badTok = !isValidResidueToken(tok)}
                <span class="res-chip" class:invalid-chip={badTok}>
                  {tok}
                  <button type="button" class="chip-x" {disabled} aria-label={`Remove motif ${tok}`} onclick={() => removeResidueTokenAt('motifSpan', i)}>×</button>
                </span>
              {/each}
            </div>
          {/if}
        </div>
        <label class="inline-check full">
          <input
            type="checkbox"
            {disabled}
            checked={Boolean(node.params.unindexedMotif)}
            onchange={(e) => setParam('unindexedMotif', e.currentTarget.checked)}
          />
          Unindexed motif (auto-place catalytic residues, e.g. AME enzymes)
        </label>
      {:else if rfdMode === 'binder'}
        <label>
          Target Chain
          <input
            type="text"
            {disabled}
            placeholder="A"
            value={String(node.params.lockChain ?? 'A')}
            oninput={(e) => setParam('lockChain', e.currentTarget.value.toUpperCase())}
          />
        </label>
        <label>
          Binder Length (aa)
          <NumericInput
            {disabled}
            ariaLabel="Binder Length (aa)"
            spec={NUMERIC_SPECS['rfd4-proteina'].length}
            value={Number(node.params.length ?? 70)}
            onChange={(v) => {
              const prevContig = String(node.params.contig ?? '');
              const nextContig = updateBinderLengthInContig(prevContig, v, true);
              onUpdate(node.id, {
                ...node.params,
                length: v,
                contig: nextContig,
                userEditedTask: true,
              });
            }}
          />
        </label>
        <div class="chip-field full">
          <label>
            Binding Hotspots (click 3D residues or enter e.g. A54, A56, A115)
            <input
              type="text"
              {disabled}
              placeholder="A54, A56, A66, A115"
              value={String(node.params.hotspots ?? '')}
              oninput={(e) => setParam('hotspots', e.currentTarget.value)}
            />
          </label>
          {#if hotspotTokens.length > 0}
            <div class="res-chips">
              {#each hotspotTokens as tok, i (`rfd_hot_${tok}_${i}`)}
                {@const badTok = !isValidResidueToken(tok)}
                <span class="res-chip" class:invalid-chip={badTok}>
                  {tok}
                  <button type="button" class="chip-x" {disabled} aria-label={`Remove hotspot ${tok}`} onclick={() => removeResidueTokenAt('hotspots', i)}>×</button>
                </span>
              {/each}
            </div>
          {/if}
        </div>
      {:else if rfdMode === 'preset'}
        <label class="full">
          Built-In All-Atom Task Preset
          <select
            {disabled}
            value={String(node.params.presetId ?? 'custom_cath_monomer')}
            onchange={(e) => setParam('presetId', e.currentTarget.value)}
          >
            {#each RFD4_BENCHMARK_OPTIONS as opt (opt.id)}
              <option value={opt.id}>[{opt.category}] {opt.label}</option>
            {/each}
          </select>
        </label>
      {:else if rfdMode === 'partial'}
        <label class="full">
          Denoising step fraction (nonlinear schedule)
          <NumericInput
            {disabled}
            ariaLabel="Denoising step fraction"
            spec={NUMERIC_SPECS['rfd4-proteina'].partialFraction}
            value={Number(node.params.partialFraction ?? 0.5)}
            onChange={(v) => setParam('partialFraction', v)}
          />
        </label>
      {:else if rfdMode === 'inverse-folding'}
        <span class="field-hint full">
          Redesigns sequence for the wired input backbone using RFD4's structure encoder (~138 MB lazy load).
        </span>
      {/if}

      {#if conditioningSourceSummary?.active && conditioningSourceSummary.hasTemplate}
        <div
          class="source-res-badge full"
          class:is-warn={conditioningSourceSummary.count === 0}
        >
          {#if conditioningSourceSummary.count > 0}
            ✓ Resolved <b>{conditioningSourceSummary.count}</b> source {conditioningSourceSummary.kindLabel} residue{conditioningSourceSummary.count === 1 ? '' : 's'}
            {#if conditioningSourceSummary.preview}
              <span class="source-res-list">({conditioningSourceSummary.preview})</span>
            {/if}
          {:else}
            ⚠ 0 template residues matched current {conditioningSourceSummary.kindLabel} selection
          {/if}
        </div>
      {/if}

      {#if !hasStructureInput && (rfdMode === 'motif' || rfdMode === 'binder' || rfdMode === 'partial' || rfdMode === 'inverse-folding')}
        <span class="field-hint err full">
          Wire a PDB Input (or upstream structure) into RFD4's <code>structure</code> port for {rfdMode} mode.
        </span>
      {/if}

      <label>
        Backbones
        <NumericInput
          {disabled}
          ariaLabel="Backbones (numDesigns)"
          spec={NUM_DESIGNS_SPEC}
          value={rfdNumDesigns}
          onChange={(v) => setParam('numDesigns', v)}
        />
      </label>
      {#if rfdNumDesigns > 1}
        <div class="fanout-chip full" role="status">
          <span class="fanout-dot" aria-hidden="true"></span>
          <span>
            <b>{rfdNumDesigns} backbones</b> · weights loaded once (B=1 sequential loop)
          </span>
        </div>
      {/if}

      <details class="adv-details full">
        <summary>
          Advanced parameters
          {#if hasCustomContig}
            <span class="adv-badge" title={`Custom contig: ${String(node.params.contig).trim()}`}>Custom contig</span>
          {/if}
        </summary>
        <div class="adv-fields">
          {#if rfdMode === 'motif' || rfdMode === 'binder'}
            <label class="full">
              {rfdMode === 'binder' ? 'Binder Contig (optional override)' : 'Contig (optional override)'}
              <input
                type="text"
                {disabled}
                placeholder={rfdMode === 'binder'
                  ? 'Auto from template (or e.g. A17-132,/0,70-70)'
                  : 'e.g. 10-10,A16-35,16-16,A52-71,10-10'}
                value={String(node.params.contig ?? '')}
                oninput={(e) => setParam('contig', e.currentTarget.value)}
              />
            </label>
          {/if}
          <label>
            Steps
            <NumericInput
              {disabled}
              ariaLabel="Diffusion steps"
              spec={NUMERIC_SPECS['rfd4-proteina'].nsteps}
              value={Number(node.params.nsteps ?? 60)}
              onChange={(v) => setParam('nsteps', v)}
            />
          </label>
          <label>
            Recycle
            <select
              {disabled}
              value={String(node.params.recycleSchedule ?? 'low')}
              onchange={(e) => setParam('recycleSchedule', e.currentTarget.value)}
            >
              <option value="none">None (fastest)</option>
              <option value="low">Low (balanced)</option>
              <option value="medium">Medium (higher quality)</option>
              <option value="high">High (highest fidelity)</option>
            </select>
          </label>
          <label>
            Seed
            <NumericInput
              {disabled}
              ariaLabel="Seed"
              spec={NUMERIC_SPECS['rfd4-proteina'].seed}
              value={Number(node.params.seed ?? 42)}
              onChange={(v) => setParam('seed', v)}
            />
          </label>
          <label
            class="inline-check"
            class:is-muted={rfdMode === 'inverse-folding'}
            title={rfdMode === 'inverse-folding'
              ? 'Early stop is disabled in inverse-folding because the backbone coordinates are fixed.'
              : 'Stop denoising early if backbone Cα displacement converges'}
          >
            <input
              type="checkbox"
              disabled={disabled || rfdMode === 'inverse-folding'}
              checked={rfdMode === 'inverse-folding' ? false : Boolean(node.params.earlyStop)}
              onchange={(e) => setParam('earlyStop', e.currentTarget.checked)}
            />
            Early stop on convergence
            {#if rfdMode === 'inverse-folding'}
              <span class="field-hint">(disabled in inverse-folding)</span>
            {/if}
          </label>
          <label>
            H-Bond Constraint
            <select
              {disabled}
              value={String(node.params.hbondMode ?? 'none')}
              onchange={(e) => setParam('hbondMode', e.currentTarget.value)}
            >
              <option value="none">None</option>
              <option value="ACCEPTOR">Acceptor</option>
              <option value="DONOR">Donor</option>
              <option value="BOTH">Both</option>
            </select>
          </label>
          {#if String(node.params.hbondMode ?? 'none') !== 'none'}
            <label class="full">
              H-Bond Target Atoms (e.g. <code>B/SAM/*/*</code>)
              <input
                type="text"
                {disabled}
                placeholder="B/SAM/*/*"
                value={String(node.params.hbondSelection ?? '')}
                oninput={(e) => setParam('hbondSelection', e.currentTarget.value)}
              />
            </label>
          {/if}
          <label class="full">
            Custom constraint spec (YAML / JSON)
            <textarea
              rows="3"
              {disabled}
              placeholder="Optional raw RFD4 YAML or JSON spec override."
              value={String(node.params.customSpecYaml ?? '')}
              oninput={(e) => setParam('customSpecYaml', e.currentTarget.value)}
            ></textarea>
          </label>
        </div>
      </details>

    {:else if node.adapterId === 'genie3'}
      {@const g3Mode = String(node.params.taskMode ?? 'unconditional')}
      {@const g3NumDesigns = Math.max(1, Math.min(MAX_DESIGNS, Math.round(Number(node.params.numDesigns ?? 1)) || 1))}
      {@const g3Hotspots = parseResidueTokens(node.params.hotspots)}
      {@const g3Motifs = parseResidueTokens(node.params.motifSpan)}
      {@const g3HasCustomContig =
        (g3Mode === 'motif' || g3Mode === 'binder') && String(node.params.contig ?? '').trim().length > 0}
      <label class="full">
        Design Task
        <select
          {disabled}
          value={g3Mode}
          onchange={(e) => setParam('taskMode', e.currentTarget.value)}
        >
          <option value="unconditional">Unconditional Monomer</option>
          <option value="motif">Motif Scaffolding (fixed residues)</option>
          <option value="binder">PPI Binder (target + hotspots)</option>
        </select>
      </label>

      {#if siblingGeneratorIds.length > 0 && onSyncTaskToSiblings}
        <div class="full">
          <button
            type="button"
            class="sync-sib-btn"
            {disabled}
            onclick={() => onSyncTaskToSiblings(node.id)}
            title="Copy shared task settings (mode, contig, hotspots, motifs) to sibling generator(s) on the same structure"
          >
            ⇄ Apply task to sibling generator{siblingGeneratorIds.length > 1 ? 's' : ''} ({siblingGeneratorIds.length})
          </button>
        </div>
      {/if}

      {#if g3Mode === 'unconditional'}
        <label>
          Length (aa)
          <NumericInput
            {disabled}
            ariaLabel="Length (aa)"
            spec={NUMERIC_SPECS.genie3.length}
            value={Number(node.params.length ?? 80)}
            onChange={(v) => setParam('length', v)}
          />
        </label>
      {:else if g3Mode === 'motif'}
        <div class="chip-field full">
          <label>
            Fixed Motif Spans (click 3D residues or enter e.g. A16-35, A52-71)
            <input
              type="text"
              {disabled}
              placeholder="A16-35, A52-71"
              value={String(node.params.motifSpan ?? '')}
              oninput={(e) => setParam('motifSpan', e.currentTarget.value)}
            />
          </label>
          {#if g3Motifs.length > 0}
            <div class="res-chips">
              {#each g3Motifs as tok, i (`g3_motif_${tok}_${i}`)}
                {@const badTok = !isValidResidueToken(tok)}
                <span class="res-chip" class:invalid-chip={badTok}>
                  {tok}
                  <button type="button" class="chip-x" {disabled} aria-label={`Remove motif ${tok}`} onclick={() => removeResidueTokenAt('motifSpan', i)}>×</button>
                </span>
              {/each}
            </div>
          {/if}
        </div>
      {:else if g3Mode === 'binder'}
        <label>
          Binder Length (aa)
          <NumericInput
            {disabled}
            ariaLabel="Binder Length (aa)"
            spec={NUMERIC_SPECS.genie3.length}
            value={Number(node.params.length ?? 80)}
            onChange={(v) => {
              const prevContig = String(node.params.contig ?? '');
              const nextContig = updateBinderLengthInContig(prevContig, v, false);
              onUpdate(node.id, {
                ...node.params,
                length: v,
                contig: nextContig,
                userEditedTask: true,
              });
            }}
          />
        </label>
        <label>
          Target Chain
          <input
            type="text"
            {disabled}
            placeholder="A"
            value={String(node.params.lockChain ?? 'A')}
            oninput={(e) => setParam('lockChain', e.currentTarget.value.toUpperCase())}
          />
        </label>
        <div class="chip-field full">
          <label>
            Binding Hotspots (click 3D residues or enter e.g. A54, A56, A115)
            <input
              type="text"
              {disabled}
              placeholder="A54, A56, A66, A115"
              value={String(node.params.hotspots ?? '')}
              oninput={(e) => setParam('hotspots', e.currentTarget.value)}
            />
          </label>
          {#if g3Hotspots.length > 0}
            <div class="res-chips">
              {#each g3Hotspots as tok, i (`g3_hot_${tok}_${i}`)}
                {@const badTok = !isValidResidueToken(tok)}
                <span class="res-chip" class:invalid-chip={badTok}>
                  {tok}
                  <button type="button" class="chip-x" {disabled} aria-label={`Remove hotspot ${tok}`} onclick={() => removeResidueTokenAt('hotspots', i)}>×</button>
                </span>
              {/each}
            </div>
          {/if}
        </div>
      {/if}

      {#if conditioningSourceSummary?.active && conditioningSourceSummary.hasTemplate}
        <div
          class="source-res-badge full"
          class:is-warn={conditioningSourceSummary.count === 0}
        >
          {#if conditioningSourceSummary.count > 0}
            ✓ Resolved <b>{conditioningSourceSummary.count}</b> source {conditioningSourceSummary.kindLabel} residue{conditioningSourceSummary.count === 1 ? '' : 's'}
            {#if conditioningSourceSummary.preview}
              <span class="source-res-list">({conditioningSourceSummary.preview})</span>
            {/if}
          {:else}
            ⚠ 0 template residues matched current {conditioningSourceSummary.kindLabel} selection
          {/if}
        </div>
      {/if}

      {#if !hasStructureInput && (g3Mode === 'motif' || g3Mode === 'binder')}
        <span class="field-hint err full">
          Wire a PDB Input into Genie 3's <code>structure</code> port for {g3Mode} design.
        </span>
      {/if}

      <label>
        Diffusion steps
        <NumericInput
          {disabled}
          ariaLabel="Diffusion sampling steps"
          spec={NUMERIC_SPECS.genie3.nSampleStep}
          value={Number(node.params.nSampleStep ?? 100)}
          onChange={(v) => setParam('nSampleStep', v)}
        />
      </label>
      <label>
        Backbones
        <NumericInput
          {disabled}
          ariaLabel="Backbones (numDesigns)"
          spec={NUM_DESIGNS_SPEC}
          value={g3NumDesigns}
          onChange={(v) => setParam('numDesigns', v)}
        />
      </label>
      {#if g3NumDesigns > 1}
        <div class="fanout-chip full" role="status">
          <span class="fanout-dot" aria-hidden="true"></span>
          <span>
            <b>{g3NumDesigns} backbones</b> · weights loaded once (B=1 sequential loop)
          </span>
        </div>
      {/if}

      <details class="adv-details full">
        <summary>
          Advanced parameters
          {#if g3HasCustomContig}
            <span class="adv-badge" title={`Custom contig: ${String(node.params.contig).trim()}`}>Custom contig</span>
          {/if}
        </summary>
        <div class="adv-fields">
          {#if g3Mode === 'motif' || g3Mode === 'binder'}
            <label class="full">
              {g3Mode === 'binder' ? 'Binder Contig (optional override)' : 'Contig (optional override)'}
              <input
                type="text"
                {disabled}
                placeholder={g3Mode === 'binder'
                  ? 'Auto from template (or e.g. A17-132,/0,80-80)'
                  : 'e.g. 25-25,A163-181,30-30'}
                value={String(node.params.contig ?? '')}
                oninput={(e) => setParam('contig', e.currentTarget.value)}
              />
            </label>
          {/if}
          <label>
            Direction scale
            <NumericInput
              {disabled}
              ariaLabel="Diffusion direction scale"
              spec={NUMERIC_SPECS.genie3.directionScale}
              value={Number(node.params.directionScale ?? 0.8)}
              onChange={(v) => setParam('directionScale', v)}
            />
          </label>
          <label>
            Noise scale
            <NumericInput
              {disabled}
              ariaLabel="Diffusion noise scale"
              spec={NUMERIC_SPECS.genie3.noiseScale}
              value={Number(node.params.noiseScale ?? 1.0)}
              onChange={(v) => setParam('noiseScale', v)}
            />
          </label>
          <label>
            Seed
            <NumericInput
              {disabled}
              ariaLabel="Seed"
              spec={NUMERIC_SPECS.genie3.seed}
              value={Number(node.params.seed ?? 42)}
              onChange={(v) => setParam('seed', v)}
            />
          </label>
          <label class="inline-check">
            <input
              type="checkbox"
              {disabled}
              checked={node.params.predictSequence !== false}
              onchange={(e) => setParam('predictSequence', e.currentTarget.checked)}
            />
            Predict native 1-step sequence
          </label>
        </div>
      </details>

    {:else if node.adapterId === 'web-protein-mpnn'}
      {@const fixedTokens = parseResidueTokens(node.params.fixedPositions)}
      {@const mpnnFamily = String(node.params.family ?? 'solublempnn')}
      {@const mpnnCheckpoints = MPNN_CHECKPOINTS_BY_FAMILY[mpnnFamily] ?? MPNN_CHECKPOINTS_BY_FAMILY.solublempnn}
      {@const mpnnModelName = String(node.params.modelName ?? 'v_48_020')}
      {@const mpnnAdvFields = getVisibleParamFields(manifest, node.params, 'advanced')}
      {@const mpnnHasCustomAdv =
        mpnnModelName !== 'v_48_020' ||
        Boolean(String(node.params.omitAminoAcids ?? '').trim()) ||
        Boolean(String(node.params.tiedPositions ?? '').trim()) ||
        Boolean(String(node.params.designChains ?? '').trim()) ||
        Number(node.params.backboneNoise ?? 0) > 0 ||
        Boolean(node.params.unlockInherited)}
      <label class="wide">
        Model Variant
        <select
          {disabled}
          value={mpnnFamily}
          onchange={(e) => {
            const nextFamily = e.currentTarget.value;
            if (isAutoMpnnLabel(node.label)) {
              const meta = MPNN_VARIANT_OPTIONS.find((v) => v.family === nextFamily);
              if (meta) onUpdateLabel?.(node.id, meta.shortLabel);
            }
            const validList = MPNN_CHECKPOINTS_BY_FAMILY[nextFamily] ?? MPNN_CHECKPOINTS_BY_FAMILY.solublempnn;
            const reconciledModel = validList.some((c) => c.value === mpnnModelName)
              ? mpnnModelName
              : 'v_48_020';
            onUpdate(node.id, {
              ...node.params,
              family: nextFamily,
              modelName: reconciledModel,
            });
          }}
        >
          {#each MPNN_VARIANT_OPTIONS as v (v.id)}
            <option value={v.family}>{v.label}</option>
          {/each}
        </select>
      </label>
      <label>
        Sequences
        <NumericInput
          {disabled}
          ariaLabel="Number of sequences"
          spec={NUMERIC_SPECS['web-protein-mpnn'].numSequences}
          value={Number(node.params.numSequences ?? 2)}
          onChange={(v) => setParam('numSequences', v)}
        />
      </label>
      <label>
        Temp
        <NumericInput
          {disabled}
          ariaLabel="Sampling temperature"
          spec={NUMERIC_SPECS['web-protein-mpnn'].temperature}
          value={Number(node.params.temperature ?? 0.1)}
          onChange={(v) => setParam('temperature', v)}
        />
      </label>
      {#if upstreamBackboneCount > 1}
        {@const mpnnPerBackbone = Math.max(1, Math.min(8, Math.round(Number(node.params.numSequences ?? 2)) || 2))}
        <div class="fanout-chip full" role="status">
          <span class="fanout-dot" aria-hidden="true"></span>
          <span>
            Runs per backbone (<b>{upstreamBackboneCount} × {mpnnPerBackbone} = {upstreamBackboneCount * mpnnPerBackbone} total sequences</b> · weights loaded once)
          </span>
        </div>
      {/if}

      <div class="chip-field full">
        <label>
          Fixed positions (click residues in 3D viewer or enter e.g. A10, A12-15)
          <input
            type="text"
            {disabled}
            placeholder="e.g. A5, A10-14"
            value={String(node.params.fixedPositions ?? '')}
            oninput={(e) => setParam('fixedPositions', e.currentTarget.value)}
          />
        </label>
        {#if fixedTokens.length > 0}
          <div class="res-chips">
            {#each fixedTokens as tok, i (`fix_${tok}_${i}`)}
              {@const badTok = !isValidResidueToken(tok)}
              <span class="res-chip" class:invalid-chip={badTok} title={badTok ? 'Expected format like A10 or A12-15' : undefined}>
                {tok}
                <button
                  type="button"
                  class="chip-x"
                  {disabled}
                  aria-label={`Remove fixed position ${tok}`}
                  onclick={() => removeResidueTokenAt('fixedPositions', i)}
                >×</button>
              </span>
            {/each}
          </div>
        {/if}
      </div>

      <details class="adv-details full">
        <summary>
          Advanced parameters
          {#if mpnnHasCustomAdv}
            <span class="adv-badge">Custom</span>
          {/if}
        </summary>
        <div class="adv-fields">
          <SchemaParamFields
            fields={mpnnAdvFields}
            params={node.params}
            {disabled}
            optionOverrides={{ modelName: mpnnCheckpoints }}
            onChange={(key, value) => setParam(key, value)}
          />
        </div>
      </details>

    {:else if node.adapterId === 'localfold-af2'}
      {@const lfModelId = String(node.params.model ?? 'monomer')}
      {@const lfMeta = selectedLocalFoldMeta ?? LOCALFOLD_MODEL_OPTIONS[0]}
      {@const lfMsaMode = String(node.params.msaMode ?? 'single') as MsaMode}
      {@const lfMaxRecycles = localFoldMaxRecycles(lfModelId, lfMeta.family)}
      {@const lfRecyclesSpec = {
        ...NUMERIC_SPECS['localfold-af2'].recycles,
        max: lfMaxRecycles,
      }}
      {@const lfDefaultTol = lfMeta.family === 'af2-multimer' ? 0.5 : 0}
      {@const lfTolSpec = {
        min: 0,
        max: 5,
        step: 0.1,
        integer: false,
        defaultValue: lfDefaultTol,
        unit: 'Å',
      }}
      {@const lfA3mSummary = lfMsaMode === 'a3m' ? parseA3mSequenceSummary(msaA3mDraft) : null}
      {@const lfHasHoloContext =
        Boolean(String(node.params.ligandCcdCodes ?? '').trim()) ||
        Boolean(String(node.params.dnaSequences ?? '').trim()) ||
        Boolean(String(node.params.rnaSequences ?? '').trim())}
      <label class="wide">
        Model
        <select
          {disabled}
          value={lfModelId}
          onchange={(e) => {
            const nextModel = e.currentTarget.value;
            const nextMeta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === nextModel);
            const autoInfo = isAutoLocalFoldLabel(node.label);
            if (autoInfo.auto && nextMeta) {
              onUpdateLabel?.(node.id, `${nextMeta.shortLabel}${autoInfo.suffix}`);
            }
            const nextMaxRec = nextMeta ? localFoldMaxRecycles(nextModel, nextMeta.family) : 10;
            const curRec = Number(node.params.recycles ?? 1);
            if (Number.isFinite(curRec) && curRec > nextMaxRec) {
              onUpdate(node.id, { ...node.params, model: nextModel, recycles: nextMaxRec });
              return;
            }
            setParam('model', nextModel);
          }}
        >
          <optgroup label="AlphaFold2-Class (Monomer)">
            {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2') as opt (opt.id)}
              <option value={opt.id}>{opt.label}</option>
            {/each}
          </optgroup>
          <optgroup label="AlphaFold2-Multimer v3 (Complexes)">
            {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2-multimer') as opt (opt.id)}
              <option value={opt.id}>{opt.label}</option>
            {/each}
          </optgroup>
          <optgroup label="AF3-Class (All-Atom)">
            {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3') as opt (opt.id)}
              <option value={opt.id}>{opt.label}</option>
            {/each}
          </optgroup>
        </select>
        <span class="field-hint">{formatLocalFoldMsaBadge(lfMsaMode)} · Weights: ~{lfMeta.approxMB} MB · License: {lfMeta.license}</span>
      </label>

      <label class="full">
        MSA Mode
        <select
          {disabled}
          value={lfMsaMode}
          onchange={(e) => {
            flushPendingTextCommit();
            setParam('msaMode', e.currentTarget.value);
          }}
        >
          <option value="single">Single-sequence (100% local)</option>
          <option value="mmseqs2">MMseqs2 MSA (ColabFold API)</option>
          <option value="target-only">Target-only MSA (binders — skips de novo chain)</option>
          <option value="a3m">Custom .a3m alignment (local)</option>
        </select>
        {#if lfMsaMode === 'mmseqs2' || lfMsaMode === 'target-only'}
          <span class="field-hint">
            Queries api.colabfold.com for evolutionary alignments; cached across candidates
          </span>
        {/if}
      </label>

      {#if lfMsaMode === 'a3m'}
        <div class="a3m-box full">
          <div class="a3m-head">
            <label class="mu-upload-btn a3m-upload-btn" class:is-disabled={disabled}>
              <span>↑ Upload .a3m file</span>
              <input
                type="file"
                accept=".a3m,.fas,.fasta,.txt"
                hidden
                {disabled}
                onchange={handleA3mFileUpload}
              />
            </label>
            {#if lfA3mSummary && lfA3mSummary.seqCount > 0}
              <span class="field-hint ok">
                ✓ {lfA3mSummary.seqCount} sequence{lfA3mSummary.seqCount === 1 ? '' : 's'} parsed{lfA3mSummary.queryLength > 0 ? ` · ${lfA3mSummary.queryLength} aa query` : ''}
              </span>
            {:else}
              <span class="field-hint warn">Paste or upload a .a3m alignment</span>
            {/if}
          </div>
          <textarea
            rows="4"
            class:input-error={!lfA3mSummary || lfA3mSummary.seqCount === 0}
            {disabled}
            placeholder=">query&#10;MKTAYIAKQRQISFVK...&#10;>hit1&#10;MKTAYIAKQRQISFVK..."
            value={msaA3mDraft}
            oninput={(e) => scheduleTextParam('msaA3mText', e.currentTarget.value)}
            onblur={flushPendingTextCommit}
          ></textarea>
        </div>
      {/if}

      <label>
        Recycles
        <NumericInput
          {disabled}
          ariaLabel="Recycles"
          spec={lfRecyclesSpec}
          value={Number(node.params.recycles ?? 1)}
          onChange={(v) => setParam('recycles', v)}
        />
        <span class="field-hint">{localFoldDefaultRecyclesHint(lfModelId, lfMeta.family)}</span>
      </label>
      {#if lfMeta.family === 'af3'}
        <label>
          Diffusion steps
          <NumericInput
            {disabled}
            ariaLabel="Diffusion steps"
            spec={NUMERIC_SPECS['localfold-af2'].diffusionSteps}
            value={Number(node.params.diffusionSteps ?? 25)}
            onChange={(v) => setParam('diffusionSteps', v)}
          />
        </label>
      {/if}
      <label>
        Max seqs to fold
        <NumericInput
          {disabled}
          ariaLabel="Max sequences to fold"
          spec={NUMERIC_SPECS['localfold-af2'].maxSequences}
          value={Number(node.params.maxSequences ?? 2)}
          onChange={(v) => setParam('maxSequences', v)}
        />
      </label>
      {#if lfMultiDesignInfo}
        {@const lfPerBackboneCap = Math.max(1, Math.min(8, Math.round(Number(node.params.maxSequences ?? 2)) || 2))}
        {@const lfCampaignCap = Math.max(1, Math.min(64, Math.round(Number(node.params.maxTotalFolds ?? 16)) || 16))}
        {@const lfPlanned = Math.min(
          lfCampaignCap,
          lfMultiDesignInfo.backboneCount * lfPerBackboneCap,
          lfMultiDesignInfo.sequenceCount ?? lfCampaignCap,
        )}
        <div class="fanout-chip full" role="status">
          <span class="fanout-dot" aria-hidden="true"></span>
          <span>
            <b>{lfMultiDesignInfo.backboneCount} backbones</b>
            {#if lfMultiDesignInfo.sequenceCount !== null}
              → {lfMultiDesignInfo.sequenceCount} sequences
            {/if}
            → up to <b>{lfPlanned} refold{lfPlanned === 1 ? '' : 's'}</b>
            (≤{lfPerBackboneCap}/backbone, cap {lfCampaignCap})
          </span>
        </div>
      {/if}
      {#if lfMeta.nonCommercial}
        <div class="full" style="font-size: 10.5px; color: var(--warn-text); background: var(--warn-bg); border: 1px solid var(--warn-border); padding: 6px 8px; border-radius: 5px;">
          AlphaFold 3 weights are subject to DeepMind's Non-Commercial Academic Terms. For permissive open weights, use <b>RoseTTAFold3</b> (BSD-3), <b>Boltz-2</b> (MIT), or <b>OpenBind-0</b> (Apache-2.0).
        </div>
      {/if}

      {#if lfMeta.family === 'af3'}
        <details class="adv-details full" open={lfHasHoloContext}>
          <summary>
            Holo co-folding (ligands / DNA / RNA)
            {#if lfHasHoloContext}
              <span class="adv-badge">Active</span>
            {/if}
          </summary>
          <div class="adv-fields">
            <label class="full">
              Ligand CCD codes
              <input
                type="text"
                {disabled}
                placeholder="e.g. SAM, ZN, ATP"
                value={String(node.params.ligandCcdCodes ?? '')}
                oninput={(e) => setParam('ligandCcdCodes', e.currentTarget.value)}
              />
              <span class="field-hint">Comma-separated PDB Chemical Component Dictionary (CCD) IDs.</span>
            </label>
            <label class="full">
              DNA sequences
              <input
                type="text"
                {disabled}
                placeholder="e.g. ACGTACGT (separate multiple chains with : or ,)"
                value={String(node.params.dnaSequences ?? '')}
                oninput={(e) => setParam('dnaSequences', e.currentTarget.value)}
              />
            </label>
            <label class="full">
              RNA sequences
              <input
                type="text"
                {disabled}
                placeholder="e.g. AUGCAUGC (separate multiple chains with : or ,)"
                value={String(node.params.rnaSequences ?? '')}
                oninput={(e) => setParam('rnaSequences', e.currentTarget.value)}
              />
            </label>
          </div>
        </details>
      {/if}

      <details class="adv-details full">
        <summary>
          Advanced parameters
          {#if Number(node.params.maxTotalFolds ?? 16) !== 16}
            <span class="adv-badge">Cap {Number(node.params.maxTotalFolds ?? 16)}</span>
          {/if}
        </summary>
        <div class="adv-fields">
          <label>
            Max total folds
            <NumericInput
              {disabled}
              ariaLabel="Max total folds across designs"
              spec={MAX_TOTAL_FOLDS_SPEC}
              value={Number(node.params.maxTotalFolds ?? 16)}
              onChange={(v) => setParam('maxTotalFolds', v)}
            />
            <span class="field-hint">
              Campaign cap across multi-backbone batches (round-robin up to {Number(node.params.maxSequences ?? 2)}/backbone).
            </span>
          </label>
          <label>
            Seed
            <NumericInput
              {disabled}
              ariaLabel="Seed"
              spec={NUMERIC_SPECS['localfold-af2'].seed}
              value={Number(node.params.seed ?? 0)}
              onChange={(v) => setParam('seed', v)}
            />
          </label>
          {#if lfMeta.family === 'af2' || lfMeta.family === 'af2-multimer'}
            <label>
              Early-stop tolerance (Å)
              <NumericInput
                {disabled}
                ariaLabel="Early-stop tolerance (Å)"
                spec={lfTolSpec}
                value={Number(node.params.tolerance ?? lfDefaultTol)}
                onChange={(v) => setParam('tolerance', v)}
              />
              <span class="field-hint">
                0 = run all recycles on AF2 monomer (0.5 Å default on AF2-Multimer).
              </span>
            </label>
          {/if}
        </div>
      </details>

    {:else if node.adapterId === 'designability-scorer'}
      {@const curMotifThresh = Number(node.params.motifRmsdThreshold ?? 1.0)}
      {@const curIpaeThresh = Number(node.params.ipaeThreshold ?? 10.0)}
      {@const curIpsaeThresh = Number(node.params.ipsaeThreshold ?? 0)}
      {@const hasCustomAdvGate = curMotifThresh !== 1.0 || curIpaeThresh !== 10.0 || curIpsaeThresh > 0}
      <label>
        scRMSD &lt; Å
        <NumericInput
          {disabled}
          ariaLabel="scRMSD threshold"
          spec={NUMERIC_SPECS['designability-scorer'].scRmsdThreshold}
          value={Number(node.params.scRmsdThreshold ?? 2.0)}
          onChange={(v) => setParam('scRmsdThreshold', v)}
        />
      </label>
      <label>
        pLDDT &ge;
        <NumericInput
          {disabled}
          ariaLabel="pLDDT threshold"
          spec={NUMERIC_SPECS['designability-scorer'].plddtThreshold}
          value={Number(node.params.plddtThreshold ?? 70)}
          onChange={(v) => setParam('plddtThreshold', v)}
        />
      </label>
      <details class="adv-details full">
        <summary>
          Advanced thresholds
          {#if hasCustomAdvGate}
            <span class="adv-badge">Custom</span>
          {/if}
        </summary>
        <div class="adv-fields">
          <label>
            Motif RMSD threshold (Å)
            <NumericInput
              {disabled}
              ariaLabel="Motif RMSD threshold (Å)"
              spec={SCORER_MOTIF_RMSD_SPEC}
              value={curMotifThresh}
              onChange={(v) => setParam('motifRmsdThreshold', v)}
            />
          </label>
          <label>
            iPAE threshold (Å)
            <NumericInput
              {disabled}
              ariaLabel="iPAE threshold (Å)"
              spec={SCORER_IPAE_SPEC}
              value={curIpaeThresh}
              onChange={(v) => setParam('ipaeThreshold', v)}
            />
          </label>
          <label>
            ipSAE threshold (0 = off)
            <NumericInput
              {disabled}
              ariaLabel="ipSAE threshold"
              spec={SCORER_IPSAE_SPEC}
              value={curIpsaeThresh}
              onChange={(v) => setParam('ipsaeThreshold', v)}
            />
            <span class="field-hint">Optional Dunbrack interface gate (e.g. 0.6; 0 disables).</span>
          </label>
        </div>
      </details>

    {:else if manifest && (manifest.modes?.length || manifest.paramSchema?.length)}
      {@const schemaModeKey = manifest.modeParamKey}
      {@const schemaActiveMode = getActiveModeSpec(manifest, node.params)}
      {@const schemaPrimaryFields = getVisibleParamFields(manifest, node.params, 'primary').filter(
        (f) => !(manifest.modes?.length && schemaModeKey && f.key === schemaModeKey),
      )}
      {@const schemaAdvancedFields = getVisibleParamFields(manifest, node.params, 'advanced').filter(
        (f) => !(manifest.modes?.length && schemaModeKey && f.key === schemaModeKey),
      )}
      {#if manifest.modes?.length && schemaModeKey}
        <label class="full">
          Mode
          <select
            {disabled}
            value={schemaActiveMode?.id ?? manifest.modes[0].id}
            onchange={(e) => setParam(schemaModeKey, e.currentTarget.value)}
          >
            {#each manifest.modes as modeSpec (modeSpec.id)}
              <option value={modeSpec.id}>{modeSpec.label}</option>
            {/each}
          </select>
          {#if schemaActiveMode?.description}
            <span class="field-hint">{schemaActiveMode.description}</span>
          {/if}
        </label>
      {/if}
      {#if schemaPrimaryFields.length > 0}
        <SchemaParamFields
          fields={schemaPrimaryFields}
          params={node.params}
          {disabled}
          onChange={(key, value) => setParam(key, value)}
        />
      {/if}
      {#if schemaAdvancedFields.length > 0}
        <details class="adv-details full">
          <summary>Advanced parameters</summary>
          <div class="adv-fields">
            <SchemaParamFields
              fields={schemaAdvancedFields}
              params={node.params}
              {disabled}
              onChange={(key, value) => setParam(key, value)}
            />
          </div>
        </details>
      {/if}
    {/if}
  </div>

  {#if manifest}
    <footer class="info-footer">
      <p class="desc">{manifest.description}</p>
      <div class="foot-meta">
        <span>{manifest.runtime === 'cpu-ts' ? 'Runs locally in browser (CPU)' : 'Runs locally in browser (WebGPU)'}</span>
        {#if selectedLocalFoldMeta}
          <span>· ~{selectedLocalFoldMeta.approxMB} MB ({selectedLocalFoldMeta.shortLabel})</span>
          <span>· {selectedLocalFoldMeta.license}</span>
        {:else if node.adapterId === 'rfd4-proteina' && rfd4NeedsEncoder}
          <span>· ~{(manifest.approxWeightsMB || 330) + 138} MB (trunk + encoder)</span>
          {#if manifest.licenseNote}
            <span>· {manifest.licenseNote}</span>
          {/if}
        {:else}
          {#if manifest.approxWeightsMB > 0}
            <span>· ~{manifest.approxWeightsMB} MB</span>
          {/if}
          {#if manifest.licenseNote}
            <span>· {manifest.licenseNote}</span>
          {/if}
        {/if}
      </div>
    </footer>
  {/if}
</div>

<style>
  .inspector {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .inspector.embedded {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 12px 14px 14px;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .title-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .stage-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  h3 {
    margin: 0;
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .title-input {
    margin: 0;
    padding: 2px 6px;
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    min-width: 90px;
    max-width: 240px;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .title-input:hover:not(:disabled) {
    border-color: var(--border-strong);
    background: var(--bg-surface-2);
  }
  .title-input:focus {
    outline: none;
    border-color: var(--accent);
    background: var(--bg-input);
  }
  .sub-kind {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }
  .head-actions {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .studio-open-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    flex-shrink: 0;
    white-space: nowrap;
  }
  .studio-open-btn:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .run-here-btn {
    background: var(--bg-surface-2);
    color: var(--accent-bright);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    flex-shrink: 0;
  }
  .run-here-btn:hover:not(:disabled) {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
  .run-here-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .fields {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
    align-items: flex-start;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }
  label.wide {
    min-width: 180px;
    flex: 1;
  }
  .full {
    width: 100%;
  }
  label.inline-check {
    flex-direction: row;
    align-items: center;
    gap: 6px;
    padding-bottom: 6px;
    cursor: pointer;
    flex-wrap: wrap;
  }
  label.inline-check.is-muted {
    opacity: 0.72;
    cursor: not-allowed;
  }
  .a3m-box {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .a3m-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .a3m-upload-btn {
    padding: 3px 9px;
    font-size: 10.5px;
  }
  .chip-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .res-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .res-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--accent-soft);
    background: var(--bg-pill);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    padding: 2px 6px 2px 8px;
  }
  .chip-x {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 12px;
    line-height: 1;
    padding: 0 2px;
    border-radius: 50%;
  }
  .chip-x:hover:not(:disabled) {
    color: var(--err-strong);
  }
  .chip-x:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .adv-details {
    border-top: 1px solid var(--border-subtle);
    padding-top: 8px;
    margin-top: 2px;
  }
  .adv-details summary {
    cursor: pointer;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    user-select: none;
    padding: 2px 0;
  }
  .adv-details summary:hover {
    color: var(--text-primary);
  }
  .adv-fields {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
    align-items: flex-start;
    margin-top: 10px;
  }
  input[type='text'],
  select,
  textarea {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 6px 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
  }
  .input-error {
    border-color: var(--err-strong) !important;
  }
  .input-warn {
    border-color: var(--warn-border) !important;
  }
  .field-hint {
    font-size: 10.5px;
    line-height: 1.35;
    color: var(--text-dim);
  }
  .field-hint.err {
    color: var(--err-strong);
  }
  .field-hint.warn {
    color: var(--warn-text);
  }
  .field-hint.ok {
    color: var(--ok-text, #10b981);
  }
  .res-chip.invalid-chip {
    border-color: var(--err-strong);
    color: var(--err-strong);
    background: color-mix(in srgb, var(--err-strong) 10%, transparent);
  }
  .pdb-fetch-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .field-lbl {
    font-size: 11px;
    color: var(--text-muted);
  }
  .pdb-fetch-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .pdb-fetch-row input {
    flex: 1;
    min-width: 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .fetch-btn {
    flex-shrink: 0;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--accent);
    border-radius: 5px;
    padding: 6px 11px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }
  .fetch-btn:hover:not(:disabled) {
    background: var(--accent);
    color: #fff;
  }
  .fetch-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .quick-chips {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
  }
  .quick-lbl {
    font-size: 10px;
    color: var(--text-dim);
    margin-right: 2px;
  }
  .quick-chip {
    background: var(--bg-surface-2);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px 7px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    cursor: pointer;
  }
  .quick-chip:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--text-primary);
  }
  .quick-chip:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .bench-pick {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-top: 2px;
  }
  .bench-pick span {
    font-size: 10px;
    color: var(--text-dim);
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    resize: vertical;
  }
  code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
  }
  .info-footer {
    margin-top: 4px;
    padding-top: 10px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .desc {
    margin: 0;
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .foot-meta {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
    opacity: 0.85;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .sync-sib-btn {
    width: 100%;
    padding: 6px 10px;
    border-radius: 6px;
    border: 1px solid var(--border-strong);
    background: var(--bg-pill);
    color: var(--accent-soft);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
  }
  .sync-sib-btn:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .sync-sib-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .file-row {
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: break-word;
  }
  .file-row input[type='file'] {
    max-width: 100%;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .filename-wrap {
    overflow-wrap: anywhere;
    word-break: break-word;
  }
  .adv-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px;
    font-weight: 600;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    vertical-align: middle;
  }
  .source-res-badge {
    font-size: 10.5px;
    line-height: 1.4;
    padding: 5px 8px;
    border-radius: 5px;
    background: var(--ok-bg);
    color: var(--ok-text);
    border: 1px solid var(--ok-border);
    overflow-wrap: anywhere;
  }
  .source-res-badge.is-warn {
    background: var(--warn-bg);
    color: var(--warn-text);
    border-color: var(--warn-border);
  }
  .fanout-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    line-height: 1.4;
    padding: 5px 8px;
    border-radius: 5px;
    background: var(--bg-pill);
    color: var(--accent-soft);
    border: 1px solid var(--border-strong);
    box-sizing: border-box;
  }
  .fanout-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }
  .source-res-list {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    margin-left: 4px;
    opacity: 0.9;
  }
  .missing-upload-card {
    display: flex;
    flex-direction: column;
    gap: 7px;
    padding: 9px 10px;
    border-radius: 6px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
  }
  .mu-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    flex-wrap: wrap;
  }
  .mu-title {
    font-size: 11.5px;
    font-weight: 700;
  }
  .mu-sha {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.14);
  }
  .mu-body {
    margin: 0;
    font-size: 11px;
    line-height: 1.45;
  }
  .mu-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .mu-upload-btn {
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 5px;
    background: var(--accent);
    color: #fff;
    border: 1px solid var(--accent);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }
  .mu-upload-btn.is-disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .mu-clear-btn {
    padding: 4px 9px;
    border-radius: 5px;
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    font-family: inherit;
  }
  .mu-clear-btn:hover:not(:disabled) {
    color: var(--text-primary);
    border-color: var(--accent);
  }
  .mu-clear-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .crop-lig-warn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
    padding: 6px 8px;
    border-radius: 6px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    color: var(--warn-text);
    font-size: 11px;
    line-height: 1.35;
  }
</style>
