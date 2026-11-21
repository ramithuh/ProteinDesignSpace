<script lang="ts">
  import {
    ADAPTER_REGISTRY,
    LOCALFOLD_MODEL_OPTIONS,
    MPNN_VARIANT_OPTIONS,
    RFD4_BENCHMARK_OPTIONS,
  } from '$lib/engine/registry';
  import { STAGE_COLORS } from '$lib/engine/graph-utils';
  import {
    NUMERIC_SPECS,
    isValidResidueToken,
    validateJsonOverride,
    validatePdbId,
    validateSequenceText,
    validateStructureText,
  } from '$lib/engine/param-validation';
  import {
    QUICK_PDB_EXAMPLES,
    fetchStructureByPdbId,
    normalizePdbId,
  } from '$lib/protein-ops/rcsb';
  import { BENCHMARK_TARGETS, getBenchmarkById } from '$lib/engine/benchmarks';
  import NumericInput from './NumericInput.svelte';
  import type { CanonicalArtifact, NodeProgressEvent, StructureArtifact, WorkflowNode } from '$lib/engine/types';

  interface Props {
    node: WorkflowNode;
    disabled?: boolean;
    embedded?: boolean;
    nodeOutput?: Record<string, CanonicalArtifact>;
    nodeState?: NodeProgressEvent;
    onUpdate: (nodeId: string, params: Record<string, unknown>) => void;
    onUpdateLabel?: (nodeId: string, label: string) => void;
    onRunUpTo?: (nodeId: string) => void;
  }

  let {
    node,
    disabled = false,
    embedded = false,
    nodeOutput = undefined,
    nodeState = undefined,
    onUpdate,
    onUpdateLabel = undefined,
    onRunUpTo = undefined,
  }: Props = $props();

  let liveStructure = $derived<StructureArtifact | null>(
    nodeOutput?.structure?.kind === 'StructureArtifact' ? nodeOutput.structure : null,
  );

  let manifest = $derived(ADAPTER_REGISTRY[node.adapterId]);
  let stageColor = $derived(STAGE_COLORS[manifest?.stage ?? 'source'] ?? '#64748b');

  let selectedLocalFoldMeta = $derived.by(() => {
    if (node.adapterId !== 'localfold-af2') return null;
    const modelId = String(node.params?.model ?? 'monomer');
    return LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
  });

  let activeModelBadge = $derived.by((): string => {
    const p = node.params ?? {};
    switch (node.adapterId) {
      case 'localfold-af2':
        return `${selectedLocalFoldMeta?.label ?? LOCALFOLD_MODEL_OPTIONS[0].label} · Single-sequence (no MSA)`;
      case 'rfd4-proteina': {
        const mode = String(p.mode ?? 'unconditional');
        const needsEnc =
          mode === 'partial' ||
          mode === 'inverse-folding' ||
          (mode === 'preset' &&
            Boolean(RFD4_BENCHMARK_OPTIONS.find((o) => o.id === String(p.presetId ?? ''))?.needsEncoder));
        return needsEnc
          ? 'RFD4-Proteina · Conditioned + structure encoder'
          : 'RFD4-Proteina · All-atom co-design';
      }
      case 'genie3':
        return 'Genie 3 · Backbone diffusion';
      case 'web-protein-mpnn': {
        const famMeta =
          MPNN_VARIANT_OPTIONS.find((v) => v.family === p.family) ?? MPNN_VARIANT_OPTIONS[0];
        const ckpt = String(p.modelName ?? 'v_48_020');
        return `${famMeta.label} · ${ckpt}`;
      }
      case 'rfd4-conditioner':
        return 'Model-agnostic hotspots, motifs, H-bonds & fold topology';
      case 'designability-scorer':
        return 'Kabsch alignment · scRMSD / scTM';
      case 'pdb-input':
        return 'PDB / mmCIF structure input';
      case 'sequence-input':
        return 'FASTA sequence input';
      default:
        return manifest?.shortName ?? node.adapterId;
    }
  });

  function setParam(key: string, value: unknown) {
    onUpdate(node.id, { ...node.params, [key]: value });
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

  let pdbFetchBusy = $state(false);
  let pdbFetchError = $state<string | null>(null);
  let pdbFetchSummary = $state<string | null>(null);

  $effect(() => {
    void node.id;
    pdbFetchError = null;
    pdbFetchSummary = null;
  });

  async function handleFetchPdbId(overrideId?: string, benchmarkIdOverride?: string) {
    const rawId = overrideId ?? String(node.params.pdbId ?? '');
    const normId = normalizePdbId(rawId);
    const check = validatePdbId(normId);
    if (check.status === 'error') {
      pdbFetchError = check.message ?? 'Invalid PDB ID';
      return;
    }
    const targetNodeId = node.id;
    const baseParams = { ...node.params };
    const nextBenchId =
      benchmarkIdOverride !== undefined
        ? benchmarkIdOverride
        : overrideId
          ? ''
          : String(baseParams.benchmarkId ?? '');
    pdbFetchBusy = true;
    pdbFetchError = null;
    pdbFetchSummary = null;
    try {
      const res = await fetchStructureByPdbId(normId);
      onUpdate(targetNodeId, {
        ...baseParams,
        mode: 'pdb-id',
        pdbId: res.pdbId,
        fetchedPdbId: res.pdbId,
        filename: res.filename,
        pdbText: res.text,
        benchmarkId: nextBenchId,
      });
      const extras = [
        res.ligandNames.length > 0 ? `ligands: ${res.ligandNames.join(', ')}` : '',
        res.nucleicAcidChains.length > 0 ? `NA chains: ${res.nucleicAcidChains.join(', ')}` : '',
      ].filter(Boolean);
      pdbFetchSummary =
        `Loaded ${res.filename} · ${res.nResidues} aa · ${res.chainCount} chain${res.chainCount === 1 ? '' : 's'}` +
        (extras.length > 0 ? ` · ${extras.join(' · ')}` : '');
    } catch (err) {
      pdbFetchError = err instanceof Error ? err.message : String(err);
    } finally {
      pdbFetchBusy = false;
    }
  }

  async function handleFileUpload(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const targetNodeId = node.id;
    const baseParams = { ...node.params };
    try {
      const text = await file.text();
      onUpdate(targetNodeId, {
        ...baseParams,
        mode: 'custom',
        filename: file.name,
        pdbText: text,
      });
    } finally {
      input.value = '';
    }
  }
</script>

<div class="inspector" class:embedded>
  <div class="head">
    <div class="title-wrap">
      <span class="stage-dot" style="background: {stageColor}"></span>
      <h3>{node.label || manifest?.displayName || node.adapterId}</h3>
      <span class="sub-kind" title="Underlying model / engine">{activeModelBadge}</span>
    </div>
    {#if onRunUpTo}
      <button
        type="button"
        class="run-here-btn"
        {disabled}
        onclick={() => onRunUpTo(node.id)}
        title="Execute only upstream blocks up to this block (memoized for full runs)"
      >
        ▶ Run to here
      </button>
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
            pdbFetchError = null;
            pdbFetchSummary = null;
            setParam('mode', e.currentTarget.value);
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
        {@const isBusy = pdbFetchBusy || isAutoFetching}
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
                pdbFetchError = null;
                pdbFetchSummary = null;
                onUpdate(node.id, {
                  ...node.params,
                  pdbId: e.currentTarget.value.toUpperCase(),
                  benchmarkId: '',
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
                pdbFetchError = null;
                pdbFetchSummary = null;
                onUpdateLabel?.(node.id, `PDB: ${bench.name} (${bench.pdbId})`);
                onUpdate(node.id, {
                  ...node.params,
                  mode: 'pdb-id',
                  pdbId: bench.pdbId,
                  fetchedPdbId: '',
                  filename: `${bench.pdbId}.cif`,
                  pdbText: '',
                  benchmarkId: bench.id,
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
          {#if pdbFetchError}
            <span class="field-hint err">{pdbFetchError}</span>
          {:else if nodeState?.status === 'error' && nodeState.error}
            <span class="field-hint err">{nodeState.error}</span>
          {:else if isBusy}
            <span class="field-hint">Fetching {normalizePdbId(pdbIdVal) || 'structure'} from RCSB PDB for 3D preview…</span>
          {:else if pdbFetchSummary}
            <span class="field-hint ok">✓ {pdbFetchSummary}</span>
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
        {@const pdbCheck = validateStructureText(String(node.params.pdbText ?? ''))}
        <label class="file-row full">
          Upload .pdb / .cif
          <input type="file" accept=".pdb,.cif,.mmcif,.ent,.txt" {disabled} onchange={handleFileUpload} />
        </label>
        <label class="full">
          PDB / mmCIF text ({String(node.params.filename || 'custom.pdb')})
          <textarea
            rows="4"
            class:input-error={pdbCheck.status === 'error'}
            {disabled}
            placeholder="Paste ATOM lines or mmCIF _atom_site block…"
            value={String(node.params.pdbText ?? '')}
            oninput={(e) => setParam('pdbText', e.currentTarget.value)}
          ></textarea>
          {#if pdbCheck.message}
            <span class="field-hint" class:err={pdbCheck.status === 'error'} class:ok={pdbCheck.status === 'ok'}>
              {pdbCheck.message}
            </span>
          {/if}
        </label>
      {/if}

    {:else if node.adapterId === 'sequence-input'}
      {@const seqCheck = validateSequenceText(String(node.params.sequence ?? ''))}
      <label class="full">
        Amino acid sequence (plain or multi-FASTA)
        <textarea
          rows="5"
          class:input-error={seqCheck.status === 'error'}
          class:input-warn={seqCheck.status === 'warn'}
          {disabled}
          placeholder=">seq1&#10;MKTAYIAKQRQISFVK..."
          value={String(node.params.sequence ?? '')}
          oninput={(e) => setParam('sequence', e.currentTarget.value)}
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

    {:else if node.adapterId === 'rfd4-conditioner'}
      {@const hotspotTokens = parseResidueTokens(node.params.hotspots)}
      {@const motifTokens = parseResidueTokens(node.params.motifSpan)}
      {@const taskPresetId = String(node.params.taskPreset ?? 'custom')}
      {@const selectedPresetMeta = taskPresetId !== 'custom' ? RFD4_BENCHMARK_OPTIONS.find((o) => o.id === taskPresetId) : null}
      <label class="full">
        Design Task / Preset
        <select
          {disabled}
          value={taskPresetId}
          onchange={(e) => setParam('taskPreset', e.currentTarget.value)}
        >
          <option value="custom">Custom Conditioning Builder</option>
          {#each RFD4_BENCHMARK_OPTIONS as opt (opt.id)}
            <option value={opt.id}>[{opt.category}] {opt.label}</option>
          {/each}
        </select>
        {#if selectedPresetMeta}
          <span class="field-hint">
            Preset category: {selectedPresetMeta.category}{selectedPresetMeta.needsEncoder ? ' · uses structure encoder (~138 MB)' : ''}
          </span>
        {/if}
      </label>

      {#if taskPresetId === 'custom'}
        {@const contigVal = String(node.params.contig ?? '70-70')}
        {@const rawJsonCheck = validateJsonOverride(String(node.params.rawJsonOverride ?? ''))}
        {@const activeBench = getBenchmarkById(String(node.params.benchmarkId ?? ''))}
        <label class="full">
          Load Curated Benchmark Spec (optional)
          <select
            {disabled}
            value={String(node.params.benchmarkId ?? '')}
            onchange={(e) => {
              const id = e.currentTarget.value;
              const b = getBenchmarkById(id);
              if (!b) {
                setParam('benchmarkId', '');
                return;
              }
              onUpdate(node.id, {
                ...node.params,
                benchmarkId: b.id,
                pdbId: b.pdbId,
                taskPreset: b.conditionParams.taskPreset,
                contig: b.conditionParams.contig,
                lockChain: b.conditionParams.lockChain ?? '',
                hotspots: b.conditionParams.hotspots ?? '',
                motifSpan: b.conditionParams.motifSpan ?? '',
                unindexedMotif: Boolean(b.conditionParams.unindexedMotif),
                hbondMode: b.conditionParams.hbondMode ?? 'none',
                hbondSelection: b.conditionParams.hbondSelection ?? '',
                cathCode: '',
                rawJsonOverride: '',
              });
            }}
          >
            <option value="">— Custom parameters below —</option>
            <optgroup label="PPI Binders (RFD4 & Genie 3)">
              {#each BENCHMARK_TARGETS.filter((b) => b.category === 'binder') as b (b.id)}
                <option value={b.id}>{b.shortLabel} · ~{b.approxTokens} tok</option>
              {/each}
            </optgroup>
            <optgroup label="Motif Scaffolding (RFD4 & Genie 3)">
              {#each BENCHMARK_TARGETS.filter((b) => b.category === 'motif') as b (b.id)}
                <option value={b.id}>{b.shortLabel} · ~{b.approxTokens} tok</option>
              {/each}
            </optgroup>
            <optgroup label="AME Enzymes, Ligands & DNA (RFD4 All-Atom)">
              {#each BENCHMARK_TARGETS.filter((b) => b.category === 'enzyme-ligand') as b (b.id)}
                <option value={b.id}>{b.shortLabel} · ~{b.approxTokens} tok</option>
              {/each}
            </optgroup>
          </select>
          {#if activeBench}
            <span class="field-hint">
              {activeBench.biology} · Wire PDB <code>{activeBench.pdbId}</code> into <code>template</code> · {activeBench.supportsGenie3 ? 'Compatible with RFD4 & Genie 3' : 'RFD4 all-atom only'}
            </span>
          {/if}
        </label>
        <label class="full">
          Contig (e.g. <code>70-70</code>, <code>A18-132,/0,70-70</code>)
          <input
            type="text"
            class:input-error={!contigVal.trim() && !String(node.params.rawJsonOverride ?? '').trim()}
            {disabled}
            value={contigVal}
            oninput={(e) => setParam('contig', e.currentTarget.value)}
          />
          {#if !contigVal.trim() && !String(node.params.rawJsonOverride ?? '').trim()}
            <span class="field-hint err">Contig is required unless a JSON constraint override is provided.</span>
          {/if}
        </label>

        <div class="chip-field full">
          <label>
            Binding Hotspots (click residues in 3D viewer or enter e.g. A54, A56)
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
              {#each hotspotTokens as tok, i (`hot_${tok}_${i}`)}
                {@const badTok = !isValidResidueToken(tok)}
                <span class="res-chip" class:invalid-chip={badTok} title={badTok ? 'Expected format like A54 or A16-35' : undefined}>
                  {tok}
                  <button
                    type="button"
                    class="chip-x"
                    {disabled}
                    aria-label={`Remove hotspot ${tok}`}
                    onclick={() => removeResidueTokenAt('hotspots', i)}
                  >×</button>
                </span>
              {/each}
            </div>
          {/if}
        </div>

        <div class="chip-field full">
          <label>
            Fixed Structural Motif (e.g. A16-35, A52-71)
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
              {#each motifTokens as tok, i (`motif_${tok}_${i}`)}
                {@const badTok = !isValidResidueToken(tok)}
                <span class="res-chip" class:invalid-chip={badTok} title={badTok ? 'Expected format like A16-35' : undefined}>
                  {tok}
                  <button
                    type="button"
                    class="chip-x"
                    {disabled}
                    aria-label={`Remove motif ${tok}`}
                    onclick={() => removeResidueTokenAt('motifSpan', i)}
                  >×</button>
                </span>
              {/each}
            </div>
          {/if}
        </div>

        <details class="adv-details full">
          <summary>Advanced parameters</summary>
          <div class="adv-fields">
            <label>
              Lock Chain
              <input
                type="text"
                {disabled}
                placeholder="e.g. A"
                style="width: 72px"
                value={String(node.params.lockChain ?? '')}
                oninput={(e) => setParam('lockChain', e.currentTarget.value)}
              />
            </label>
            <label>
              CATH Fold
              <input
                type="text"
                {disabled}
                placeholder="e.g. 3.40.50"
                style="width: 96px"
                value={String(node.params.cathCode ?? '')}
                oninput={(e) => setParam('cathCode', e.currentTarget.value)}
              />
            </label>
            <label>
              3D Click Brush
              <select
                {disabled}
                value={String(node.params.paintBrush ?? 'hotspots')}
                onchange={(e) => setParam('paintBrush', e.currentTarget.value)}
              >
                <option value="hotspots">Paint Binding Hotspots</option>
                <option value="motifSpan">Paint Fixed Motif</option>
              </select>
            </label>
            <label class="inline-check">
              <input
                type="checkbox"
                {disabled}
                checked={Boolean(node.params.unindexedMotif)}
                onchange={(e) => setParam('unindexedMotif', e.currentTarget.checked)}
              />
              Unindexed Motif (auto-place sequence position)
            </label>
            <label>
              Hydrogen-Bond Constraint
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
                Target Atoms / Ligand for H-Bond (e.g. <code>B/SAM/*/*</code>)
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
              Advanced JSON Constraint Override
              <textarea
                rows="3"
                class:input-error={rawJsonCheck.status === 'error'}
                {disabled}
                placeholder={'{"contig": "70-70", "conditions": {}}'}
                value={String(node.params.rawJsonOverride ?? '')}
                oninput={(e) => setParam('rawJsonOverride', e.currentTarget.value)}
              ></textarea>
              {#if rawJsonCheck.message}
                <span class="field-hint" class:err={rawJsonCheck.status === 'error'} class:ok={rawJsonCheck.status === 'ok'}>
                  {rawJsonCheck.message}
                </span>
              {/if}
            </label>
          </div>
        </details>
      {/if}

    {:else if node.adapterId === 'rfd4-proteina'}
      {@const rfdMode = String(node.params.mode ?? 'unconditional')}
      <label class="wide">
        Mode
        <select
          {disabled}
          value={rfdMode}
          onchange={(e) => setParam('mode', e.currentTarget.value)}
        >
          <option value="unconditional">Unconditional Monomer</option>
          <option value="conditioned">Conditioned (upstream block)</option>
          <option value="preset">Preset Design Task</option>
          <option value="partial">Partial Diffusion (refine input structure)</option>
          <option value="inverse-folding">Inverse Folding (design sequence for backbone)</option>
        </select>
      </label>
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
      {:else if rfdMode === 'preset'}
        <label class="full">
          Preset Design Task
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
        <label>
          Partial diffusion fraction
          <NumericInput
            {disabled}
            ariaLabel="Partial diffusion fraction"
            spec={NUMERIC_SPECS['rfd4-proteina'].partialFraction}
            value={Number(node.params.partialFraction ?? 0.5)}
            onChange={(v) => setParam('partialFraction', v)}
          />
        </label>
      {/if}

      <details class="adv-details full">
        <summary>Advanced parameters</summary>
        <div class="adv-fields">
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
          <label class="inline-check">
            <input
              type="checkbox"
              {disabled}
              checked={node.params.earlyStop !== false}
              onchange={(e) => setParam('earlyStop', e.currentTarget.checked)}
            />
            Early stop on convergence
          </label>
          {#if rfdMode === 'unconditional'}
            <label class="full">
              Custom constraint spec (YAML / JSON)
              <textarea
                rows="3"
                {disabled}
                placeholder="Leave blank for pure monomer, or wire an RFD4 Condition block into the condition port."
                value={String(node.params.customSpecYaml ?? '')}
                oninput={(e) => setParam('customSpecYaml', e.currentTarget.value)}
              ></textarea>
            </label>
          {/if}
        </div>
      </details>

    {:else if node.adapterId === 'genie3'}
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

      <details class="adv-details full">
        <summary>Advanced parameters</summary>
        <div class="adv-fields">
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
      <label class="wide">
        Model Variant
        <select
          {disabled}
          value={String(node.params.family ?? 'solublempnn')}
          onchange={(e) => setParam('family', e.currentTarget.value)}
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
        <summary>Advanced parameters</summary>
        <div class="adv-fields">
          <label class="wide">
            Checkpoint
            <select
              {disabled}
              value={String(node.params.modelName ?? 'v_48_020')}
              onchange={(e) => setParam('modelName', e.currentTarget.value)}
            >
              <option value="v_48_020">v_48_020 (default)</option>
              <option value="v_48_010">v_48_010</option>
            </select>
          </label>
          <label>
            Seed
            <NumericInput
              {disabled}
              ariaLabel="Seed"
              spec={NUMERIC_SPECS['web-protein-mpnn'].seed}
              value={Number(node.params.seed ?? 42)}
              onChange={(v) => setParam('seed', v)}
            />
          </label>
        </div>
      </details>

    {:else if node.adapterId === 'localfold-af2'}
      {@const lfModelId = String(node.params.model ?? 'monomer')}
      {@const lfMeta = selectedLocalFoldMeta ?? LOCALFOLD_MODEL_OPTIONS[0]}
      <label class="wide">
        Model
        <select
          {disabled}
          value={lfModelId}
          onchange={(e) => setParam('model', e.currentTarget.value)}
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
        <span class="field-hint">Single-sequence (no MSA) · Weights: ~{lfMeta.approxMB} MB · License: {lfMeta.license}</span>
      </label>
      <label>
        Recycles
        <NumericInput
          {disabled}
          ariaLabel="Recycles"
          spec={NUMERIC_SPECS['localfold-af2'].recycles}
          value={Number(node.params.recycles ?? 1)}
          onChange={(v) => setParam('recycles', v)}
        />
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
      {#if lfMeta.nonCommercial}
        <div class="full" style="font-size: 10.5px; color: var(--warn-text); background: var(--warn-bg); border: 1px solid var(--warn-border); padding: 6px 8px; border-radius: 5px;">
          AlphaFold 3 weights are subject to DeepMind's Non-Commercial Academic Terms. For permissive open weights, use <b>RoseTTAFold3</b> (BSD-3), <b>Boltz-2</b> (MIT), or <b>OpenBind-0</b> (Apache-2.0).
        </div>
      {/if}

      <details class="adv-details full">
        <summary>Advanced parameters</summary>
        <div class="adv-fields">
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
        </div>
      </details>

    {:else if node.adapterId === 'designability-scorer'}
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
  .sub-kind {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
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
</style>
