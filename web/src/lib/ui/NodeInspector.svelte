<script lang="ts">
  import {
    ADAPTER_REGISTRY,
    LOCALFOLD_MODEL_OPTIONS,
    RFD4_BENCHMARK_OPTIONS,
  } from '$lib/engine/registry';
  import { STAGE_COLORS } from '$lib/engine/graph-utils';
  import type { WorkflowNode } from '$lib/engine/types';

  interface Props {
    node: WorkflowNode;
    disabled?: boolean;
    embedded?: boolean;
    onUpdate: (nodeId: string, params: Record<string, unknown>) => void;
    onRunUpTo?: (nodeId: string) => void;
  }

  let {
    node,
    disabled = false,
    embedded = false,
    onUpdate,
    onRunUpTo = undefined,
  }: Props = $props();

  let manifest = $derived(ADAPTER_REGISTRY[node.adapterId]);
  let stageColor = $derived(STAGE_COLORS[manifest?.stage ?? 'source'] ?? '#64748b');

  let activeModelBadge = $derived.by((): string => {
    const p = node.params ?? {};
    switch (node.adapterId) {
      case 'localfold-af2': {
        const modelId = String(p.model ?? 'monomer');
        const meta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === modelId) ?? LOCALFOLD_MODEL_OPTIONS[0];
        return `${meta.label} · INT5 (~${meta.approxMB} MB · ${meta.license})`;
      }
      case 'rfd4-proteina': {
        const mode = String(p.mode ?? 'unconditional');
        const needsEnc =
          mode === 'partial' ||
          mode === 'inverse-folding' ||
          (mode === 'preset' &&
            Boolean(RFD4_BENCHMARK_OPTIONS.find((o) => o.id === String(p.presetId ?? ''))?.needsEncoder));
        return needsEnc
          ? 'RFD4-Proteina · INT8 + 138M encoder'
          : 'RFD4-Proteina · INT8 (FP16 compute)';
      }
      case 'web-protein-mpnn': {
        const fam =
          p.family === 'proteinmpnn'
            ? 'ProteinMPNN'
            : p.family === 'ca'
              ? 'Cα-ProteinMPNN'
              : 'SolubleMPNN';
        const ckpt = String(p.modelName ?? 'v_48_020');
        return `${fam} · ${ckpt} (ONNX)`;
      }
      case 'rfd4-conditioner':
        return 'RFD4 condition_spec DSL';
      case 'designability-scorer':
        return 'Kabsch SVD · scRMSD / scTM';
      case 'pdb-input':
        return 'PDB / mmCIF parser';
      case 'sequence-input':
        return 'FASTA parser';
      default:
        return manifest?.shortName ?? node.adapterId;
    }
  });

  function setParam(key: string, value: unknown) {
    onUpdate(node.id, { ...node.params, [key]: value });
  }

  function setNumParam(key: string, raw: string) {
    if (raw.trim() === '') return;
    const n = Number(raw);
    if (Number.isFinite(n)) setParam(key, n);
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
          onchange={(e) => setParam('mode', e.currentTarget.value)}
        >
          <option value="demo-bundle">Built-in 53 aa 3-helix bundle demo</option>
          <option value="custom">Upload or paste PDB / mmCIF</option>
        </select>
      </label>

      {#if node.params.mode === 'custom'}
        <label class="file-row full">
          Upload .pdb / .cif
          <input type="file" accept=".pdb,.cif,.mmcif,.ent,.txt" {disabled} onchange={handleFileUpload} />
        </label>
        <label class="full">
          PDB / mmCIF text ({String(node.params.filename || 'custom.pdb')})
          <textarea
            rows="4"
            {disabled}
            placeholder="Paste ATOM lines or mmCIF _atom_site block…"
            value={String(node.params.pdbText ?? '')}
            oninput={(e) => setParam('pdbText', e.currentTarget.value)}
          ></textarea>
        </label>
      {/if}

    {:else if node.adapterId === 'sequence-input'}
      <label class="full">
        Amino acid sequence (plain or multi-FASTA)
        <textarea
          rows="5"
          {disabled}
          placeholder=">seq1&#10;MKTAYIAKQRQISFVK..."
          value={String(node.params.sequence ?? '')}
          oninput={(e) => setParam('sequence', e.currentTarget.value)}
        ></textarea>
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
      <label class="full">
        Task / Benchmark Preset
        <select
          {disabled}
          value={String(node.params.taskPreset ?? 'custom')}
          onchange={(e) => setParam('taskPreset', e.currentTarget.value)}
        >
          <option value="custom">Custom Conditioning Builder</option>
          {#each RFD4_BENCHMARK_OPTIONS as opt (opt.id)}
            <option value={opt.id}>[{opt.category}] {opt.label}</option>
          {/each}
        </select>
      </label>

      {#if String(node.params.taskPreset ?? 'custom') === 'custom'}
        <label class="full">
          Contig (e.g. <code>70-70</code>, <code>A18-132,/0,70-70</code>)
          <input
            type="text"
            {disabled}
            value={String(node.params.contig ?? '70-70')}
            oninput={(e) => setParam('contig', e.currentTarget.value)}
          />
        </label>

        <div class="chip-field full">
          <label>
            Hotspots (<code>C_HOT</code> — click in Mol* or type e.g. A54, A56)
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
                <span class="res-chip">
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
            Fixed Motif (<code>C_CRDGRP</code> — e.g. A16-35, A52-71)
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
                <span class="res-chip">
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
                <option value="hotspots">Paint Hotspots (C_HOT)</option>
                <option value="motifSpan">Paint Motif (C_CRDGRP)</option>
              </select>
            </label>
            <label class="inline-check">
              <input
                type="checkbox"
                {disabled}
                checked={Boolean(node.params.unindexedMotif)}
                onchange={(e) => setParam('unindexedMotif', e.currentTarget.checked)}
              />
              Unindexed Motif (<code>C_IDX=false</code>)
            </label>
            <label>
              H-Bond (<code>C_HB</code>)
              <select
                {disabled}
                value={String(node.params.hbondMode ?? 'none')}
                onchange={(e) => setParam('hbondMode', e.currentTarget.value)}
              >
                <option value="none">None</option>
                <option value="ACCEPTOR">ACCEPTOR</option>
                <option value="DONOR">DONOR</option>
                <option value="BOTH">BOTH</option>
              </select>
            </label>
            {#if String(node.params.hbondMode ?? 'none') !== 'none'}
              <label class="full">
                H-Bond Selection (<code>sel(...)</code>, e.g. <code>B/SAM/*/*</code>)
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
              Raw RFD4 <code>condition_spec</code> JSON override
              <textarea
                rows="3"
                {disabled}
                placeholder={'{"contig": "70-70", "conditions": {"C_CATH": {"3.40.50": [{"select": "sel(\'*/*/*/*\')"}]}}}'}
                value={String(node.params.rawJsonOverride ?? '')}
                oninput={(e) => setParam('rawJsonOverride', e.currentTarget.value)}
              ></textarea>
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
          <option value="preset">Official RFD4 Benchmark</option>
          <option value="partial">Partial Diffusion (+138M enc)</option>
          <option value="inverse-folding">Inverse Folding (+138M enc)</option>
        </select>
      </label>
      {#if rfdMode === 'unconditional'}
        <label>
          Length (aa)
          <input
            type="number"
            min="30"
            max="200"
            {disabled}
            value={Number(node.params.length ?? 70)}
            oninput={(e) => setNumParam('length', e.currentTarget.value)}
          />
        </label>
      {:else if rfdMode === 'preset'}
        <label class="full">
          Upstream Benchmark
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
          Partial frac
          <input
            type="number"
            min="0.1"
            max="0.9"
            step="0.05"
            {disabled}
            value={Number(node.params.partialFraction ?? 0.5)}
            oninput={(e) => setNumParam('partialFraction', e.currentTarget.value)}
          />
        </label>
      {/if}

      <details class="adv-details full">
        <summary>Advanced parameters</summary>
        <div class="adv-fields">
          <label>
            Steps
            <input
              type="number"
              min="10"
              max="120"
              {disabled}
              value={Number(node.params.nsteps ?? 60)}
              oninput={(e) => setNumParam('nsteps', e.currentTarget.value)}
            />
          </label>
          <label>
            Recycle
            <select
              {disabled}
              value={String(node.params.recycleSchedule ?? 'low')}
              onchange={(e) => setParam('recycleSchedule', e.currentTarget.value)}
            >
              <option value="low">low (fast)</option>
              <option value="medium">medium (higher quality)</option>
            </select>
          </label>
          <label>
            Seed
            <input
              type="number"
              min="0"
              {disabled}
              value={Number(node.params.seed ?? 42)}
              oninput={(e) => setNumParam('seed', e.currentTarget.value)}
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
              Inline RFD4 YAML/JSON spec
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

    {:else if node.adapterId === 'web-protein-mpnn'}
      {@const fixedTokens = parseResidueTokens(node.params.fixedPositions)}
      <label class="wide">
        Family
        <select
          {disabled}
          value={String(node.params.family ?? 'solublempnn')}
          onchange={(e) => setParam('family', e.currentTarget.value)}
        >
          <option value="solublempnn">SolubleMPNN</option>
          <option value="proteinmpnn">ProteinMPNN (vanilla)</option>
          <option value="ca">ProteinMPNN (Cα-only)</option>
        </select>
      </label>
      <label>
        Sequences
        <input
          type="number"
          min="1"
          max="8"
          {disabled}
          value={Number(node.params.numSequences ?? 2)}
          oninput={(e) => setNumParam('numSequences', e.currentTarget.value)}
        />
      </label>
      <label>
        Temp
        <input
          type="number"
          min="0.01"
          max="1.5"
          step="0.05"
          {disabled}
          value={Number(node.params.temperature ?? 0.1)}
          oninput={(e) => setNumParam('temperature', e.currentTarget.value)}
        />
      </label>

      <div class="chip-field full">
        <label>
          Fixed positions (click residues in Mol* or type e.g. A10, A12-15)
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
              <span class="res-chip">
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
            <input
              type="number"
              min="0"
              {disabled}
              value={Number(node.params.seed ?? 42)}
              oninput={(e) => setNumParam('seed', e.currentTarget.value)}
            />
          </label>
        </div>
      </details>

    {:else if node.adapterId === 'localfold-af2'}
      {@const lfModelId = String(node.params.model ?? 'monomer')}
      {@const lfMeta = LOCALFOLD_MODEL_OPTIONS.find((m) => m.id === lfModelId) ?? LOCALFOLD_MODEL_OPTIONS[0]}
      <label class="wide">
        Model
        <select
          {disabled}
          value={lfModelId}
          onchange={(e) => setParam('model', e.currentTarget.value)}
        >
          <optgroup label="AlphaFold 2 (Monomer INT5)">
            {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2') as opt (opt.id)}
              <option value={opt.id}>{opt.label} (~{opt.approxMB} MB)</option>
            {/each}
          </optgroup>
          <optgroup label="All-Atom / AF3-Lineage (INT5)">
            {#each LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3') as opt (opt.id)}
              <option value={opt.id}>{opt.label} (~{opt.approxMB} MB · {opt.license})</option>
            {/each}
          </optgroup>
        </select>
      </label>
      <label>
        Recycles
        <input
          type="number"
          min="0"
          max="4"
          {disabled}
          value={Number(node.params.recycles ?? 1)}
          oninput={(e) => setNumParam('recycles', e.currentTarget.value)}
        />
      </label>
      {#if lfMeta.family === 'af3'}
        <label>
          Diffusion steps
          <input
            type="number"
            min="10"
            max="200"
            step="5"
            {disabled}
            value={Number(node.params.diffusionSteps ?? 25)}
            oninput={(e) => setNumParam('diffusionSteps', e.currentTarget.value)}
          />
        </label>
      {/if}
      <label>
        Max seqs to fold
        <input
          type="number"
          min="1"
          max="8"
          {disabled}
          value={Number(node.params.maxSequences ?? 2)}
          oninput={(e) => setNumParam('maxSequences', e.currentTarget.value)}
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
            <input
              type="number"
              min="0"
              {disabled}
              value={Number(node.params.seed ?? 0)}
              oninput={(e) => setNumParam('seed', e.currentTarget.value)}
            />
          </label>
        </div>
      </details>

    {:else if node.adapterId === 'designability-scorer'}
      <label>
        scRMSD &lt; Å
        <input
          type="number"
          min="0.5"
          max="5"
          step="0.1"
          {disabled}
          value={Number(node.params.scRmsdThreshold ?? 2.0)}
          oninput={(e) => setNumParam('scRmsdThreshold', e.currentTarget.value)}
        />
      </label>
      <label>
        pLDDT &ge;
        <input
          type="number"
          min="30"
          max="95"
          step="5"
          {disabled}
          value={Number(node.params.plddtThreshold ?? 70)}
          oninput={(e) => setNumParam('plddtThreshold', e.currentTarget.value)}
        />
      </label>
    {/if}
  </div>

  {#if manifest}
    <footer class="info-footer">
      <p class="desc">{manifest.description}</p>
      <div class="foot-meta">
        <span>{manifest.runtime}</span>
        {#if manifest.approxWeightsMB > 0}
          <span>· ~{manifest.approxWeightsMB} MB</span>
        {/if}
        {#if manifest.licenseNote}
          <span>· {manifest.licenseNote}</span>
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
    align-items: flex-end;
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
    align-items: flex-end;
    margin-top: 10px;
  }
  input[type='number'],
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
  input[type='number'] {
    width: 82px;
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
