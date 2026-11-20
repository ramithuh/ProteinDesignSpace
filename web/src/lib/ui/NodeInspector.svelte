<script lang="ts">
  import { ADAPTER_REGISTRY, RFD4_BENCHMARK_OPTIONS } from '$lib/engine/registry';
  import type { WorkflowNode } from '$lib/engine/types';

  interface Props {
    node: WorkflowNode;
    disabled?: boolean;
    onUpdate: (nodeId: string, params: Record<string, unknown>) => void;
  }

  let { node, disabled = false, onUpdate }: Props = $props();
  let manifest = $derived(ADAPTER_REGISTRY[node.adapterId]);

  function setParam(key: string, value: unknown) {
    onUpdate(node.id, { ...node.params, [key]: value });
  }

  async function handleFileUpload(ev: Event) {
    const input = ev.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const text = await file.text();
    onUpdate(node.id, {
      ...node.params,
      mode: 'custom',
      filename: file.name,
      pdbText: text,
    });
  }
</script>

<div class="inspector">
  <div class="head">
    <div>
      <h3>{manifest?.displayName ?? node.adapterId}</h3>
      <p class="desc">{manifest?.description}</p>
    </div>
    <div class="meta">
      <span class="pill">{manifest?.runtime}</span>
      {#if manifest && manifest.approxWeightsMB > 0}
        <span class="pill">~{manifest.approxWeightsMB} MB</span>
      {/if}
    </div>
  </div>

  <div class="fields">
    {#if node.adapterId === 'pdb-input'}
      <label>
        Source
        <select
          {disabled}
          value={String(node.params.mode ?? 'demo-bundle')}
          onchange={(e) => setParam('mode', e.currentTarget.value)}
        >
          <option value="demo-bundle">Built-in 56 aa 3-helix bundle demo</option>
          <option value="custom">Upload or paste PDB / mmCIF</option>
        </select>
      </label>

      {#if node.params.mode === 'custom'}
        <label class="file-row">
          Upload .pdb / .cif
          <input type="file" accept=".pdb,.cif,.mmcif,.ent,.txt" {disabled} onchange={handleFileUpload} />
        </label>
        <label class="full">
          PDB / mmCIF text ({String(node.params.filename || 'custom.pdb')})
          <textarea
            rows="4"
            {disabled}
            placeholder="Paste ATOM lines or mmCIF _atom_site block (e.g. exported from web.rfd4-proteina.org)..."
            value={String(node.params.pdbText ?? '')}
            oninput={(e) => setParam('pdbText', e.currentTarget.value)}
          ></textarea>
        </label>
      {/if}
    {:else if node.adapterId === 'sequence-input'}
      <label class="wide">
        Label
        <input
          type="text"
          {disabled}
          value={String(node.params.label ?? 'custom-seq-1')}
          oninput={(e) => setParam('label', e.currentTarget.value)}
        />
      </label>
      <label class="full">
        Amino acid sequence (plain or multi-FASTA)
        <textarea
          rows="4"
          {disabled}
          placeholder=">seq1&#10;MKTAYIAKQRQISFVK..."
          value={String(node.params.sequence ?? '')}
          oninput={(e) => setParam('sequence', e.currentTarget.value)}
        ></textarea>
      </label>
    {:else if node.adapterId === 'rfd4-conditioner'}
      <label class="wide">
        Task / Benchmark Preset
        <select
          {disabled}
          value={String(node.params.taskPreset ?? 'custom')}
          onchange={(e) => setParam('taskPreset', e.currentTarget.value)}
        >
          <option value="custom">Custom Conditioning Builder (below)</option>
          {#each RFD4_BENCHMARK_OPTIONS as opt (opt.id)}
            <option value={opt.id}>[{opt.category}] {opt.label}</option>
          {/each}
        </select>
      </label>

      {#if String(node.params.taskPreset ?? 'custom') === 'custom'}
        <label class="wide">
          Contig (e.g. <code>70-70</code>, <code>A18-132,/0,70-70</code>, <code>10-20,A16-35,10-20</code>)
          <input
            type="text"
            {disabled}
            value={String(node.params.contig ?? '70-70')}
            oninput={(e) => setParam('contig', e.currentTarget.value)}
          />
        </label>
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
        <label class="wide">
          Hotspots (<code>C_HOT</code> — click in Mol* or type e.g. A54, A56, A66)
          <input
            type="text"
            {disabled}
            placeholder="A54, A56, A66, A115"
            value={String(node.params.hotspots ?? '')}
            oninput={(e) => setParam('hotspots', e.currentTarget.value)}
          />
        </label>
        <label class="wide">
          Fixed Motif (<code>C_CRDGRP</code> + <code>C_SEQ</code> — e.g. A16-35, A52-71)
          <input
            type="text"
            {disabled}
            placeholder="A16-35, A52-71"
            value={String(node.params.motifSpan ?? '')}
            oninput={(e) => setParam('motifSpan', e.currentTarget.value)}
          />
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
          <label class="wide">
            H-Bond Selection (<code>sel(...)</code>, e.g. <code>B/SAM/*/*</code> or <code>A/*/42/OG</code>)
            <input
              type="text"
              {disabled}
              placeholder="B/SAM/*/*"
              value={String(node.params.hbondSelection ?? '')}
              oninput={(e) => setParam('hbondSelection', e.currentTarget.value)}
            />
          </label>
        {/if}
        <details class="full">
          <summary>Advanced: Raw RFD4 condition_spec JSON override</summary>
          <textarea
            rows="4"
            {disabled}
            placeholder={'{"contig": "70-70", "conditions": {"C_CATH": {"3.40.50": [{"select": "sel(\'*/*/*/*\')"}]}}}'}
            value={String(node.params.rawJsonOverride ?? '')}
            oninput={(e) => setParam('rawJsonOverride', e.currentTarget.value)}
          ></textarea>
        </details>
      {/if}
    {:else if node.adapterId === 'rfd4-proteina'}
      {@const rfdMode = String(node.params.mode ?? 'unconditional')}
      <label>
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
            oninput={(e) => setParam('length', Number(e.currentTarget.value))}
          />
        </label>
      {:else if rfdMode === 'preset'}
        <label class="wide">
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
            oninput={(e) => setParam('partialFraction', Number(e.currentTarget.value))}
          />
        </label>
      {/if}
      <label>
        Steps
        <input
          type="number"
          min="10"
          max="100"
          {disabled}
          value={Number(node.params.nsteps ?? 30)}
          oninput={(e) => setParam('nsteps', Number(e.currentTarget.value))}
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
          oninput={(e) => setParam('seed', Number(e.currentTarget.value))}
        />
      </label>
      {#if rfdMode === 'unconditional'}
        <details class="full">
          <summary>Optional inline RFD4 YAML/JSON spec</summary>
          <textarea
            rows="3"
            {disabled}
            placeholder={'Leave blank for pure monomer, or wire an RFD4 Condition block into the condition port.'}
            value={String(node.params.customSpecYaml ?? '')}
            oninput={(e) => setParam('customSpecYaml', e.currentTarget.value)}
          ></textarea>
        </details>
      {/if}
    {:else if node.adapterId === 'legacy-teacher'}
      <label>
        Sampler
        <select
          {disabled}
          value={String(node.params.sampler ?? 'sde-proteina')}
          onchange={(e) => setParam('sampler', e.currentTarget.value)}
        >
          <option value="sde-proteina">SDE-Proteina</option>
          <option value="euler-fm">Euler-FM (ODE)</option>
        </select>
      </label>
      <label>
        Length
        <input
          type="number"
          min="30"
          max="200"
          {disabled}
          value={Number(node.params.length ?? 80)}
          oninput={(e) => setParam('length', Number(e.currentTarget.value))}
        />
      </label>
      <label>
        NFE
        <input
          type="number"
          min="4"
          max="400"
          {disabled}
          value={Number(node.params.nfe ?? 32)}
          oninput={(e) => setParam('nfe', Number(e.currentTarget.value))}
        />
      </label>
      <label>
        Seed
        <input
          type="number"
          min="0"
          {disabled}
          value={Number(node.params.seed ?? 42)}
          oninput={(e) => setParam('seed', Number(e.currentTarget.value))}
        />
      </label>
      {#if node.params.sampler === 'sde-proteina'}
        <label>
          γ
          <input
            type="number"
            min="0"
            max="2"
            step="0.05"
            {disabled}
            value={Number(node.params.sdeGamma ?? 0.1)}
            oninput={(e) => setParam('sdeGamma', Number(e.currentTarget.value))}
          />
        </label>
        <label>
          g(t) max
          <input
            type="number"
            min="0"
            max="20"
            step="0.5"
            {disabled}
            value={Number(node.params.sdeGtMax ?? 3.0)}
            oninput={(e) => setParam('sdeGtMax', Number(e.currentTarget.value))}
          />
        </label>
      {/if}
    {:else if node.adapterId === 'web-protein-mpnn'}
      <label>
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
        Sequences
        <input
          type="number"
          min="1"
          max="8"
          {disabled}
          value={Number(node.params.numSequences ?? 2)}
          oninput={(e) => setParam('numSequences', Number(e.currentTarget.value))}
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
          oninput={(e) => setParam('temperature', Number(e.currentTarget.value))}
        />
      </label>
      <label>
        Seed
        <input
          type="number"
          min="0"
          {disabled}
          value={Number(node.params.seed ?? 42)}
          oninput={(e) => setParam('seed', Number(e.currentTarget.value))}
        />
      </label>
      <label class="wide">
        Fixed positions (click residues in Mol* or type e.g. A10,A12-15)
        <input
          type="text"
          {disabled}
          placeholder="e.g. A5,A10-14"
          value={String(node.params.fixedPositions ?? '')}
          oninput={(e) => setParam('fixedPositions', e.currentTarget.value)}
        />
      </label>
    {:else if node.adapterId === 'localfold-af2'}
      <label>
        Recycles
        <input
          type="number"
          min="0"
          max="4"
          {disabled}
          value={Number(node.params.recycles ?? 1)}
          oninput={(e) => setParam('recycles', Number(e.currentTarget.value))}
        />
      </label>
      <label>
        Max seqs to fold
        <input
          type="number"
          min="1"
          max="8"
          {disabled}
          value={Number(node.params.maxSequences ?? 2)}
          oninput={(e) => setParam('maxSequences', Number(e.currentTarget.value))}
        />
      </label>
      <label>
        Seed
        <input
          type="number"
          min="0"
          {disabled}
          value={Number(node.params.seed ?? 0)}
          oninput={(e) => setParam('seed', Number(e.currentTarget.value))}
        />
      </label>
    {:else if node.adapterId === 'designability-scorer'}
      <label>
        scRMSD pass (&lt; Å)
        <input
          type="number"
          min="0.5"
          max="5"
          step="0.1"
          {disabled}
          value={Number(node.params.scRmsdThreshold ?? 2.0)}
          oninput={(e) => setParam('scRmsdThreshold', Number(e.currentTarget.value))}
        />
      </label>
      <label>
        pLDDT pass (&ge;)
        <input
          type="number"
          min="30"
          max="95"
          step="5"
          {disabled}
          value={Number(node.params.plddtThreshold ?? 70)}
          oninput={(e) => setParam('plddtThreshold', Number(e.currentTarget.value))}
        />
      </label>
    {/if}
  </div>

  {#if manifest?.licenseNote}
    <div class="license">{manifest.licenseNote}</div>
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
  .head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }
  h3 {
    margin: 0 0 4px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .desc {
    margin: 0;
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.45;
  }
  .meta {
    display: flex;
    gap: 6px;
    flex-shrink: 0;
  }
  .pill {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--accent-soft);
    background: var(--bg-pill);
    border: 1px solid var(--border-strong);
    padding: 2px 7px;
    border-radius: 999px;
  }
  .fields {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 14px;
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
    min-width: 240px;
    flex: 1;
  }
  label.full,
  details.full {
    width: 100%;
  }
  label.inline-check {
    flex-direction: row;
    align-items: center;
    gap: 6px;
    padding-bottom: 6px;
  }
  details summary {
    cursor: pointer;
    font-size: 11px;
    color: var(--accent-soft);
    margin-bottom: 6px;
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
    width: 84px;
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    resize: vertical;
  }
  .license {
    font-size: 10px;
    color: var(--text-dim);
    border-top: 1px solid var(--border-subtle);
    padding-top: 8px;
  }
</style>
