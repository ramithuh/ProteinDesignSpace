<script lang="ts">
  import { onMount } from 'svelte';

  interface Props {
    onClose: () => void;
  }

  let { onClose }: Props = $props();

  let closeBtn = $state<HTMLButtonElement | null>(null);

  onMount(() => {
    closeBtn?.focus();
  });

  function handleKeydown(ev: KeyboardEvent) {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      onClose();
    }
  }

  interface CreditItem {
    name: string;
    role: string;
    detail: string;
    license?: string;
    url: string;
    linkLabel: string;
  }

  const ENGINE_CREDITS: CreditItem[] = [
    {
      name: 'LocalFold',
      role: 'WebGPU Structure Prediction Runtime',
      detail:
        'Browser-native WebGPU engine by Sergey Ovchinnikov powering AlphaFold2, AF2-Multimer v3, RoseTTAFold3, Boltz-2, Protenix-v2, OpenBind-0 (OpenFold3 v0.5.0), OpenDDE, Mol-1, IntelliFold-2 & AlphaFold 3 refolding.',
      license: 'Beer-Ware (code); upstream weight terms',
      url: 'https://github.com/sokrypton/localfold',
      linkLabel: 'sokrypton/localfold',
    },
    {
      name: 'RFD4-Proteina WebGPU',
      role: 'All-Atom Structure & Sequence Co-Design',
      detail:
        'WebGPU INT8 inference runtime by Sergey Ovchinnikov running NVIDIA RFD4-Proteina weights for motif scaffolding, binder design, and all-atom enzyme/ligand/DNA tasks.',
      license: 'MIT (code) · BSD-3-Clause (NVIDIA weights)',
      url: 'https://github.com/sokrypton/rfd4-proteina-webgpu',
      linkLabel: 'sokrypton/rfd4-proteina-webgpu',
    },
    {
      name: 'Genie 3',
      role: 'SE(3) Diffusion Backbone Generator',
      detail:
        'Fast 23.3M-parameter backbone diffusion & SequenceNet architecture (AlQuraishi Lab), with browser WebGPU FP16 shards streamed from Hugging Face.',
      license: 'MIT',
      url: 'https://huggingface.co/ramithuh/genie3-webgpu',
      linkLabel: 'ramithuh/genie3-webgpu',
    },
    {
      name: 'webProteinMPNN',
      role: 'Inverse Folding (ProteinMPNN & SolubleMPNN)',
      detail:
        'Browser ONNX WebGPU wrapper by the Murrell Group for ProteinMPNN, SolubleMPNN, and Cα-MPNN (Dauparas et al., Institute for Protein Design, UW).',
      license: 'MIT',
      url: 'https://github.com/MurrellGroup/webProteinMPNN',
      linkLabel: 'MurrellGroup/webProteinMPNN',
    },
    {
      name: 'ThermoMPNN',
      role: 'Point-Mutation Stability Prediction (ΔΔG)',
      detail:
        'Transfer-learned protein stability predictor (Dieckhaus et al., PNAS 2024; Kuhlman Lab, UNC) computing full L×20 site-saturation ΔΔG landscapes in browser WebGPU.',
      license: 'MIT',
      url: 'https://github.com/Kuhlman-Lab/ThermoMPNN',
      linkLabel: 'Kuhlman-Lab/ThermoMPNN',
    },
    {
      name: 'LigandMPNN',
      role: 'Ligand- & Nucleic-Acid-Aware Inverse Folding',
      detail:
        'Atomic context-conditioned sequence design preserving small molecules, metal ions, and DNA/RNA (Dauparas et al., Nature Methods 2025; Baker Lab), ported to native WebGPU.',
      license: 'MIT',
      url: 'https://github.com/dauparas/LigandMPNN',
      linkLabel: 'dauparas/LigandMPNN',
    },
    {
      name: 'LASErMPNN',
      role: 'Ligand-Aware Sequence & Sidechain Rotamer Co-Design',
      detail:
        'All-atom inverse folding and discrete χ1–χ4 sidechain rotamer packing with an embedded SPICE ligand encoder (Polizzi Lab, Harvard/DFCI), ported to native WebGPU.',
      license: 'MIT',
      url: 'https://github.com/polizzilab/LASErMPNN',
      linkLabel: 'polizzilab/LASErMPNN',
    },
    {
      name: 'AntiFold & ESM-IF1',
      role: 'Antibody CDR & General GVP-Transformer Inverse Folding',
      detail:
        'Antibody and nanobody CDR inverse folding by OPIG (Høie et al., Bioinformatics 2024) built on Meta AI ESM-IF1 (Hsu et al., ICML 2022), running natively in WebGPU.',
      license: 'BSD-3-Clause (AntiFold) · MIT (Meta ESM-IF1)',
      url: 'https://github.com/oxpig/AntiFold',
      linkLabel: 'oxpig/AntiFold',
    },
  ];

  const TOOL_CREDITS: CreditItem[] = [
    {
      name: 'Mol*',
      role: '3D Molecular Viewer',
      detail: 'High-performance WebGL2 cartoon, ball-and-stick, and pLDDT structure visualization.',
      license: 'MIT',
      url: 'https://github.com/molstar/molstar',
      linkLabel: 'molstar/molstar',
    },
    {
      name: 'Svelte Flow (@xyflow/svelte)',
      role: 'Interactive DAG Canvas',
      detail: 'Node-graph workflow editor for wiring and branching protein design pipelines.',
      license: 'MIT',
      url: 'https://svelteflow.dev',
      linkLabel: 'svelteflow.dev',
    },
    {
      name: 'ONNX Runtime Web',
      role: 'In-Browser ML Runtime',
      detail: 'WebGPU and WebAssembly execution provider used by inverse folding models.',
      license: 'MIT',
      url: 'https://github.com/microsoft/onnxruntime',
      linkLabel: 'microsoft/onnxruntime',
    },
    {
      name: 'RCSB Protein Data Bank',
      role: 'Experimental Structure & Benchmark Data',
      detail:
        'Live mmCIF/PDB fetching for custom targets and curated RFdiffusion, BindCraft & AME benchmark motifs.',
      license: 'CC0 / PDB Data Policy',
      url: 'https://www.rcsb.org',
      linkLabel: 'rcsb.org',
    },
    {
      name: 'ColabFold MMseqs2 API',
      role: 'Evolutionary Multiple Sequence Alignments (Optional)',
      detail:
        'Fast MMseqs2 homology search service (api.colabfold.com; Mirdita et al., Nature Methods 2022) used only when MMseqs2 or Target-Only MSA mode is enabled on LocalFold.',
      license: 'MIT (ColabFold)',
      url: 'https://github.com/sokrypton/ColabFold',
      linkLabel: 'sokrypton/ColabFold',
    },
    {
      name: 'ipSAE',
      role: 'Multi-Chain Interface Quality Scoring',
      detail:
        'Interface Predicted Structural Alignment Error (Dunbrack, 2025) for evaluating binder-target and complex interfaces directly from PAE matrices.',
      license: 'MIT',
      url: 'https://github.com/DunbrackLab/IPSAE',
      linkLabel: 'DunbrackLab/IPSAE',
    },
  ];
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="modal-backdrop cr-backdrop"
  role="presentation"
  onclick={(ev) => {
    if (ev.target === ev.currentTarget) onClose();
  }}
>
  <div
    class="modal-card"
    role="dialog"
    aria-modal="true"
    aria-labelledby="credits-title"
  >
    <div class="modal-hdr">
      <div class="hdr-top">
        <div class="hdr-badge">OPEN-SOURCE CREDITS &amp; LICENSES</div>
        <button
          type="button"
          class="icon-close"
          aria-label="Close credits"
          onclick={onClose}
        >
          ×
        </button>
      </div>
      <h2 id="credits-title">Built on open protein design tools</h2>
      <p class="hdr-sub">
        <strong>denovo.run</strong> runs entirely in your browser over WebGPU — no server compute
        and no data uploads in default local mode (optional ColabFold MMseqs2 MSA queries only when
        enabled). Huge thanks to the researchers and open-source maintainers whose models, WebGPU
        runtimes, and libraries make this possible.
      </p>
    </div>

    <div class="credits-scroll">
      <section class="credit-sec">
        <div class="sec-title">WebGPU Models &amp; Inference Runtimes</div>
        <div class="credit-list">
          {#each ENGINE_CREDITS as item (item.name)}
            <div class="credit-row">
              <div class="credit-head">
                <div class="credit-name-wrap">
                  <strong class="credit-name">{item.name}</strong>
                  <span class="credit-role">{item.role}</span>
                </div>
                <a
                  class="credit-link"
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {item.linkLabel} ↗
                </a>
              </div>
              <p class="credit-detail">{item.detail}</p>
              {#if item.license}
                <div class="credit-lic">{item.license}</div>
              {/if}
            </div>
          {/each}
        </div>
      </section>

      <section class="credit-sec">
        <div class="sec-title">Visualization, Canvas &amp; Data</div>
        <div class="credit-list">
          {#each TOOL_CREDITS as item (item.name)}
            <div class="credit-row">
              <div class="credit-head">
                <div class="credit-name-wrap">
                  <strong class="credit-name">{item.name}</strong>
                  <span class="credit-role">{item.role}</span>
                </div>
                <a
                  class="credit-link"
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {item.linkLabel} ↗
                </a>
              </div>
              <p class="credit-detail">{item.detail}</p>
              {#if item.license}
                <div class="credit-lic">{item.license}</div>
              {/if}
            </div>
          {/each}
        </div>
      </section>

      <p class="weights-note">
        Model weights are streamed directly from upstream repositories into your browser's local
        storage (OPFS) and remain subject to their respective upstream model licenses (e.g.,
        Google DeepMind AlphaFold2 / AlphaFold 3 terms, NVIDIA RFD4-Proteina BSD-3-Clause,
        RosettaCommons RF3, MIT Boltz-2, ByteDance Protenix, and OpenBind-0 (OpenFold3 v0.5.0)).
      </p>
    </div>

    <div class="modal-actions">
      <button
        type="button"
        class="btn-close"
        bind:this={closeBtn}
        onclick={onClose}
      >
        Done
      </button>
    </div>
  </div>
</div>

<style>
  .modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 200;
    background: rgba(10, 14, 24, 0.62);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    box-sizing: border-box;
  }

  .modal-card {
    width: 100%;
    max-width: min(560px, calc(100vw - 24px));
    max-height: min(86vh, 720px);
    box-sizing: border-box;
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    padding: 18px 20px 16px;
    color: var(--text-body);
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .modal-hdr {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex-shrink: 0;
  }

  .hdr-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .hdr-badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent-bright);
    background: var(--bg-pill);
    padding: 2px 8px;
    border-radius: 999px;
  }

  .icon-close {
    width: 26px;
    height: 26px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .icon-close:hover {
    background: var(--bg-surface-hover);
    border-color: var(--border);
    color: var(--text-primary);
  }

  h2 {
    margin: 2px 0 0;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .hdr-sub {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.45;
    color: var(--text-muted);
  }

  .credits-scroll {
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding-right: 4px;
  }

  .credit-sec {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .sec-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.45px;
    color: var(--text-dim);
  }

  .credit-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .credit-row {
    background: var(--bg-surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 11px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .credit-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .credit-name-wrap {
    display: inline-flex;
    align-items: baseline;
    gap: 7px;
    flex-wrap: wrap;
  }

  .credit-name {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .credit-role {
    font-size: 11px;
    color: var(--text-muted);
  }

  .credit-link {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--accent-bright);
    text-decoration: none;
    white-space: nowrap;
  }

  .credit-link:hover {
    text-decoration: underline;
  }

  .credit-detail {
    margin: 0;
    font-size: 11.8px;
    line-height: 1.42;
    color: var(--text-secondary);
  }

  .credit-lic {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
  }

  .weights-note {
    margin: 0;
    font-size: 11px;
    line-height: 1.45;
    color: var(--text-dim);
    border-top: 1px solid var(--border-subtle);
    padding-top: 10px;
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    padding-top: 2px;
    flex-shrink: 0;
  }

  .btn-close {
    padding: 6px 14px;
    border-radius: 7px;
    border: 1px solid var(--border-strong);
    background: var(--bg-surface-2);
    color: var(--text-primary);
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
  }

  .btn-close:hover {
    background: var(--bg-surface-hover);
    border-color: var(--accent);
  }
</style>
