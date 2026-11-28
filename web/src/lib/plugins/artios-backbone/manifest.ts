// Layer 3 Plugin Manifest — artios-backbone (ArtiosBackbone all-atom EDM diffusion · Ligo Biosciences)
// Declared here so the plugin folder stays self-contained; `ADAPTER_REGISTRY['artios-backbone']`
// in `$lib/engine/registry` re-exports this exact object (single source of truth, no duplication).

import type { AdapterManifest } from '$lib/engine/types';

/** INT8 / FP16 / FP32 bundle sizes in MiB, measured from `tools/backbone/export_backbone.py` manifests. */
export const ARTIOS_BACKBONE_WEIGHTS_MB = Object.freeze({
  int8: 131.3,
  fp16: 260.7,
  fp32: 521.4,
});

export const ARTIOS_BACKBONE_MANIFEST: AdapterManifest = {
  id: 'artios-backbone',
  displayName: 'ArtiosBackbone All-Atom Diffusion',
  shortName: 'ArtiosBackbone',
  stage: 'generate',
  runtime: 'wgsl-custom',
  estimatedVramMB: 200,
  approxWeightsMB: ARTIOS_BACKBONE_WEIGHTS_MB.int8,
  precisionWeightsMB: { ...ARTIOS_BACKBONE_WEIGHTS_MB },
  licenseNote:
    'Apache-2.0 (Ligo-Biosciences/artios · weights Ligo-Biosciences/ArtiosBackbone @ 3b8939e3 `280000_ema_0.9999.pt` · WebGPU port ramithuh/artios-webgpu)',
  description:
    'ArtiosBackbone (136.7M parameters, 16 AdaLN DiT blocks × 16 heads) is an all-atom EDM diffusion module for de novo backbone generation and motif + ligand scaffolding. Atom-level conformer features (CCD ideal coordinates, element, charge, atom-name characters) and token-level motif conditioning (restype, motif centre coordinates, binned motif pair distances, burial depth, ligand RASA) drive a Karras reverse-diffusion sampler with Kabsch-aligned self-conditioning. Connect a structure to scaffold a motif and keep its HETATM ligand; leave it unconnected for unconditional design.',
  inputs: {
    structure: { type: 'StructureArtifact', required: false },
  },
  outputs: {
    structure: { type: 'StructureArtifact' },
  },
  paramSchema: [
    {
      key: 'precision',
      label: 'Weight Precision',
      kind: 'select',
      tier: 'primary',
      defaultValue: 'int8',
      options: [
        { value: 'int8', label: 'INT8 (default · ~131 MiB, 6 shards)' },
        { value: 'fp16', label: 'FP16 (~261 MiB, 11 shards)' },
        { value: 'fp32', label: 'FP32 (reference · ~521 MiB, 23 shards)' },
      ],
      description:
        'Storage precision of the sharded safetensors bundle streamed from Hugging Face. Accumulation is always FP32; INT8 is symmetric per-output-row with FP32 scales.',
    },
    {
      key: 'length',
      label: 'Length',
      kind: 'integer',
      tier: 'primary',
      defaultValue: 100,
      min: 30,
      max: 512,
      step: 1,
      unit: 'aa',
      description:
        'Total designed chain length in residues (scaffold + any placed motif residues). Ligand atoms from a connected structure add their own tokens on top.',
    },
    {
      key: 'numSteps',
      label: 'Diffusion steps',
      kind: 'integer',
      tier: 'primary',
      defaultValue: 200,
      min: 20,
      max: 400,
      step: 10,
      upstreamDefaultHint: 'Upstream EDM default: 200',
      warnBelow: {
        threshold: 100,
        message:
          'ArtiosBackbone uses the upstream Karras (rho=7) inference schedule with 200 steps; below ~100 steps the all-atom geometry degrades (chain breaks and clashes).',
        fixValue: 200,
      },
      description:
        'Reverse-diffusion steps of the EDM Karras schedule (sigma 160 -> 4e-4, rho = 7). 200 matches the upstream sampler.',
    },
    {
      key: 'numDesigns',
      label: 'Designs',
      kind: 'integer',
      tier: 'primary',
      defaultValue: 1,
      min: 1,
      max: 16,
      step: 1,
      description:
        'Number of independent designs to sample. Designs share one conditioning cache and are batched on WebGPU.',
    },
    {
      key: 'motifContig',
      label: 'Motif residues (optional)',
      kind: 'text',
      tier: 'primary',
      defaultValue: '',
      placeholder: 'e.g. A10-25 or A10-25, A40-55 (with a structure connected)',
      description:
        'Residue spans of the connected structure to scaffold. Motif residues keep their identity, all-atom coordinates and native numbering inside the designed chain; everything else is generated. Empty = unconditional de novo design.',
    },
    {
      key: 'unindexedMotif',
      label: 'Unindexed motif',
      kind: 'boolean',
      tier: 'advanced',
      defaultValue: true,
      description:
        'Upstream ArtiosBackbone places each motif residue on its own unindexed chain and stitches it onto the nearest scaffold residue post-diffusion. Disable for fixed native-numbered placement.',
    },
    {
      key: 'includeLigand',
      label: 'Keep ligand',
      kind: 'boolean',
      tier: 'advanced',
      defaultValue: true,
      description:
        'Tokenize non-water HETATM groups of the connected structure atom-by-atom and condition on their coordinates through the CCD reference-conformer featurizer (ideal conformers fetched from the RCSB CCD when not bundled).',
    },
    {
      key: 'ligandRasa',
      label: 'Ligand RASA target',
      kind: 'number',
      tier: 'advanced',
      defaultValue: -1,
      min: -1,
      max: 1,
      step: 0.05,
      description:
        'Target relative solvent accessibility of the ligand (0 = fully buried, 1 = fully exposed), binned at 0.05. Use -1 to leave it unconditioned (null bin).',
    },
    {
      key: 'motifDepth',
      label: 'Motif depth target',
      kind: 'number',
      tier: 'advanced',
      defaultValue: -1,
      min: -1,
      max: 2,
      step: 0.05,
      description:
        'Target motif burial depth ||CoM(motif) - CoM(scaffold)|| / Rg, binned at 0.05 over [0, 2). Use -1 to leave it unconditioned (null bin).',
    },
    {
      key: 'stepScaleEta',
      label: 'Step scale (eta)',
      kind: 'number',
      tier: 'advanced',
      defaultValue: 1.5,
      min: 0.5,
      max: 3,
      step: 0.1,
      description: 'EDM sampler Euler step multiplier (upstream default 1.5).',
    },
    {
      key: 'noiseScaleLambda',
      label: 'Noise scale (lambda)',
      kind: 'number',
      tier: 'advanced',
      defaultValue: 1.003,
      min: 0.5,
      max: 1.5,
      step: 0.001,
      description: 'Churn noise multiplier applied to the injected per-step noise (upstream default 1.003).',
    },
    {
      key: 'gamma0',
      label: 'Churn (gamma0)',
      kind: 'number',
      tier: 'advanced',
      defaultValue: 0.8,
      min: 0,
      max: 2,
      step: 0.05,
      description:
        'Noise inflation applied while sigma > 1 (t_hat = sigma * (1 + gamma0)); 0 makes the sampler a deterministic Euler integrator (upstream default 0.8).',
    },
    {
      key: 'selfConditioning',
      label: 'Self-conditioning',
      kind: 'boolean',
      tier: 'advanced',
      defaultValue: true,
      description:
        'Feed the previous denoised prediction back into the network, Kabsch-aligned onto the current noisy coordinates (upstream `align_pred_to_true`).',
    },
    {
      key: 'seed',
      label: 'Seed',
      kind: 'integer',
      tier: 'advanced',
      defaultValue: 42,
      min: 0,
      max: 2_147_483_647,
      step: 1,
      description: 'Base seed; design #n uses `seed + n`.',
    },
  ],
  crossValidate: (p, ctx) => {
    const errs: string[] = [];
    const contig = String(p.motifContig ?? '').trim();
    if (contig) {
      const tokens = contig.split(/[,;\s]+/).filter(Boolean);
      const invalid = tokens.filter((tok) => {
        const m = /^(?:([A-Za-z0-9]):?)?(-?\d+)(?:-(-?\d+))?$/.exec(tok);
        if (!m) return true;
        const lo = Number(m[2]);
        const hi = m[3] !== undefined ? Number(m[3]) : lo;
        return !Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo;
      });
      if (tokens.length === 0 || invalid.length > 0) {
        errs.push(
          `Invalid ArtiosBackbone motifContig "${contig}" (expected residue spans like A10-25 or A10-25, A40-55).`,
        );
      }
      if (ctx?.connectedInputs && !ctx.connectedInputs.has('structure')) {
        errs.push(
          'ArtiosBackbone motifContig requires a StructureArtifact on the structure input (connect a PDB Input block or leave motifContig empty for unconditional design).',
        );
      }
    }
    const rasa = Number(p.ligandRasa ?? -1);
    if (Number.isFinite(rasa) && rasa < 0 && rasa !== -1) {
      errs.push(`ArtiosBackbone ligandRasa must be -1 (unconditioned) or within [0, 1], got ${rasa}.`);
    }
    const depth = Number(p.motifDepth ?? -1);
    if (Number.isFinite(depth) && depth < 0 && depth !== -1) {
      errs.push(`ArtiosBackbone motifDepth must be -1 (unconditioned) or within [0, 2], got ${depth}.`);
    }
    return errs;
  },
};
