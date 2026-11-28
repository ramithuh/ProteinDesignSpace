// Pure-TS, edge-safe generator for `/llms.txt`.
// STRICT IMPORT RULE: Relative imports ONLY (./registry, ./benchmarks, ./flow-dsl, and `import type` from ./types).
// ZERO $lib/ runtime imports, ZERO Svelte imports, ZERO Node/DOM globals at top level.

import { BENCHMARK_TARGETS, type BenchmarkTargetMeta } from './benchmarks';
import {
  ADAPTER_REGISTRY,
  ESMFOLD2_STEPS,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_VARIANT_OPTIONS,
  WORKFLOW_PRESETS,
} from './registry';
import {
  ADAPTER_TO_SHORT,
  FLOW_MODEL_ALIASES,
  LOCALFOLD_MODEL_ALIASES,
  encodeGraphToFlow,
} from './flow-dsl';
import type { AdapterManifest, AdapterParamField, PortSpec, PortType } from './types';

export interface GenerateLlmsTxtOptions {
  /** Base origin for generated URLs and endpoint docs (default: 'https://denovo.run'). */
  origin?: string;
  /** Include curated `?flow=` URLs for all built-in WORKFLOW_PRESETS (default: true). */
  includePresets?: boolean;
  /** Include advanced-tier parameter schemas (seed, nsteps, fixedAtoms, etc.) per adapter (default: true). */
  includeAdvancedParams?: boolean;
}

const PORT_TYPES: readonly PortType[] = [
  'StructureArtifact',
  'TrajectoryFrames',
  'SequenceSet',
  'FoldedStructure',
  'DesignabilityMetrics',
  'MutationMatrix',
  'ResidueTrack',
  'SequenceEmbedding',
  'ConditionSpec',
];

const HOST_GPU_ALLOWED_ADAPTERS: readonly string[] = [
  'web-protein-mpnn',
  'localfold-af2',
  'rfd4-proteina',
  'genie3',
  'thermompnn',
  'ligandmpnn',
  'lasermpnn',
  'artios-potts',
  'antifold-esmif1',
  'protpardelle',
  'rfdiffusion',
  'rfd3',
  'boltzgen',
  'artios-backbone',
  'evodiff',
  'la-proteina',
  'proteina',
  'genie2',
  'frameflow',
  'foldflow2',
  'sid-protein',
  'esmc',
  'dyna1',
  'gnina',
  'cg2all',
];

function formatPorts(ports: Record<string, PortSpec>): string {
  const entries = Object.entries(ports);
  if (entries.length === 0) return 'none';
  return entries
    .map(([name, spec]) => {
      const tags: string[] = [];
      if (spec.required === false) tags.push('optional');
      if (spec.multiple) tags.push('multi');
      return tags.length > 0 ? `${name}: ${spec.type} (${tags.join(', ')})` : `${name}: ${spec.type}`;
    })
    .join(', ');
}

function formatDefaultValue(val: unknown): string {
  if (val === undefined) return 'undefined';
  if (val === null) return 'null';
  if (typeof val === 'string') {
    if (val.length === 0) return '""';
    if (val.length > 32) return JSON.stringify(val.slice(0, 29) + '...');
    return JSON.stringify(val);
  }
  if (typeof val === 'number' || typeof val === 'boolean') return String(val);
  return JSON.stringify(val);
}

function formatParamField(field: AdapterParamField): string {
  const parts: string[] = [field.kind];
  if (field.defaultValue !== undefined && field.defaultValue !== '') {
    parts.push(`default=${formatDefaultValue(field.defaultValue)}`);
  }
  if (field.options && field.options.length > 0) {
    parts.push(`options=${field.options.map((o) => o.value).join('|')}`);
  }
  return `\`${field.key}\` (${parts.join(', ')})`;
}

function formatAdapterSection(
  manifest: AdapterManifest,
  aliasesByAdapter: Map<string, string[]>,
  includeAdvancedParams: boolean,
): string[] {
  const shortCode = ADAPTER_TO_SHORT[manifest.id] ?? manifest.id;
  const aliases = aliasesByAdapter.get(manifest.id) ?? [];
  const aliasText = aliases.length > 0 ? aliases.map((a) => `\`${a}\``).join(', ') : 'none';

  const lines: string[] = [
    `### \`${manifest.id}\` (${manifest.displayName})`,
    `- Short code: \`${shortCode}\` | Stage: \`${manifest.stage}\` | Runtime: \`${manifest.runtime}\` | Aliases: ${aliasText}`,
    `- Inputs: ${formatPorts(manifest.inputs)} | Outputs: ${formatPorts(manifest.outputs)}`,
  ];

  if (manifest.modes && manifest.modes.length > 0) {
    const modeKey = manifest.modeParamKey ?? 'mode';
    const modeList = manifest.modes.map((m) => `\`${m.id}\``).join(', ');
    lines.push(`- Modes (\`${modeKey}\`): ${modeList}`);
  }

  const primaryFields = (manifest.paramSchema ?? []).filter((f) => f.tier === 'primary');
  if (primaryFields.length > 0) {
    lines.push(`- Primary params: ${primaryFields.map(formatParamField).join(', ')}`);
  }

  if (includeAdvancedParams) {
    const advancedFields = (manifest.paramSchema ?? []).filter((f) => f.tier === 'advanced');
    if (advancedFields.length > 0) {
      lines.push(`- Advanced params: ${advancedFields.map(formatParamField).join(', ')}`);
    }
  }

  return lines;
}

function formatBenchmarkLine(b: BenchmarkTargetMeta): string {
  const cp = b.conditionParams;
  const extras: string[] = [];
  if (b.cropSegments) extras.push(`cropSegments=\`${b.cropSegments}\``);
  extras.push(`contig=\`${cp.contig}\``);
  if (cp.motifSpan) extras.push(`motifSpan=\`${cp.motifSpan}\``);
  if (cp.unindexedMotif) extras.push('unindexedMotif=true');
  if (cp.motifAtoms) extras.push(`motifAtoms=\`${cp.motifAtoms}\``);
  if (cp.lockChain) extras.push(`lockChain=\`${cp.lockChain}\``);
  if (cp.hotspots) extras.push(`hotspots=\`${cp.hotspots}\``);
  if (cp.hbondDonorSelection) extras.push(`hbondDonor=\`${cp.hbondDonorSelection}\``);
  if (cp.hbondAcceptorSelection) extras.push(`hbondAcceptor=\`${cp.hbondAcceptorSelection}\``);
  return `- \`${b.id}\` (${b.shortLabel}, category=\`${b.category}\`, pdbId=\`${b.pdbId}\`, ~${b.approxTokens} tok): ${extras.join(', ')}`;
}

/**
 * Generate deterministic `/llms.txt` documentation directly from live
 * `ADAPTER_REGISTRY`, `BENCHMARK_TARGETS`, `ADAPTER_TO_SHORT`, `FLOW_MODEL_ALIASES`,
 * `LOCALFOLD_MODEL_OPTIONS`, `MPNN_VARIANT_OPTIONS`, and `WORKFLOW_PRESETS`.
 */
export function generateLlmsTxt(options: GenerateLlmsTxtOptions = {}): string {
  const origin = (options.origin || 'https://denovo.run').replace(/\/+$/, '');
  const includePresets = options.includePresets !== false;
  const includeAdvancedParams = options.includeAdvancedParams !== false;

  const localFoldIds = new Set(LOCALFOLD_MODEL_OPTIONS.map((opt) => opt.id.toLowerCase()));
  const aliasesByAdapter = new Map<string, string[]>();
  for (const [alias, res] of Object.entries(FLOW_MODEL_ALIASES)) {
    if (res.adapterId === 'localfold-af2' && localFoldIds.has(alias.toLowerCase())) {
      continue;
    }
    const list = aliasesByAdapter.get(res.adapterId) ?? [];
    list.push(alias);
    aliasesByAdapter.set(res.adapterId, list);
  }

  const localFoldAliasesByModel = new Map<string, string[]>();
  for (const [alias, modelId] of Object.entries(LOCALFOLD_MODEL_ALIASES)) {
    const list = localFoldAliasesByModel.get(modelId) ?? [];
    list.push(alias);
    localFoldAliasesByModel.set(modelId, list);
  }

  const manifests = Object.values(ADAPTER_REGISTRY);

  const sections: string[] = [];

  sections.push(
    [
      '# denovo.run',
      '',
      '> Browser-native WebGPU workspace for de novo protein design, inverse folding, structure prediction, ligand docking, and designability evaluation.',
      '',
      `For LLMs without browser tools: construct runnable workflow DAGs directly as Markdown-safe \`?flow=<dsl>\` URLs on \`${origin}\` (e.g. \`${origin}/?flow=r3:length=80~m~f\`). Every link opens and validates the full workflow graph client-side with zero server state.`,
    ].join('\n'),
  );

  sections.push(
    [
      '## URL Syntax (?flow=)',
      '',
      'Primary query format: `?flow=<dsl>` (`#flow=` is also accepted by the client).',
      '- Edge operator: `~` connects nodes left-to-right (`a~b~c`).',
      '- Node syntax: `model`, `model:k=v,k2=v2`, `id:model`, `id:model:k=v`, or `id:model,k=v` (where `model` is any canonical adapter ID, short code, or alias).',
      '- Parameter separator: `,` separates `key=val` pairs; commas inside structured values such as `contig=10-20,A16-35,10-20`, `hotspots=A54,A56`, `cropSegments=A102,A104,A129,A132,A:ZN,A:DHZ`, `motifSpan`, `fixedAtoms`, `motifAtoms`, and `fixedPositions` are preserved automatically.',
      '- Branch / statement separator: `;` separates parallel branches or independent chains (`src~r3~cmp;src~r4~cmp`).',
      '- Semicolons inside parameter values (`fixedAtoms`, `motifAtoms`, `tiedPositions`, `hbond*`): `parseFlowDsl` automatically recognizes `<Res>:<Atoms>;<Res>:<Atoms>` and `A1=B1;A2=B2` continuations (both raw `;` and percent-encoded `%3B`, e.g. `fixedAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD`), or use `|` (`%7C`). For arbitrary free-text values containing `;`, wrap in URL-encoded double quotes (`%22...%3B...%22`).',
      '- Explicit port wiring: `.` selects ports (`from.port~to.port`, e.g. `src.structure~cmp:compare.reference`).',
      '- Special parameters: `_label=` sets the node card display label (avoid unencoded spaces/quotes, e.g. `_label=Scaffold_80`, or percent-encode spaces as `%20`); `label=` / `seqLabel=` on `sequence-input` sets FASTA sequence label; `model=` (`foldModel=` / `checkpoint=`) on `localfold-af2` selects checkpoint; `steps=` is accepted as a universal step-count alias across all structure generators; `maxSequences=` infers `capMode=fixed`; `pdbId=` / `uniprotId=` infers `pdb-input` `mode`; generator params (`hotspots`, `lockChain`, `motifSpan`, `motifAtoms`, `unindex`, `fixedAtoms`, `contig`, `cathCode`, `partialFraction`, `partialT`, `symmetry`, `presetId`) infer task mode.',
      '- Markdown autolinker safety: emit ONLY `[A-Za-z0-9\\-._~:;=,]` and `%XX` percent-encoding (note: `/0` binder chain breaks also parse raw, or encode `/` as `%2F` for strict charset safety; never use raw `+` for spaces or separators in query strings). Never emit unencoded `(`, `)`, `[`, `]`, `<`, `>`, `{`, `}`, `"`, `\'`, whitespace, `~~`, or trailing `[.,;:!?_~-]`.',
    ].join('\n'),
  );

  sections.push(
    [
      '## Cross-Model Fair Comparison & Universal Parameter Aliases',
      '',
      '- **Universal step-count alias (`steps=`)**: `flow-dsl` maps `steps=<int>` to each generator\'s native step field (`rfd3`/`la-proteina`/`proteina`/`sid-protein`/`genie2`/`frameflow`/`foldflow2`: `steps`; `rfd4-proteina`: `nsteps`; `boltzgen`/`protpardelle`/`rfdiffusion`: `numSteps`; `genie3`: `nSampleStep`). Native defaults differ (`rfd3` `steps=50` vs `rfd4-proteina` `nsteps=60` with `earlyStop=true`), so for head-to-head benchmarks explicitly pin `steps=50,seed=42` on `rfd3` and `nsteps=50,earlyStop=false,seed=42` (or `steps=50,earlyStop=false,seed=42`) on `rfd4-proteina`.',
      '- **Reproducibility & batching**: Every structure generator supports `seed=<int>` (default `42`) and `numDesigns=<int>` (`1..8` or `1..16`); inverse folders (`web-protein-mpnn`, `ligandmpnn`, `lasermpnn`, `antifold-esmif1`) use `numSequences=<int>`, `temperature=<float>`, and `seed=<int>`.',
      '- **Symmetric RFD3 vs RFD4 conditioning**:',
      '  - *Indexed motif*: `rfd3:mode=motif,contig=25,A163-181,31` <-> `rfd4:mode=motif,contig=25-25,A163-181,31-31,motifSpan=A163-181`.',
      '  - *Unindexed catalytic / tip-atom theozyme (e.g. 1CTT)*: `rfd3:mode=motif,length=180,unindex=A102,A104,A129,A132,fixedAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD%3BA129:SG,CB%3BA132:SG,CB,CA` <-> `rfd4:mode=motif,length=180,motifSpan=A102,A104,A129,A132,unindexedMotif=true,motifAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD%3BA129:SG,CB%3BA132:SG,CB,CA`.',
      '  - *PPI hotspot binder (e.g. 5O45 PD-L1)*: `rfd3:mode=binder,length=70,target=A,hotspots=A54,A56,A66,A115` <-> `rfd4:mode=binder,length=70,lockChain=A,hotspots=A54,A56,A66,A115`.',
    ].join('\n'),
  );

  sections.push(
    [
      '## Auto-Wiring, Port Semantics & Designability Gates',
      '',
      `- Port types (${PORT_TYPES.length}): ${PORT_TYPES.map((t) => `\`${t}\``).join(', ')}.`,
      '- `designability-scorer` (`s` / `scorer`): requires both `structure` (`StructureArtifact`) and `folds` (`FoldedStructure`). When `folds` is wired (e.g. `r3~m~f~s`), upstream BFS lineage automatically wires the ancestor generator or `pdb-input` `.structure` into `designability-scorer.structure`. Standard pass gate: `scRmsd < 2.0` Å (`scRmsdThreshold=2.0`) AND `plddt >= 70` (0-100 scale, `plddtThreshold=70`), plus `motifRmsd < 1.0` Å on motif/theozyme tasks and `ipsae` / `iptm` on binder complexes.',
      '- `localfold-af2` (`f`): requires `sequences` (`SequenceSet`) and accepts optional `template` (`StructureArtifact`, e.g. `seq~af2;pdb:pdbId=1UBQ~af2`). Connecting a structure generator directly to `af2` without `sequences` wires `template` and fails validation until `sequences` (e.g. via `~m~` / `~lm~`) is connected.',
      '- `genie2` (`g2`): optional input port is `template` (`StructureArtifact`), auto-wired from `pdb-input.structure`. Connecting any Cα-only generator (`genie3`, `genie2`, `proteina`, `sid-protein`) to `web-protein-mpnn` (`m`) automatically infers `family=ca` unless `family=` is explicitly set.',
      '- `structure-compare` (`c` / `compare`): fan-in port `structures` (`StructureArtifact`, `multiple: true`, max 6) plus optional `reference` (`StructureArtifact`). Auto-wiring prefers `reference` for `pdb-input` when free, or use explicit `src.structure~cmp:compare.reference`.',
      '- `esmc` & `dyna1`: both `sequences` and `structure` inputs are optional in the manifest, but at least one of `sequences` or `structure` must be connected.',
      '- Parallel branches: reuse explicit node IDs (`src~r3~cmp;src~r4~cmp`) or repeat bare model names (`rfd3~mpnn~af2;rfd4~mpnn~af2` auto-allocates `mpnn_2`, `af2_2`).',
    ].join('\n'),
  );

  sections.push(
    [
      '## Fallback Formats (?recipe=, ?p=, ?g=) & Stateless Edge API (/api/link, /api/og)',
      '',
      '- `?recipe=<urlencoded-JSON-or-YAML>` (`#recipe=` also accepted on client):',
      '  - JSON shape: `{"id":"my-flow","name":"My Flow","nodes":[{"id":"src","adapterId":"pdb-input","label":"1CTT","params":{"mode":"pdb-id","pdbId":"1CTT"}},{"id":"r3","adapterId":"rfd3","params":{"mode":"motif","length":180,"steps":50}}],"edges":[{"fromNode":"src","fromPort":"structure","toNode":"r3","toPort":"structure"}]}`.',
      '  - YAML shape (`denovo: v1`): header `denovo: v1` + optional `name: "..."`, `nodes:` section with `nodeId: { "model": "<adapterId>", "label": "...", ...params }`, and `wires:` section with `- fromNode.fromPort -> toNode.toPort`.',
      '- `?p=<presetId>`: preset ID with optional `&<nodeId>.<param>=<value>` overrides; `?g=<base64url>`: compressed DAG payload (`#p=` / `#g=` preserved).',
      `- \`GET ${origin}/api/link?flow=...\` (also accepts \`recipe\`, \`p\`, \`g\`, or \`url\`) and \`POST ${origin}/api/link\` (JSON \`{ flow?, recipe?, url?, graph? }\` or \`text/plain\`): validates statelessly and returns \`{ ok, url, shortUrl, nodes, edges, errors }\` where \`shortUrl\` uses compact \`?g=\` or \`?p=\` encoding.`,
      `- \`GET ${origin}/api/og?flow=...\` (or \`?recipe=...\`): renders a dynamic 1200x630 PNG workflow preview card. Discovery is advertised via HTML \`<link rel="llms" type="text/plain" href="/llms.txt" />\`, HTTP \`Link: </llms.txt>; rel="llms"\`, HTML body summary, and \`/robots.txt\`.`,
    ].join('\n'),
  );

  sections.push(
    [
      '## Optional Self-Hosted / Headless Host-GPU Runner API',
      '',
      'For automated multi-seed sweeps outside an interactive browser, `server/selfhost-server.mjs` (default `http://127.0.0.1:8790`, optional `Authorization: Bearer <DENOVO_SELFHOST_TOKEN>`) executes any GPU node headlessly via Chromium/Vulkan WebGPU:',
      '- `GET /api/health` -> `{ ok, service: "denovo-selfhost", gpu, activeAdapter, queueDepth, busy, maxFoldResidues, authRequired, buildSha, headSha, staleBuild }` (returns 404 JSON on public static `denovo.run`).',
      '- `POST /api/run-node` -> JSON body `{ "node": { "id": "r3_1", "adapterId": "rfd3", "label": "RFD3", "params": { ... } }, "inputs": { "<portName>": <Artifact> } }`. Streams `text/event-stream` SSE events: `event: started` (`{ jobId, cancelToken }`), `event: progress` (`{ status, phase, progress, detail }`), and terminal `event: result` (`{ outputs: { "<portName>": <Artifact> } }`) or `event: error` (`{ error }`).',
      '- `POST /api/cancel` (`{ jobId, cancelToken }`) cancels an active/queued job; `POST /api/release-gpu` (`{}`) evicts idle GPU worker VRAM.',
      `- Allowed GPU \`adapterId\`s (${HOST_GPU_ALLOWED_ADAPTERS.length}): ${HOST_GPU_ALLOWED_ADAPTERS.map((id) => `\`${id}\``).join(', ')}. Source/CPU nodes (\`pdb-input\`, \`sequence-input\`, \`top-k-filter\`, \`designability-scorer\`, \`structure-compare\`, \`pulchra\`) construct, rebuild or score plain JSON \`StructureArtifact\` / \`SequenceSet\` / \`FoldedStructure\` objects directly.`,
      '- `remote-codec` binary/float envelopes in `inputs` & `outputs`: TypedArrays serialize as `{"__denovoTypedArray":"Float32Array"|"Float64Array"|"Uint8Array"|"Int32Array"|"Uint16Array"|"Uint32Array","b64":"<base64>"}` and non-finite numbers as `{"__denovoNum":"NaN"|"Infinity"|"-Infinity"}`.',
    ].join('\n'),
  );

  const catalogLines: string[] = [
    `## Live Model Catalog (ADAPTER_REGISTRY — ${manifests.length} Models)`,
    '',
  ];
  for (const manifest of manifests) {
    catalogLines.push(...formatAdapterSection(manifest, aliasesByAdapter, includeAdvancedParams));
    catalogLines.push('');
  }
  sections.push(catalogLines.join('\n').trimEnd());

  const localFoldLines: string[] = [
    `## LocalFold Checkpoints (${LOCALFOLD_MODEL_OPTIONS.length} Models) & Aliases`,
    '',
    `Use as \`localfold-af2:model=<id>\` or directly as a node model name (e.g. \`~rf3\`, \`~boltz2\`, \`~af3\`, \`~ef2-fast-300m\`). \`ESMFOLD2_STEPS\`: ${ESMFOLD2_STEPS.join(', ')}.`,
    '',
  ];
  for (const opt of LOCALFOLD_MODEL_OPTIONS) {
    const aliases = localFoldAliasesByModel.get(opt.id) ?? [];
    const aliasSuffix = aliases.length > 0 ? ` | aliases: ${aliases.map((a) => `\`${a}\``).join(', ')}` : '';
    localFoldLines.push(
      `- \`${opt.id}\` (${opt.shortLabel}, family: \`${opt.family}\`, ~${opt.approxMB} MB, ${opt.license})${aliasSuffix}`,
    );
  }
  sections.push(localFoldLines.join('\n'));

  const mpnnLines: string[] = [
    `## ProteinMPNN Variants (${MPNN_VARIANT_OPTIONS.length} Variants)`,
    '',
    'Use with `web-protein-mpnn:family=<id>` (`m:family=...`) or aliases `mpnn`, `solublempnn`, `proteinmpnn`, `campnn`, `ca-mpnn`:',
    '',
  ];
  for (const v of MPNN_VARIANT_OPTIONS) {
    mpnnLines.push(`- \`${v.id}\` (${v.label}, ~${v.approxMB} MB, ${v.license}): ${v.description}`);
  }
  sections.push(mpnnLines.join('\n'));

  const benchLines: string[] = [
    `## Curated Benchmark Targets (BENCHMARK_TARGETS — ${BENCHMARK_TARGETS.length} Targets)`,
    '',
    'Use with `pdb:pdbId=<PDB>,benchmarkId=<id>,cropSegments=<crop>` (or pass the target\'s `contig` / `motifSpan` / `motifAtoms` / `hotspots` directly to `rfd3`, `rfd4`, `genie3`, `rfdiffusion`, `protpardelle`, or `boltzgen`):',
    '',
  ];
  for (const b of BENCHMARK_TARGETS) {
    benchLines.push(formatBenchmarkLine(b));
  }
  sections.push(benchLines.join('\n'));

  sections.push(
    [
      '## Curated Copy-Paste ?flow= Examples',
      '',
      '### Linear Chains',
      `- RFD3 -> ProteinMPNN -> AF2: ${origin}/?flow=r3:length=80~m~f`,
      `- PDB 1PRW -> ThermoMPNN ddG scan: ${origin}/?flow=pdb:pdbId=1PRW~thermompnn`,
      `- Sequence FASTA -> RoseTTAFold3: ${origin}/?flow=q:sequence=MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALE~rf3`,
      `- PDB 1UBQ -> GNINA 1.3 scoring: ${origin}/?flow=pdb:pdbId=1UBQ~gnina:taskMode=score`,
      '',
      '### Multi-Branch & Fair Head-to-Head Comparisons',
      `- RFD3 vs RFD4 Unconditional Paired (50 steps, seed 42) -> MPNN -> AF2 -> scRMSD: ${origin}/?flow=r3:length=80,steps=50,seed=42~m1:m~f1:af2~s1:s;r4:rfd4:length=80,steps=50,earlyStop=false,seed=42~m2:m~f2:af2~s2:s`,
      `- RFD3 vs RFD4 on 1CTT Catalytic Zn/DHZ Theozyme (tip-atom %3B map, 50 steps, seed 42): ${origin}/?flow=src:pdb:pdbId=1CTT,cropSegments=A102,A104,A129,A132,A:ZN,A:DHZ~r3:length=180,steps=50,seed=42,unindex=A102,A104,A129,A132,fixedAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD%3BA129:SG,CB%3BA132:SG,CB,CA~lm1:ligandmpnn~f1:af3~s1:s;src~r4:rfd4:length=180,steps=50,earlyStop=false,seed=42,motifSpan=A102,A104,A129,A132,unindexedMotif=true,motifAtoms=A102:ND1,CG,CE1%3BA104:OE1,CD%3BA129:SG,CB%3BA132:SG,CB,CA~lm2:ligandmpnn~f2:af3~s2:s`,
      `- RFD3 vs RFD4 on 1UBQ -> 3D Compare (structures[]): ${origin}/?flow=src:pdb:pdbId=1UBQ~rfd3:length=80~compare;src~rfd4:length=80~compare`,
      `- RFD4 Fork (AlphaFold2 vs RoseTTAFold3): ${origin}/?flow=gen:rfd4:length=65~f_af2:af2~s_af2:scorer;gen~f_rf3:rf3~s_rf3:scorer`,
      '',
      '### Fan-Out / Fan-In & Designability Scorer',
      `- RFD3 -> MPNN (4 seqs) -> AF2 -> scRMSD/scTM: ${origin}/?flow=r3:length=70~m:numSequences=4~f~s`,
      `- Genie 3 -> Cα-MPNN -> Top-K Filter -> AF2 -> scRMSD/scTM: ${origin}/?flow=g3:length=80~m:numSequences=4~tk:topK=2~f~s`,
      '',
      '### Custom Node Labels (_label=)',
      `- Labeled RFD3 -> MPNN -> RF3 -> Scorer: ${origin}/?flow=r3:length=80,_label=Scaffold_80~m~rf3:_label=RF3_Refold~s`,
      '',
      '### Explicit from.port~to.port & structure-compare.reference',
      `- 1PRW -> LASErMPNN (packed structure + seqs) -> RF3 -> Scorer + 3D Compare vs native reference: ${origin}/?flow=src:pdb:pdbId=1PRW~pack:lasermpnn:numSequences=2~fold:rf3~score:scorer;src.structure~cmp:compare.reference;pack.structure~cmp;fold.structure~cmp`,
      '',
      '### Genie2 Template, Binder (target-only MSA), & Antibody Scoring',
      `- Genie 2 (template port + auto Cα-MPNN): ${origin}/?flow=pdb:pdbId=1UBQ~genie2:length=90~mpnn~af2~scorer`,
      `- PD-L1 (5O45) PPI binder with hotspots + target-only MSA: ${origin}/?flow=src:pdb:pdbId=5O45~rfd4:mode=binder,length=75,lockChain=A,hotspots=A54,A56~m~af2:msaMode=target-only~s`,
      `- Antibody (1IGT) AntiFold CDRs + ESM-C 300M + Dyna-1: ${origin}/?flow=src:pdb:pdbId=1IGT~af:variant=antifold~ec:mode=wildtype;src~dy:inputMode=struct-only`,
    ].join('\n'),
  );

  if (includePresets) {
    const presetLines: string[] = [
      `## Built-In Workflow Presets (${WORKFLOW_PRESETS.length} Presets)`,
      '',
    ];
    for (const preset of WORKFLOW_PRESETS) {
      const encoded = encodeGraphToFlow(preset.graph);
      presetLines.push(`- **${preset.id}** (${preset.name}): \`?p=${preset.id}\` | ${origin}/?${encoded.query}`);
    }
    sections.push(presetLines.join('\n'));
  }

  return sections.join('\n\n') + '\n';
}
