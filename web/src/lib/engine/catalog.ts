// Pure-TypeScript catalog grouping, multi-variant flyout sub-items, and search filtering
// for the `/` and `+ Add Block` workflow block picker.

import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_VARIANT_OPTIONS,
} from './registry';
import { BENCHMARK_TARGETS } from './benchmarks';
import { PORT_COLORS } from './graph-utils';

export interface CatalogInsertRequest {
  adapterId: string;
  params?: Record<string, unknown>;
  label?: string;
}

export interface CatalogSubItem {
  key: string;
  label: string;
  subtitle?: string;
  meta?: string;
  badge?: string;
  nonCommercial?: boolean;
  insert: CatalogInsertRequest;
  searchBlob: string;
}

export interface CatalogSubGroup {
  title: string;
  items: CatalogSubItem[];
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export interface CatalogItem {
  id: string;
  group: string;
  shortName: string;
  displayName: string;
  stage: string;
  role: string;
  subCountLabel?: string;
  subGroups: CatalogSubGroup[];
  identityBlob: string;
  ownSearchBlob: string;
  searchBlob: string;
}

export interface FilteredCatalogEntry {
  item: CatalogItem;
  directParentMatch: boolean;
  matchingSubGroups: CatalogSubGroup[];
  matchingSubItems: CatalogSubItem[];
  hasNarrowedSubMatch: boolean;
}

export const CATALOG_STAGE_GROUPS: { title: string; ids: string[] }[] = [
  { title: 'Inputs', ids: ['pdb-input', 'sequence-input'] },
  { title: 'Structure Generation', ids: ['rfd4-proteina', 'genie3'] },
  { title: 'Sequence Design', ids: ['web-protein-mpnn'] },
  { title: 'Structure Prediction', ids: ['localfold-af2'] },
  { title: 'Evaluation', ids: ['top-k-filter', 'designability-scorer'] },
];

export const CONCISE_BLOCK_ROLES: Record<string, string> = {
  'pdb-input': 'Fetch by PDB ID, upload/paste PDB & mmCIF, or 3-helix demo',
  'sequence-input': 'Paste amino acid sequences (plain text or FASTA)',
  'rfd4-proteina': 'All-atom backbone and sequence co-design',
  genie3: 'Fast de novo backbone diffusion (23.3M parameters)',
  'web-protein-mpnn': 'Inverse folding (SolubleMPNN, ProteinMPNN, Cα-only)',
  'localfold-af2': 'Predict 3D structures (AlphaFold2, RF3, Boltz-2, AlphaFold 3 & more)',
  'top-k-filter': 'Rank and filter top-K sequences per backbone or globally before folding',
  'designability-scorer': 'Kabsch alignment, scRMSD, scTM & pLDDT pass rate',
};

function buildSubGroupsForAdapter(adapterId: string): {
  subCountLabel?: string;
  subGroups: CatalogSubGroup[];
} {
  if (adapterId === 'localfold-af2') {
    const af2Items: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af2').map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: opt.shortLabel,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.shortLabel,
        opt.license,
        'af2',
        'af2-class',
        'alphafold2',
        'monomer',
      ]
        .join(' ')
        .toLowerCase(),
    }));

    const af2MultimerItems: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter(
      (m) => m.family === 'af2-multimer',
    ).map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: opt.shortLabel,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.shortLabel,
        opt.license,
        'af2',
        'af2-multimer',
        'alphafold2',
        'multimer',
        'complex',
        'binder',
        'ppi',
      ]
        .join(' ')
        .toLowerCase(),
    }));

    const af3Items: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3').map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      badge: opt.nonCommercial ? 'Non-Commercial' : undefined,
      nonCommercial: opt.nonCommercial,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: opt.shortLabel,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.shortLabel,
        opt.license,
        'af3',
        'af3-class',
        'alphafold3',
        'all-atom',
        opt.nonCommercial ? 'non-commercial' : '',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase(),
    }));

    return {
      subCountLabel: `${LOCALFOLD_MODEL_OPTIONS.length} models`,
      subGroups: [
        { title: 'AF2-Class (Monomer)', items: af2Items },
        { title: 'AF2-Multimer v3 (Complexes)', items: af2MultimerItems },
        { title: 'AF3-Class (All-Atom)', items: af3Items },
      ],
    };
  }

  if (adapterId === 'web-protein-mpnn') {
    const items: CatalogSubItem[] = MPNN_VARIANT_OPTIONS.map((v) => ({
      key: v.id,
      label: v.label,
      subtitle: v.description,
      meta: `~${v.approxMB} MB · ${v.license}`,
      insert: {
        adapterId: 'web-protein-mpnn',
        params: { family: v.family, modelName: v.modelName },
        label: v.shortLabel,
      },
      searchBlob: [
        v.id,
        v.family,
        v.label,
        v.shortLabel,
        v.description,
        v.license,
        v.family === 'ca' ? 'ca alpha carbon backbone trace' : '',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase(),
    }));

    return {
      subCountLabel: `${items.length} variants`,
      subGroups: [{ title: 'Model Variants', items }],
    };
  }

  return { subGroups: [] };
}

export const CATALOG_ITEMS: CatalogItem[] = CATALOG_STAGE_GROUPS.flatMap((grp) =>
  grp.ids
    .map((id): CatalogItem | null => {
      const m = ADAPTER_REGISTRY[id];
      if (!m) return null;

      const portTokens = [
        ...Object.values(m.inputs),
        ...Object.values(m.outputs),
      ].flatMap((spec) => {
        const pc = PORT_COLORS[spec.type];
        return pc ? [pc.label, pc.short] : [];
      });

      const role = CONCISE_BLOCK_ROLES[id] ?? m.description;
      const { subCountLabel, subGroups } = buildSubGroupsForAdapter(id);

      const identityBlob = [
        m.shortName,
        m.displayName.replace(/\s*\([^)]*\)/g, ''),
        m.stage,
        grp.title,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      const ownSearchBlob = [
        m.id,
        m.displayName,
        m.shortName,
        m.stage,
        grp.title,
        m.description,
        role,
        ...portTokens,
      ]
        .join(' ')
        .toLowerCase();

      const subText = subGroups.flatMap((sg) => [sg.title, ...sg.items.map((s) => s.searchBlob)]).join(' ');
      const searchBlob = `${ownSearchBlob} ${subText}`.toLowerCase();

      return {
        id,
        group: grp.title,
        shortName: m.shortName,
        displayName: m.displayName,
        stage: m.stage,
        role,
        subCountLabel,
        subGroups,
        identityBlob,
        ownSearchBlob,
        searchBlob,
      };
    })
    .filter((x): x is CatalogItem => x !== null),
);

export interface CatalogCategory {
  id: string;
  title: string;
  stage: string;
  subtitle: string;
  countLabel: string;
  footnote?: string;
  subGroups: CatalogSubGroup[];
  identityBlob: string;
  searchBlob: string;
}

export interface FilteredCategoryEntry {
  category: CatalogCategory;
  matchingSubGroups: CatalogSubGroup[];
  matchingSubItems: CatalogSubItem[];
  hasNarrowedSubMatch: boolean;
}

export const CATALOG_CATEGORIES: CatalogCategory[] = [
  (() => {
    const pdb = ADAPTER_REGISTRY['pdb-input'];
    const seq = ADAPTER_REGISTRY['sequence-input'];
    const items: CatalogSubItem[] = [
      {
        key: 'pdb-input',
        label: pdb.shortName,
        subtitle: CONCISE_BLOCK_ROLES['pdb-input'],
        insert: { adapterId: 'pdb-input' },
        searchBlob: `${pdb.id} ${pdb.shortName} ${pdb.displayName} ${CONCISE_BLOCK_ROLES['pdb-input']} pdb mmcif structure upload template input`.toLowerCase(),
      },
      {
        key: 'sequence-input',
        label: seq.shortName,
        subtitle: CONCISE_BLOCK_ROLES['sequence-input'],
        insert: { adapterId: 'sequence-input' },
        searchBlob: `${seq.id} ${seq.shortName} ${seq.displayName} ${CONCISE_BLOCK_ROLES['sequence-input']} fasta amino acid sequence input`.toLowerCase(),
      },
    ];
    const makeBenchmarkItem = (b: (typeof BENCHMARK_TARGETS)[number], extraTokens: string): CatalogSubItem => ({
      key: `bench-${b.id}`,
      label: b.shortLabel,
      subtitle: b.biology,
      meta: `${b.sourceFamily} · ~${b.approxTokens} tok`,
      badge: b.tier === 'fast' ? 'Fast' : 'Standard',
      insert: {
        adapterId: 'pdb-input',
        params: {
          mode: 'pdb-id',
          pdbId: b.pdbId,
          filename: `${b.pdbId}.cif`,
          pdbText: '',
          benchmarkId: b.id,
          ...(b.cropSegments ? { cropSegments: b.cropSegments } : {}),
        },
        label: `PDB: ${b.name} (${b.pdbId})`,
      },
      searchBlob: `${b.id} ${b.name} ${b.shortLabel} ${b.pdbId} ${b.biology} benchmark ${extraTokens}`.toLowerCase(),
    });

    const binderBenchItems: CatalogSubItem[] = BENCHMARK_TARGETS.filter(
      (b) => b.category === 'binder',
    ).map((b) => makeBenchmarkItem(b, 'ppi binder target'));

    const motifBenchItems: CatalogSubItem[] = BENCHMARK_TARGETS.filter(
      (b) => b.category === 'motif',
    ).map((b) => makeBenchmarkItem(b, 'motif scaffold'));

    const enzymeBenchItems: CatalogSubItem[] = BENCHMARK_TARGETS.filter(
      (b) => b.category === 'enzyme-ligand',
    ).map((b) => makeBenchmarkItem(b, 'enzyme ligand dna cofactor'));

    const subGroups: CatalogSubGroup[] = [
      { title: 'Starting Inputs', items },
      { title: 'PPI Binders', items: binderBenchItems, collapsible: true, defaultCollapsed: true },
      { title: 'Motif Scaffolding', items: motifBenchItems, collapsible: true, defaultCollapsed: true },
      { title: 'Enzymes, Ligands & DNA', items: enzymeBenchItems, collapsible: true, defaultCollapsed: true },
    ];
    const allItems = [...items, ...binderBenchItems, ...motifBenchItems, ...enzymeBenchItems];
    const identityBlob = 'inputs input source pdb sequence fasta mmcif benchmark';
    const searchBlob = `${identityBlob} ${allItems.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'inputs',
      title: 'Inputs',
      stage: 'source',
      subtitle: 'PDB / mmCIF, FASTA & curated benchmark targets',
      countLabel: `${allItems.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const rfd4 = ADAPTER_REGISTRY['rfd4-proteina'];
    const g3 = ADAPTER_REGISTRY.genie3;
    const items: CatalogSubItem[] = [
      {
        key: 'rfd4-proteina',
        label: rfd4.shortName,
        subtitle: CONCISE_BLOCK_ROLES['rfd4-proteina'],
        meta: `~${rfd4.approxWeightsMB} MB`,
        insert: { adapterId: 'rfd4-proteina' },
        searchBlob: `${rfd4.id} ${rfd4.shortName} ${rfd4.displayName} ${CONCISE_BLOCK_ROLES['rfd4-proteina']} rfdiffusion4 rfd4 proteina generate de novo backbone co-design`.toLowerCase(),
      },
      {
        key: 'genie3',
        label: g3.shortName,
        subtitle: CONCISE_BLOCK_ROLES.genie3,
        meta: `~${g3.approxWeightsMB} MB · Apache-2.0`,
        insert: { adapterId: 'genie3' },
        searchBlob: `${g3.id} ${g3.shortName} ${g3.displayName} ${CONCISE_BLOCK_ROLES.genie3} genie genie3 diffusion generate de novo backbone`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [{ title: 'De Novo Generators', items }];
    const identityBlob = 'structure generation generate generator de novo backbone diffusion';
    const searchBlob = `${identityBlob} ${items.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'generation',
      title: 'Structure Generation',
      stage: 'generate',
      subtitle: 'RFD4-Proteina & Genie 3 de novo design',
      countLabel: `${items.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const mpnn = buildSubGroupsForAdapter('web-protein-mpnn');
    const total = mpnn.subGroups.reduce((a, g) => a + g.items.length, 0);
    const identityBlob = 'sequence design inverse fold folding proteinmpnn mpnn';
    const searchBlob = `${identityBlob} ${mpnn.subGroups.flatMap((g) => [g.title.toLowerCase(), ...g.items.map((i) => i.searchBlob)]).join(' ')}`;
    return {
      id: 'sequence-design',
      title: 'Sequence Design',
      stage: 'inverse-fold',
      subtitle: 'SolubleMPNN, ProteinMPNN & Cα-only',
      countLabel: `${total}`,
      footnote: 'Powered by webProteinMPNN (MurrellGroup)',
      subGroups: mpnn.subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const lf = buildSubGroupsForAdapter('localfold-af2');
    const total = lf.subGroups.reduce((a, g) => a + g.items.length, 0);
    const identityBlob = 'structure prediction folding fold localfold predict';
    const searchBlob = `${identityBlob} ${lf.subGroups.flatMap((g) => [g.title.toLowerCase(), ...g.items.map((i) => i.searchBlob)]).join(' ')}`;
    return {
      id: 'prediction',
      title: 'Structure Prediction',
      stage: 'fold',
      subtitle: 'AlphaFold2, RoseTTAFold3, Boltz-2, AlphaFold 3…',
      countLabel: `${total}`,
      footnote: 'Powered by LocalFold (sokrypton/localfold)',
      subGroups: lf.subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
  (() => {
    const sc = ADAPTER_REGISTRY['designability-scorer'];
    const tk = ADAPTER_REGISTRY['top-k-filter'];
    const items: CatalogSubItem[] = [
      {
        key: 'top-k-filter',
        label: tk.displayName,
        subtitle: CONCISE_BLOCK_ROLES['top-k-filter'],
        insert: { adapterId: 'top-k-filter' },
        searchBlob: `${tk.id} ${tk.shortName} ${tk.displayName} ${CONCISE_BLOCK_ROLES['top-k-filter']} topk top-k filter ranker rank sequence batch multi-design`.toLowerCase(),
      },
      {
        key: 'designability-scorer',
        label: sc.shortName,
        subtitle: CONCISE_BLOCK_ROLES['designability-scorer'],
        insert: { adapterId: 'designability-scorer' },
        searchBlob: `${sc.id} ${sc.shortName} ${sc.displayName} ${CONCISE_BLOCK_ROLES['designability-scorer']} score evaluation designability kabsch scrmsd sctm plddt`.toLowerCase(),
      },
    ];
    const subGroups: CatalogSubGroup[] = [{ title: 'Filtering & Designability Metrics', items }];
    const identityBlob = 'evaluation score scoring metrics designability scrmsd sctm filter ranker top-k';
    const searchBlob = `${identityBlob} ${items.map((i) => i.searchBlob).join(' ')}`;
    return {
      id: 'evaluation',
      title: 'Evaluation',
      stage: 'score',
      subtitle: 'Top-K sequence filter, Kabsch scRMSD, scTM & pLDDT pass rate',
      countLabel: `${items.length}`,
      subGroups,
      identityBlob,
      searchBlob,
    };
  })(),
];

export function filterCatalogCategories(
  query: string,
  categories: CatalogCategory[] = CATALOG_CATEGORIES,
): FilteredCategoryEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return categories.map((category) => ({
      category,
      matchingSubGroups: category.subGroups,
      matchingSubItems: category.subGroups.flatMap((g) => g.items),
      hasNarrowedSubMatch: false,
    }));
  }

  const terms = q.split(/\s+/).filter(Boolean);
  const results: FilteredCategoryEntry[] = [];

  for (const category of categories) {
    if (!terms.every((t) => category.searchBlob.includes(t))) continue;
    const totalSubCount = category.subGroups.reduce((a, g) => a + g.items.length, 0);

    // Pure category-level match (e.g. "prediction", "localfold", "generation") -> show all sub-items
    if (terms.every((t) => category.identityBlob.includes(t))) {
      results.push({
        category,
        matchingSubGroups: category.subGroups,
        matchingSubItems: category.subGroups.flatMap((g) => g.items),
        hasNarrowedSubMatch: false,
      });
      continue;
    }

    const subTargetTerms = terms.filter(
      (t) =>
        !category.identityBlob.includes(t) &&
        category.subGroups.some((sg) => {
          const titleL = sg.title.toLowerCase();
          return sg.items.some((sub) => `${titleL} ${sub.searchBlob}`.includes(t));
        }),
    );

    if (subTargetTerms.length > 0) {
      const narrowed: CatalogSubGroup[] = [];
      for (const sg of category.subGroups) {
        const titleL = sg.title.toLowerCase();
        const matched = sg.items.filter((sub) => {
          const hay = `${titleL} ${sub.searchBlob}`;
          return subTargetTerms.every((t) => hay.includes(t));
        });
        if (matched.length > 0) {
          narrowed.push({ title: sg.title, items: matched });
        }
      }
      if (narrowed.length > 0) {
        const matchingSubItems = narrowed.flatMap((g) => g.items);
        results.push({
          category,
          matchingSubGroups: narrowed,
          matchingSubItems,
          hasNarrowedSubMatch: matchingSubItems.length < totalSubCount || totalSubCount === 1,
        });
        continue;
      }
    }

    results.push({
      category,
      matchingSubGroups: category.subGroups,
      matchingSubItems: category.subGroups.flatMap((g) => g.items),
      hasNarrowedSubMatch: false,
    });
  }

  return results;
}

export function filterCatalog(query: string, items: CatalogItem[] = CATALOG_ITEMS): FilteredCatalogEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return items.map((item) => {
      const matchingSubItems = item.subGroups.flatMap((g) => g.items);
      return {
        item,
        directParentMatch: true,
        matchingSubGroups: item.subGroups,
        matchingSubItems,
        hasNarrowedSubMatch: false,
      };
    });
  }

  const terms = q.split(/\s+/).filter(Boolean);
  const results: FilteredCatalogEntry[] = [];

  for (const item of items) {
    if (!terms.every((t) => item.searchBlob.includes(t))) continue;

    const directParentMatch = terms.every((t) => item.ownSearchBlob.includes(t));
    const totalSubCount = item.subGroups.reduce((acc, g) => acc + g.items.length, 0);

    if (item.subGroups.length === 0) {
      results.push({
        item,
        directParentMatch,
        matchingSubGroups: [],
        matchingSubItems: [],
        hasNarrowedSubMatch: false,
      });
      continue;
    }

    // 1. If every query term matches the block's own identity (e.g. "rfd4", "localfold", "proteinmpnn"),
    // expose all sub-groups without mis-narrowing to a coincidental sub-item match.
    if (terms.every((t) => item.identityBlob.includes(t))) {
      const matchingSubItems = item.subGroups.flatMap((g) => g.items);
      results.push({
        item,
        directParentMatch,
        matchingSubGroups: item.subGroups,
        matchingSubItems,
        hasNarrowedSubMatch: false,
      });
      continue;
    }

    // 2. Narrow across parent + sub-item terms (e.g. "localfold boltz", "proteinmpnn soluble", "rfd4 motif", "rfd4 hotspot")
    // before falling back to directParentMatch.
    const subTargetTerms = terms.filter(
      (t) =>
        !item.identityBlob.includes(t) &&
        item.subGroups.some((sg) => {
          const sgTitleLower = sg.title.toLowerCase();
          return sg.items.some((sub) => `${sgTitleLower} ${sub.searchBlob}`.includes(t));
        }),
    );
    const parentOnlyTerms = terms.filter((t) => !subTargetTerms.includes(t));

    if (subTargetTerms.length > 0 && parentOnlyTerms.every((t) => item.ownSearchBlob.includes(t))) {
      const narrowedGroups: CatalogSubGroup[] = [];
      for (const sg of item.subGroups) {
        const sgTitleLower = sg.title.toLowerCase();
        const matched = sg.items.filter((sub) => {
          const hay = `${sgTitleLower} ${sub.searchBlob}`;
          return subTargetTerms.every((t) => hay.includes(t));
        });
        if (matched.length > 0) {
          narrowedGroups.push({ title: sg.title, items: matched });
        }
      }
      if (narrowedGroups.length > 0) {
        const matchingSubItems = narrowedGroups.flatMap((g) => g.items);
        results.push({
          item,
          directParentMatch,
          matchingSubGroups: narrowedGroups,
          matchingSubItems,
          hasNarrowedSubMatch: matchingSubItems.length < totalSubCount,
        });
        continue;
      }
    }

    // 3. Fallback: if parent matches directly (e.g. "prediction" or role keywords), expose all sub-groups
    if (directParentMatch) {
      const matchingSubItems = item.subGroups.flatMap((g) => g.items);
      results.push({
        item,
        directParentMatch,
        matchingSubGroups: item.subGroups,
        matchingSubItems,
        hasNarrowedSubMatch: false,
      });
      continue;
    }

    // 4. Cross-boundary query matched multiple distinct sub-items without one unifying sub-item
    results.push({
      item,
      directParentMatch,
      matchingSubGroups: [],
      matchingSubItems: [],
      hasNarrowedSubMatch: false,
    });
  }

  return results;
}
