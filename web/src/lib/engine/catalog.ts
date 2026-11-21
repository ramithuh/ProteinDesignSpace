// Pure-TypeScript catalog grouping, multi-variant flyout sub-items, and search filtering
// for the `/` and `+ Add Block` workflow block picker.

import {
  ADAPTER_REGISTRY,
  LOCALFOLD_MODEL_OPTIONS,
  MPNN_VARIANT_OPTIONS,
  RFD4_BENCHMARK_OPTIONS,
} from './registry';
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
  { title: 'Conditioning', ids: ['rfd4-conditioner'] },
  { title: 'Structure Generation', ids: ['rfd4-proteina', 'genie3'] },
  { title: 'Sequence Design', ids: ['web-protein-mpnn'] },
  { title: 'Structure Prediction', ids: ['localfold-af2'] },
  { title: 'Evaluation', ids: ['designability-scorer'] },
];

export const CONCISE_BLOCK_ROLES: Record<string, string> = {
  'pdb-input': 'Built-in 3-helix bundle or upload/paste PDB & mmCIF',
  'sequence-input': 'Paste amino acid sequences (plain text or FASTA)',
  'rfd4-conditioner': 'Binding hotspots, fixed motifs, H-bonds & CATH folds',
  'rfd4-proteina': 'All-atom backbone and sequence co-design',
  genie3: 'Fast de novo backbone diffusion (23.3M parameters)',
  'web-protein-mpnn': 'Inverse folding (SolubleMPNN, ProteinMPNN, Cα-only)',
  'localfold-af2': 'Predict 3D structures (AlphaFold2, RF3, Boltz-2, AlphaFold 3 & more)',
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
        label: `LocalFold (${opt.shortLabel})`,
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

    const af3Items: CatalogSubItem[] = LOCALFOLD_MODEL_OPTIONS.filter((m) => m.family === 'af3').map((opt) => ({
      key: opt.id,
      label: opt.label,
      meta: `~${opt.approxMB} MB · ${opt.license}`,
      badge: opt.nonCommercial ? 'Non-Commercial' : undefined,
      nonCommercial: opt.nonCommercial,
      insert: {
        adapterId: 'localfold-af2',
        params: { model: opt.id },
        label: `LocalFold (${opt.shortLabel})`,
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

  if (adapterId === 'rfd4-conditioner') {
    const customItem: CatalogSubItem = {
      key: 'custom',
      label: 'Custom Conditioning Builder',
      subtitle: 'Define custom binding hotspots, structural constraints & fold topology',
      insert: {
        adapterId: 'rfd4-conditioner',
        params: { taskPreset: 'custom' },
        label: 'RFD4 Condition',
      },
      searchBlob: 'custom conditioning builder hotspots cath hbond h-bond',
    };

    const presetItems: CatalogSubItem[] = RFD4_BENCHMARK_OPTIONS.map((opt) => ({
      key: opt.id,
      label: opt.label,
      badge: opt.category,
      meta: opt.needsEncoder ? 'Requires structure encoder (~138 MB)' : undefined,
      insert: {
        adapterId: 'rfd4-conditioner',
        params: { taskPreset: opt.id },
        label: `Condition: ${opt.category}`,
      },
      searchBlob: [
        opt.id,
        opt.label,
        opt.category,
        opt.needsEncoder ? 'encoder' : '',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase(),
    }));

    return {
      subCountLabel: `${RFD4_BENCHMARK_OPTIONS.length} presets`,
      subGroups: [
        { title: 'Custom', items: [customItem] },
        { title: 'Task Presets', items: presetItems },
      ],
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
