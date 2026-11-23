// Layer 1 Core Kernel — Atom-Capable Conditioning IR (Entities, Covalent/Metal Bonds, Restraints, Symmetry).
// Designed for modern all-atom foundation models (AF3, Boltz-2, Chai, RFD3/AtomWorks, BoltzGen, covalent docking).
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

import type { ArtifactEnvelope } from './artifacts';

/**
 * Residue- or atom-level selector addressing any entity in a multi-modal biomolecular composition.
 * Supports whole-chain, residue ranges (`A45-60`), single residues (`A:145`), and specific atoms (`A:145:SG`, `LIG:C12`).
 */
export interface EntitySelector {
  entityId: string;
  chain?: string;
  resSeq?: number;
  resRange?: readonly [number, number];
  insCode?: string;
  atomName?: string;
  atomIndex?: number;
}

export type BondConstraintKind =
  | 'covalent'
  | 'disulfide'
  | 'cyclic-peptide'
  | 'metal-coordination'
  | 'hydrogen-bond'
  | (string & {});

/** Explicit chemical or coordination linkage between two atom/residue selectors (e.g. covalent warhead, cyclization, Zn2+ triad). */
export interface BondConstraint {
  kind: BondConstraintKind;
  from: EntitySelector;
  to: EntitySelector;
  bondOrder?: 1 | 2 | 3 | 1.5;
  targetDistanceAngstrom?: number;
  toleranceAngstrom?: number;
  label?: string;
}

export type GeometricRestraintKind =
  | 'distance'
  | 'contact'
  | 'pocket'
  | 'angle'
  | 'dihedral'
  | 'hotspot'
  | (string & {});

/** Pairwise or multi-atom geometric restraint (catalytic geometry, pocket contact, epitope hotspot). */
export interface GeometricRestraint {
  kind: GeometricRestraintKind;
  selectors: readonly EntitySelector[];
  min?: number;
  max?: number;
  target?: number;
  unit?: 'angstrom' | 'degree' | 'radian';
  weight?: number;
  label?: string;
}

/** Point-group or helical oligomeric symmetry specification (e.g. C3, D2, T, I). */
export interface SymmetryConstraint {
  pointGroup:
    | 'C1'
    | 'C2'
    | 'C3'
    | 'C4'
    | 'C5'
    | 'C6'
    | 'D2'
    | 'D3'
    | 'D4'
    | 'T'
    | 'O'
    | 'I'
    | (string & {});
  copies?: number;
  axis?: readonly [number, number, number];
}

export interface EntityModificationSpec {
  /** 1-based monomer position within the polymer entity. */
  position: number;
  /** Standard CCD / PTM code (e.g. 'SEP', 'TPO', 'PTR', 'MLY', 'NAG'). */
  ccdCode: string;
}

export type BiomolecularEntityKind =
  | 'protein'
  | 'antibody-heavy'
  | 'antibody-light'
  | 'nanobody'
  | 'dna'
  | 'rna'
  | 'ligand-ccd'
  | 'ligand-smiles'
  | 'ligand-sdf'
  | 'ion'
  | 'glycan'
  | 'cyclic-peptide'
  | (string & {});

/** First-class entity declaration in a multi-modal biomolecular complex. */
export interface TypedEntitySpec {
  entityId: string;
  entityKind: BiomolecularEntityKind;
  sequence?: string;
  smiles?: string;
  ccdCode?: string;
  sdfText?: string;
  copyCount?: number;
  cyclic?: boolean;
  modifications?: readonly EntityModificationSpec[];
}

export interface ConditionIntent<TPayload = Record<string, unknown>> {
  domainId: string;
  intentKind: string;
  payload: TPayload;
  summary?: string;
  /** Optional multi-entity complex composition (proteins, nucleic acids, ligands, ions, glycans, PTMs). */
  entities?: readonly TypedEntitySpec[];
  /** Optional atom- or residue-level covalent, cyclic, disulfide, or metal-coordination bonds. */
  bonds?: readonly BondConstraint[];
  /** Optional distance, contact, pocket, angle, or hotspot restraints. */
  restraints?: readonly GeometricRestraint[];
  /** Optional oligomeric point-group symmetry. */
  symmetry?: SymmetryConstraint;
  /** Optional fixed/locked residue or atom selectors. */
  fixedSelectors?: readonly EntitySelector[];
}

export interface ConditionTargetCompiler<
  TIntent extends ConditionIntent = ConditionIntent,
  TCompiled = unknown,
> {
  readonly id: string;
  readonly domainId: string;
  readonly targetPluginId: string;
  compile: (intent: TIntent, ctx?: { templateArtifact?: ArtifactEnvelope }) => TCompiled;
}
