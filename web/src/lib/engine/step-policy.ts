// Single-source-of-truth per-model sampling step operating envelope for denovo.run.
// Pure TypeScript with ZERO Svelte, DOM, or registry imports so Web Workers,
// model-cards.ts, param-schema.ts, flow-dsl.ts, and server routes can all import it safely.

export interface ModeStepRule {
  /** Mode values this rule applies to (omit or empty = fallback / all modes). */
  modes?: readonly string[];
  /** Human label for the mode scope (e.g. "SDE (default)", "ODE", "Full diffusion", "Partial diffusion", "All modes"). */
  modeLabel?: string;
  /** Hard minimum safe step count enforced in workers, /api/run-node, and ?flow= DSL. */
  minSafeSteps: number;
  /** Recommended minimum step count (warn in UI & Model Card below this threshold). */
  recommendedMinSteps: number;
  /** Upstream default step count. */
  upstreamDefault: number;
  /** Maximum supported step count. */
  maxSteps: number;
  /** Concise scientific rationale shown in Model Card, UI warning, and worker error. */
  rationale: string;
}

export interface ModelSamplingEnvelope {
  adapterId: string;
  displayName: string;
  paramKey: 'steps' | 'nsteps' | 'numSteps' | 'diffusionSteps';
  paramLabel: string;
  /** Mode param key used to match ModeStepRule.modes (e.g. 'mode', 'taskMode'). */
  modeParamKey?: string;
  /** Default mode value when params[modeParamKey] is omitted. */
  defaultMode?: string;
  /** Ordered rules: first matching mode rule wins; last rule should have no `modes` (catch-all). */
  rules: readonly ModeStepRule[];
}

export interface StepValidationResult {
  ok: boolean;
  level: 'ok' | 'warn' | 'error';
  adapterId: string;
  paramKey: string;
  requested: number;
  activeMode: string | null;
  rule: ModeStepRule;
  envelope: ModelSamplingEnvelope;
  message?: string;
}

export const MODEL_STEP_POLICIES: Readonly<Record<string, ModelSamplingEnvelope>> = {
  'la-proteina': {
    adapterId: 'la-proteina',
    displayName: 'La-Proteina',
    paramKey: 'steps',
    paramLabel: 'Sampling steps',
    modeParamKey: 'mode',
    defaultMode: 'sde',
    rules: [
      {
        modes: ['sde'],
        modeLabel: 'SDE (default)',
        minSafeSteps: 200,
        recommendedMinSteps: 350,
        upstreamDefault: 400,
        maxSteps: 400,
        rationale:
          'Euler-Maruyama SDE sampling fractures Cα backbones below 200 steps; 350–400 steps match upstream stability. Switch to ODE mode for fast low-step runs.',
      },
      {
        modeLabel: 'ODE',
        minSafeSteps: 50,
        recommendedMinSteps: 100,
        upstreamDefault: 400,
        maxSteps: 400,
        rationale:
          'Deterministic ODE velocity integration stays geometrically coherent down to 50 steps; ≥100 steps recommended.',
      },
    ],
  },

  protpardelle: {
    adapterId: 'protpardelle',
    displayName: 'ProtPardelle-1c',
    paramKey: 'numSteps',
    paramLabel: 'Diffusion steps',
    modeParamKey: 'taskMode',
    defaultMode: 'unconditional',
    rules: [
      {
        modes: ['partial'],
        modeLabel: 'Partial diffusion',
        minSafeSteps: 50,
        recommendedMinSteps: 200,
        upstreamDefault: 200,
        maxSteps: 500,
        rationale:
          'Partial diffusion only executes the final rewindSteps of the schedule; keep ≥50 total schedule steps.',
      },
      {
        modeLabel: 'Full diffusion',
        minSafeSteps: 200,
        recommendedMinSteps: 200,
        upstreamDefault: 200,
        maxSteps: 500,
        rationale:
          'ProtPardelle EDM reverse diffusion requires ≥200 steps to avoid fractured backbones and steric clashes (upstream default: 200).',
      },
    ],
  },

  boltzgen: {
    adapterId: 'boltzgen',
    displayName: 'BoltzGen-1',
    paramKey: 'numSteps',
    paramLabel: 'Diffusion steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 200,
        recommendedMinSteps: 200,
        upstreamDefault: 200,
        maxSteps: 500,
        rationale:
          'BoltzGen-1 EDM/AF3 diffusion requires ≥200 steps for coherent backbone geometry (upstream default: 200; design.yaml: 500).',
      },
    ],
  },

  'rfd4-proteina': {
    adapterId: 'rfd4-proteina',
    displayName: 'RFD4-Proteina',
    paramKey: 'nsteps',
    paramLabel: 'Steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 50,
        recommendedMinSteps: 58,
        upstreamDefault: 60,
        maxSteps: 120,
        rationale:
          'RFD4-Proteina flow matching converges reliably at ≥58 steps (upstream default: 60); below 50 steps frequently collapses sequence/backbone quality. Use Recycle=None for faster runs.',
      },
    ],
  },

  proteina: {
    adapterId: 'proteina',
    displayName: 'Proteina v1.2 (200M)',
    paramKey: 'steps',
    paramLabel: 'Flow steps',
    modeParamKey: 'mode',
    defaultMode: 'sc',
    rules: [
      {
        modes: ['sc'],
        modeLabel: 'SDE (sc, default)',
        minSafeSteps: 50,
        recommendedMinSteps: 350,
        upstreamDefault: 400,
        maxSteps: 400,
        rationale:
          'Proteina stochastic (SDE) mode stays structurally stable down to 50 steps, though 350–400 steps match upstream designability.',
      },
      {
        modeLabel: 'ODE (vf)',
        minSafeSteps: 50,
        recommendedMinSteps: 100,
        upstreamDefault: 400,
        maxSteps: 400,
        rationale:
          'Deterministic velocity-field (ODE) mode; ≥100 steps recommended for full quality.',
      },
    ],
  },

  genie2: {
    adapterId: 'genie2',
    displayName: 'Genie 2',
    paramKey: 'steps',
    paramLabel: 'Diffusion steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 1000,
        recommendedMinSteps: 1000,
        upstreamDefault: 1000,
        maxSteps: 1000,
        rationale:
          'Genie 2 uses a fixed 1000-step cosine diffusion schedule; fewer steps are unsupported and yield distorted backbones (use Genie 3 or SiD-Protein for fast runs).',
      },
    ],
  },

  rfd3: {
    adapterId: 'rfd3',
    displayName: 'RFdiffusion3',
    paramKey: 'steps',
    paramLabel: 'Diffusion steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 20,
        recommendedMinSteps: 50,
        upstreamDefault: 200,
        maxSteps: 200,
        rationale:
          'RFdiffusion3 browser default is 50 steps (upstream: 200); fewer than 20 steps degrade backbone geometry.',
      },
    ],
  },

  rfdiffusion: {
    adapterId: 'rfdiffusion',
    displayName: 'RFdiffusion (Base)',
    paramKey: 'numSteps',
    paramLabel: 'Diffusion steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 20,
        recommendedMinSteps: 50,
        upstreamDefault: 50,
        maxSteps: 50,
        rationale:
          'RFdiffusion full reverse schedule is 50 steps; fewer than 20 steps degrade backbone coherence.',
      },
    ],
  },

  frameflow: {
    adapterId: 'frameflow',
    displayName: 'FrameFlow',
    paramKey: 'steps',
    paramLabel: 'Flow steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 20,
        recommendedMinSteps: 50,
        upstreamDefault: 100,
        maxSteps: 200,
        rationale:
          'FrameFlow SE(3) Euler integration needs ≥20 steps (and dt*expRate ≤ 1.0 in exponential mode); 100 steps match upstream.',
      },
    ],
  },

  foldflow2: {
    adapterId: 'foldflow2',
    displayName: 'FoldFlow 2',
    paramKey: 'steps',
    paramLabel: 'Flow steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 20,
        recommendedMinSteps: 50,
        upstreamDefault: 50,
        maxSteps: 100,
        rationale:
          'FoldFlow 2 SE(3) flow matching defaults to 50 steps; fewer than 20 steps degrade backbone geometry.',
      },
    ],
  },

  'sid-protein': {
    adapterId: 'sid-protein',
    displayName: 'SiD-Protein',
    paramKey: 'steps',
    paramLabel: 'Distilled steps',
    rules: [
      {
        modeLabel: 'All modes',
        minSafeSteps: 1,
        recommendedMinSteps: 4,
        upstreamDefault: 16,
        maxSteps: 32,
        rationale:
          'SiD-Protein is a distilled few-step sampler (1–32 steps supported; 16 matches upstream, ≥4 recommended for best geometry).',
      },
    ],
  },
};

export function getModelStepPolicy(adapterId: string): ModelSamplingEnvelope | null {
  if (!Object.prototype.hasOwnProperty.call(MODEL_STEP_POLICIES, adapterId)) {
    return null;
  }
  return MODEL_STEP_POLICIES[adapterId] ?? null;
}

export function resolveStepRule(
  adapterId: string,
  params?: Record<string, unknown>,
): { envelope: ModelSamplingEnvelope; rule: ModeStepRule; activeMode: string | null } | null {
  const envelope = getModelStepPolicy(adapterId);
  if (!envelope || envelope.rules.length === 0) return null;

  let activeMode: string | null = null;
  if (envelope.modeParamKey) {
    const rawMode = params?.[envelope.modeParamKey];
    const modeStr =
      typeof rawMode === 'string' || typeof rawMode === 'number' || typeof rawMode === 'boolean'
        ? String(rawMode).trim().toLowerCase()
        : '';
    activeMode = modeStr !== '' ? modeStr : (envelope.defaultMode ?? null);
  }

  for (const rule of envelope.rules) {
    if (!rule.modes || rule.modes.length === 0) {
      return { envelope, rule, activeMode };
    }
    if (activeMode !== null && rule.modes.some((m) => m.toLowerCase() === activeMode)) {
      return { envelope, rule, activeMode };
    }
  }

  return { envelope, rule: envelope.rules[envelope.rules.length - 1], activeMode };
}

function isOmittedStepValue(raw: unknown): boolean {
  return raw === undefined || raw === null || (typeof raw === 'string' && raw.trim() === '');
}

export function validateStepCount(
  adapterId: string,
  params?: Record<string, unknown>,
): StepValidationResult | null {
  const resolved = resolveStepRule(adapterId, params);
  if (!resolved) return null;

  const { envelope, rule, activeMode } = resolved;
  const raw = params?.[envelope.paramKey];
  const modeTag = activeMode ? ` (${activeMode} mode)` : '';

  if (isOmittedStepValue(raw)) {
    return {
      ok: true,
      level: 'ok',
      adapterId: envelope.adapterId,
      paramKey: envelope.paramKey,
      requested: rule.upstreamDefault,
      activeMode,
      rule,
      envelope,
    };
  }

  const requested =
    typeof raw === 'number' || typeof raw === 'string' ? Number(raw) : Number.NaN;

  if (!Number.isFinite(requested)) {
    return {
      ok: false,
      level: 'error',
      adapterId: envelope.adapterId,
      paramKey: envelope.paramKey,
      requested,
      activeMode,
      rule,
      envelope,
      message: `${envelope.displayName}${modeTag}: ${envelope.paramKey} must be a valid finite number (${rule.minSafeSteps}–${rule.maxSteps}).`,
    };
  }

  if (requested < rule.minSafeSteps) {
    return {
      ok: false,
      level: 'error',
      adapterId: envelope.adapterId,
      paramKey: envelope.paramKey,
      requested,
      activeMode,
      rule,
      envelope,
      message: `${envelope.displayName}${modeTag}: ${envelope.paramKey}=${requested} is below the safe minimum (${rule.minSafeSteps}). ${rule.rationale}`,
    };
  }

  if (requested > rule.maxSteps) {
    return {
      ok: false,
      level: 'error',
      adapterId: envelope.adapterId,
      paramKey: envelope.paramKey,
      requested,
      activeMode,
      rule,
      envelope,
      message: `${envelope.displayName}${modeTag}: ${envelope.paramKey}=${requested} exceeds the supported maximum (${rule.maxSteps}).`,
    };
  }

  if (requested < rule.recommendedMinSteps) {
    return {
      ok: true,
      level: 'warn',
      adapterId: envelope.adapterId,
      paramKey: envelope.paramKey,
      requested,
      activeMode,
      rule,
      envelope,
      message: `${envelope.displayName}${modeTag}: ${envelope.paramKey}=${requested} is below the recommended minimum (${rule.recommendedMinSteps}). ${rule.rationale}`,
    };
  }

  return {
    ok: true,
    level: 'ok',
    adapterId: envelope.adapterId,
    paramKey: envelope.paramKey,
    requested,
    activeMode,
    rule,
    envelope,
  };
}

export function assertSafeStepCount(
  adapterId: string,
  params?: Record<string, unknown>,
  fallbackDefault?: number,
): number {
  const resolved = resolveStepRule(adapterId, params);
  if (!resolved) {
    return fallbackDefault ?? 0;
  }

  const { envelope, rule } = resolved;
  const raw = params?.[envelope.paramKey];

  if (isOmittedStepValue(raw)) {
    const effective = fallbackDefault ?? rule.upstreamDefault;
    const check = validateStepCount(adapterId, { ...(params ?? {}), [envelope.paramKey]: effective });
    if (check && !check.ok) {
      throw new Error(check.message);
    }
    return Math.round(effective);
  }

  const check = validateStepCount(adapterId, params);
  if (check && !check.ok) {
    throw new Error(check.message);
  }
  return Math.round(Number(raw));
}

export function formatLowStepGeometryWarning(
  adapterId: string,
  params: Record<string, unknown> | undefined,
  geometry: { caBreaks?: number; clashes?: number; minCaDist?: number } | undefined,
): string | null {
  if (!geometry) return null;

  const caBreaks = geometry.caBreaks ?? 0;
  const clashes = geometry.clashes ?? 0;
  const severeCompression =
    typeof geometry.minCaDist === 'number' &&
    Number.isFinite(geometry.minCaDist) &&
    geometry.minCaDist < 2.0;

  if (caBreaks <= 0 && clashes <= 0 && !severeCompression) {
    return null;
  }

  const resolved = resolveStepRule(adapterId, params);
  if (!resolved) return null;

  const { envelope, rule, activeMode } = resolved;
  const raw = params?.[envelope.paramKey];
  if (isOmittedStepValue(raw)) return null;

  const requested = typeof raw === 'number' || typeof raw === 'string' ? Number(raw) : Number.NaN;
  if (!Number.isFinite(requested)) return null;

  if (requested >= rule.recommendedMinSteps && requested >= rule.upstreamDefault) {
    return null;
  }

  const defects: string[] = [];
  if (caBreaks > 0) {
    defects.push(`${caBreaks} Cα break${caBreaks === 1 ? '' : 's'}`);
  }
  if (clashes > 0) {
    defects.push(`${clashes} steric clash${clashes === 1 ? '' : 'es'}`);
  } else if (severeCompression) {
    defects.push(`min Cα–Cα ${geometry.minCaDist!.toFixed(2)} Å`);
  }

  const modeTag = activeMode ? ` (${activeMode} mode)` : '';
  const target =
    requested < rule.recommendedMinSteps
      ? `recommended ≥${rule.recommendedMinSteps} (upstream default: ${rule.upstreamDefault})`
      : `upstream default (${rule.upstreamDefault})`;

  return `${envelope.displayName}${modeTag} generated ${defects.join(' and ')} at ${envelope.paramKey}=${requested} (below Model Card ${target}). Increase ${envelope.paramKey} for coherent backbone geometry.`;
}
