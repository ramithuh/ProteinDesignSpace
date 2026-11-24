export interface ViewerSpinInput {
  /** User's spin toggle preference (kept unmutated across pipeline runs). */
  spin: boolean;
  /** Whether at least one structure is loaded in the viewport. */
  hasAnyStructure: boolean;
  /** Whether side-by-side / grid canonical front camera lock is active. */
  canonicalCamera: boolean;
  /** External run-pause signal (e.g. WebGPU workflow run active). */
  pauseSpin: boolean;
}

export interface ViewerSpinState {
  /** True only when Mol* trackball spin animation should actively rotate the camera. */
  effectiveSpin: boolean;
  /**
   * True when 16-sample SSAO occlusion should remain disabled:
   * - while actively spinning (`effectiveSpin`), OR
   * - while temporarily paused for GPU compute (`pauseSpin`), preventing a sudden
   *   16-sample SSAO shader/render burst when compute starts or while dragging mid-run.
   */
  disableOcclusion: boolean;
  /** True when user spin is enabled for a non-canonical loaded structure (even if run-paused). */
  spinConfigured: boolean;
}

export function resolveViewerSpinState(input: ViewerSpinInput): ViewerSpinState {
  const spinConfigured = Boolean(input.spin && input.hasAnyStructure && !input.canonicalCamera);
  const effectiveSpin = Boolean(spinConfigured && !input.pauseSpin);
  const disableOcclusion = Boolean(effectiveSpin || input.pauseSpin);
  return { effectiveSpin, disableOcclusion, spinConfigured };
}
