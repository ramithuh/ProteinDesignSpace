// PDB writers for poly-glycine backbones (N/CA/C per residue).
// Output strings are directly loadable by Mol*.

const ATOM_NAMES = ['N', 'CA', 'C'] as const;

function pad(s: string, n: number): string {
  return s.length >= n ? s : ' '.repeat(n - s.length) + s;
}
function padR(s: string, n: number): string {
  return s.length >= n ? s : s + ' '.repeat(n - s.length);
}

/**
 * Convert backbone coords [nResidues*3, 3] flat (atom-major: N, CA, C per residue)
 * into a single-model poly-GLY PDB string. Mirrors coords_to_poly_gly_pdb in
 * tools/exports/legacy-teacher/sample_onnx.py.
 */
export function coordsToPolyGlyPdb(coords: Float32Array, nResidues: number): string {
  const lines: string[] = [];
  let atomIdx = 1;
  for (let r = 0; r < nResidues; r++) {
    for (let a = 0; a < 3; a++) {
      const flatIdx = r * 3 + a;
      const x = coords[flatIdx * 3 + 0];
      const y = coords[flatIdx * 3 + 1];
      const z = coords[flatIdx * 3 + 2];
      const name = ATOM_NAMES[a];
      const element = name[0];
      lines.push(
        `ATOM  ${pad(String(atomIdx), 5)}  ${padR(name, 3)} GLY A${pad(String(r + 1), 4)}    ` +
          `${pad(x.toFixed(3), 8)}${pad(y.toFixed(3), 8)}${pad(z.toFixed(3), 8)}` +
          `  1.00  0.00           ${element}  `,
      );
      atomIdx++;
    }
  }
  lines.push('END');
  return lines.join('\n') + '\n';
}

/**
 * Build a multi-model PDB from a trajectory of per-frame coord buffers.
 *
 * Layout: MODEL 1 is the FINAL frame (so Mol* infers chain topology from a
 * well-folded structure, not from pure noise). MODELs 2..T+1 are the
 * trajectory frames 0..T-1 in order. The user-facing slider should map index
 * i ∈ [0, T-1] to modelIndex i+1, skipping MODEL 1.
 */
export function coordsToMultiModelPdb(
  trajectory: Float32Array[],
  nResidues: number,
): string {
  const chunks: string[] = [];
  const terSerial = nResidues * 3 + 1;
  const terLine = `TER   ${pad(String(terSerial), 5)}      GLY A${pad(String(nResidues), 4)}`;
  const T = trajectory.length;
  // [final, 0, 1, 2, ..., T-1]
  const order: number[] = [T - 1];
  for (let i = 0; i < T; i++) order.push(i);

  let modelIdx = 1;
  for (const src of order) {
    const body = coordsToPolyGlyPdb(trajectory[src], nResidues)
      .replace(/END\s*$/, '')
      .replace(/\s+$/, '');
    chunks.push(`MODEL     ${modelIdx}`);
    chunks.push(body);
    chunks.push(terLine);
    chunks.push('ENDMDL');
    modelIdx++;
  }
  chunks.push('END');
  return chunks.join('\n') + '\n';
}
