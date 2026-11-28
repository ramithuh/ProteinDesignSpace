// PULCHRA 3.06 TypeScript port: table decoding, C-reference parity, reconstruction behaviour,
// plugin runner (collections + sequence threading) and native plugin registration.
//
// Golden digests/coordinates below were generated from the native PULCHRA 3.06 binary
// (`cc -O3 pulchra.c pulchra_data.c -lm`) on the exact PDB text produced by `tracePdb()`.
// "classic" goldens come from the same source with `_PRESERVE = 0` (3.06 hard-codes
// `_PRESERVE = 1`, which disables its Cα optimiser); digests hash every printed
// coordinate (milli-Å), so they pin the reference to its last printed digit.
// Inputs: Cα traces of 1L2Y (model 1), 1UBQ, 4INS chains A/B, 7RSA 88-98, 2LZM 15-27.

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { fnv1a } from '$lib/kernel';
import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import { ADAPTER_TO_SHORT, SHORT_TO_ADAPTER, parseFlowDsl } from '$lib/engine/flow-dsl';
import { CATALOG_ITEMS } from '$lib/engine/catalog';
import { defaultParamsForAdapter } from '$lib/engine/graph-utils';
import { getModelCardForAdapter } from '$lib/engine/model-cards';
import type { SequenceSet, StructureArtifact, WorkflowNode } from '$lib/engine/types';
import { createDefaultPluginRegistry } from '$lib/plugins';
import { pulchraPlugin } from '$lib/plugins/pulchra';
import { pulchraOptionsFromParams, runPulchraCpu } from '$lib/plugins/pulchra/cpu-runner';
import {
  PULCHRA_TABLES_URL,
  decodePulchraTables,
  loadPulchraTables,
  reconstructPdb,
  type PulchraOptions,
} from '$lib/plugins/pulchra/pulchra-core';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { parsePdbBackbone, parseStructureText } from '$lib/protein-ops/pdb';
import { formatNodeParamSummary } from '$lib/ui/node-display-helpers';

// ─── Fixtures (RES num x y z; '; '-separated) and C-reference goldens ───────

const TRP_CAGE_CA = `
ASN 1 -8.608 3.135 -1.618; LEU 2 -4.923 4.002 -2.452; TYR 3 -3.690 2.738 0.981; ILE 4 -5.857 -0.449 0.613
GLN 5 -4.122 -1.167 -2.743; TRP 6 -0.716 -0.631 -0.993; LEU 7 -1.641 -2.932 1.963; LYS 8 -3.024 -5.791 -0.269
ASP 9 0.466 -6.016 -1.905; GLY 10 2.060 -6.618 1.593; GLY 11 2.626 -2.967 2.723; PRO 12 6.333 -2.533 3.806
SER 13 7.049 -6.179 2.704; SER 14 6.389 -5.315 -1.015; GLY 15 9.451 -3.116 -1.870; ARG 16 7.289 0.084 -2.054
PRO 17 6.782 3.088 0.345; PRO 18 3.287 4.031 1.686; PRO 19 1.185 6.543 -0.353; SER 20 0.852 10.027 1.285
`;
const UBIQUITIN_CA = `
MET 1 26.266 25.413 2.842; GLN 2 26.850 29.021 3.898; ILE 3 26.235 30.058 7.497; PHE 4 26.772 33.436 9.197
VAL 5 28.605 33.965 12.503; LYS 6 27.691 37.315 14.143; THR 7 30.225 38.643 16.662; LEU 8 29.607 41.180 19.467
THR 9 31.422 43.940 17.553; GLY 10 28.978 43.960 14.678; LYS 11 31.191 42.012 12.331; THR 12 29.542 39.020 10.653
ILE 13 31.720 36.289 9.176; THR 14 30.505 33.884 6.512; LEU 15 31.677 30.275 6.639; GLU 16 31.220 27.341 4.275
VAL 17 30.288 24.245 6.193; GLU 18 28.468 20.940 5.980; PRO 19 25.829 19.825 8.494; SER 20 28.054 16.835 9.210
ASP 21 30.796 19.083 10.566; THR 22 31.398 19.064 14.286; ILE 23 31.288 22.201 16.417; GLU 24 35.031 21.722 17.069
ASN 25 35.590 21.945 13.302; VAL 26 33.533 25.097 12.978; LYS 27 35.596 26.715 15.736; ALA 28 38.794 25.761 13.880
LYS 29 37.471 27.391 10.668; ILE 30 36.731 30.570 12.645; GLN 31 40.269 30.508 14.115; ASP 32 41.718 30.022 10.643
LYS 33 39.808 32.994 9.233; GLU 34 39.676 35.547 12.072; GLY 35 42.345 34.269 14.431; ILE 36 40.226 33.716 17.509
PRO 37 41.461 30.751 19.594; PRO 38 38.817 28.020 19.889; ASP 39 39.063 28.063 23.695; GLN 40 37.738 31.637 23.712
GLN 41 34.738 30.875 21.473; ARG 42 31.200 30.329 22.780; LEU 43 28.762 29.573 19.906; ILE 44 25.034 30.170 20.401
PHE 45 22.126 29.062 18.183; ALA 46 18.443 29.143 19.083; GLY 47 19.399 29.894 22.655; LYS 48 21.550 26.796 23.133
GLN 49 25.349 26.872 23.643; LEU 50 26.826 24.521 21.012; GLU 51 29.015 21.657 22.288; ASP 52 32.262 20.670 20.514
GLY 53 31.568 16.962 19.825; ARG 54 28.108 17.439 18.276; THR 55 27.574 18.192 14.563; LEU 56 25.594 21.109 13.072
SER 57 22.924 18.583 12.025; ASP 58 22.418 17.638 15.693; TYR 59 21.079 21.149 16.251; ASN 60 19.065 21.352 12.999
ILE 61 21.184 24.263 11.690; GLN 62 20.081 24.773 8.033; LYS 63 21.656 26.847 5.240; GLU 64 21.907 30.563 5.881
SER 65 21.419 30.253 9.620; THR 66 23.212 32.762 11.891; LEU 67 25.149 31.609 14.980; HIS 68 26.179 34.127 17.650
LEU 69 29.801 34.145 18.829; VAL 70 30.479 35.369 22.374; LEU 71 34.145 35.472 23.481; ARG 72 35.161 34.174 26.896
LEU 73 38.668 35.502 27.680; ARG 74 40.873 33.802 30.253; GLY 75 41.845 36.550 32.686; GLY 76 40.373 39.813 33.944
`;
const INSULIN_A_CA = `
GLY 1 -9.929 17.026 13.244; ILE 2 -10.488 14.266 10.600; VAL 3 -6.966 12.901 10.576; GLU 4 -6.889 12.474 14.295
GLN 5 -10.407 11.299 14.630; CYS 6 -10.050 8.518 12.065; CYS 7 -6.964 7.186 13.808; THR 8 -7.862 7.732 17.520
SER 9 -11.509 6.803 17.121; ILE 10 -13.350 5.723 13.932; CYS 11 -14.665 7.679 10.880; SER 12 -16.999 6.978 8.005
LEU 13 -16.530 7.444 4.259; TYR 14 -19.282 10.035 4.368; GLN 15 -17.178 12.138 6.774; LEU 16 -14.185 11.826 4.470
GLU 17 -16.223 13.179 1.589; ASN 18 -16.029 16.534 3.332; TYR 19 -12.358 16.724 2.380; CYS 20 -12.838 16.309 -1.389
ASN 21 -12.404 19.399 -3.608
`;
const INSULIN_B_CA = `
PHE 1 -20.374 1.368 4.053; VAL 2 -18.754 0.598 7.406; ASN 3 -16.872 2.691 9.950; GLN 4 -13.137 2.542 9.571
HIS 5 -10.083 3.000 11.779; LEU 6 -7.468 4.709 9.611; CYS 7 -4.058 5.835 10.569; GLY 8 -1.070 7.261 8.632
SER 9 -0.863 6.544 4.980; HIS 10 -3.940 4.379 5.188; LEU 11 -6.071 7.518 5.617; VAL 12 -4.579 9.057 2.495
GLU 13 -5.432 5.946 0.542; ALA 14 -9.061 6.164 1.719; LEU 15 -9.224 9.854 0.717; TYR 16 -7.838 8.961 -2.686
LEU 17 -10.566 6.290 -3.167; VAL 18 -13.351 8.545 -1.933; CYS 19 -12.433 11.758 -3.756; GLY 20 -10.685 10.359 -6.851
GLU 21 -10.398 13.043 -9.501; ARG 22 -12.089 15.732 -7.456; GLY 23 -8.844 15.673 -5.491; PHE 24 -8.584 16.432 -1.765
PHE 25 -6.813 18.302 1.052; TYR 26 -4.996 16.333 3.717; THR 27 -4.902 18.555 6.772; PRO 28 -3.494 17.055 9.954
LYS 29 -3.227 18.941 13.307; ALA 30 0.148 20.539 13.902
`;
const RNASE_CISPRO_CA = `
GLY 88 27.478 -3.934 11.661; SER 89 23.732 -4.382 12.046; SER 90 22.792 -1.132 10.239; LYS 91 20.555 -1.790 7.259
TYR 92 18.656 0.896 5.387; PRO 93 16.084 2.138 6.263; ASN 94 16.942 0.899 9.749; CYS 95 20.230 2.728 10.122
ALA 96 22.021 1.934 13.355; TYR 97 24.868 3.697 15.059; LYS 98 27.288 3.450 17.931
`;
const T4L_CHIRAL_CA = `
LEU 15 47.636 17.681 21.834; LYS 16 47.212 21.475 21.810; ILE 17 44.074 23.338 20.565; TYR 18 44.310 23.876 16.740
LYS 19 42.265 25.113 13.774; ASP 20 40.547 22.754 11.219; THR 21 41.039 22.983 7.550; GLU 22 37.885 25.043 8.022
GLY 23 39.425 27.326 10.601; TYR 24 37.617 25.660 13.497; TYR 25 38.919 25.024 16.909; THR 26 39.819 21.317 17.300
ILE 27 41.952 19.059 19.352
`;
const TRP_CAGE_SG = `
ASN 1 -8.608 3.135 -1.618; ASN 1 CM -10.857 3.158 -2.642; LEU 2 -4.923 4.002 -2.452
LEU 2 CM -4.542 6.613 -1.699; TYR 3 -3.690 2.738 0.981; TYR 3 CM -1.743 3.194 4.149
ILE 4 -5.857 -0.449 0.613; ILE 4 CM -7.890 -0.349 1.189; GLN 5 -4.122 -1.167 -2.743
GLN 5 CM -5.283 -1.507 -5.275; TRP 6 -0.716 -0.631 -0.993; TRP 6 CM 1.971 0.547 1.695
LEU 7 -1.641 -2.932 1.963; LEU 7 CM -2.395 -1.889 4.258; LYS 8 -3.024 -5.791 -0.269
LYS 8 CM -3.979 -5.475 -3.721; ASP 9 0.466 -6.016 -1.905; ASP 9 CM 0.700 -4.903 -4.118
GLY 10 2.060 -6.618 1.593; GLY 11 2.626 -2.967 2.723; PRO 12 6.333 -2.533 3.806
PRO 12 CM 5.521 -2.318 5.475; SER 13 7.049 -6.179 2.704; SER 13 CM 6.611 -7.318 4.161
SER 14 6.389 -5.315 -1.015; SER 14 CM 4.672 -5.368 -1.812; GLY 15 9.451 -3.116 -1.870
ARG 16 7.289 0.084 -2.054; ARG 16 CM 4.930 -2.099 -4.261; PRO 17 6.782 3.088 0.345
PRO 17 CM 7.684 3.959 -1.033; PRO 18 3.287 4.031 1.686; PRO 18 CM 4.271 4.265 3.246
PRO 19 1.185 6.543 -0.353; PRO 19 CM 0.082 5.047 -0.456; SER 20 0.852 10.027 1.285
SER 20 CM 2.546 10.806 1.598
`;
const THREAD_SEQ = 'MKVEHFCRTGGPDSGEPWPA';
const TRP_CAGE_ALLATOM_GOLDEN = `
1 N -8.683 4.242 -0.587; 1 CA -8.608 3.135 -1.618; 1 CB -9.821 3.303 -2.542; 1 CG -11.116 2.886 -1.867
1 OD1 -11.258 1.738 -1.441; 1 ND2 -12.047 3.813 -1.736; 1 C -7.168 2.958 -1.987; 1 O -6.814 1.803 -2.189
2 N -6.363 4.006 -2.120; 2 CA -4.923 4.002 -2.452; 2 CB -4.410 5.410 -2.743; 2 CG -5.216 6.182 -3.804
2 CD1 -4.529 7.494 -4.183; 2 CD2 -5.341 5.306 -5.035; 2 C -4.174 3.076 -1.365; 2 O -3.373 2.200 -1.656
3 N -4.416 3.450 -0.109; 3 CA -3.690 2.738 0.981; 3 CB -4.121 3.292 2.344; 3 CG -3.544 2.553 3.534
3 CD1 -2.170 2.567 3.793; 3 CD2 -4.374 1.855 4.410; 3 CE1 -1.642 1.908 4.892; 3 CE2 -3.854 1.191 5.517
3 CZ -2.486 1.224 5.754; 3 OH -1.967 0.591 6.862; 3 C -4.086 1.252 0.938; 3 O -3.278 0.356 1.174
4 N -5.357 1.014 0.621; 4 CA -5.857 -0.449 0.613; 4 CB -7.348 -0.482 0.316; 4 CG1 -8.233 0.204 1.353
4 CG2 -7.861 -1.906 0.097; 4 CD1 -8.079 -0.291 2.758; 4 C -5.049 -1.180 -0.510; 4 O -4.620 -2.309 -0.298
5 N -4.840 -0.549 -1.663; 5 CA -4.122 -1.167 -2.743; 5 CB -4.606 -0.642 -4.106; 5 CG -4.518 0.857 -4.304
5 CD -5.312 1.342 -5.513; 5 OE1 -4.898 2.273 -6.209; 5 NE2 -6.464 0.719 -5.759; 5 C -2.681 -1.453 -2.255
5 O -2.164 -2.534 -2.528; 6 N -2.078 -0.531 -1.517; 6 CA -0.716 -0.631 -0.993; 6 CB -0.055 0.638 -0.527
6 CG 0.115 1.594 -1.689; 6 CD1 -0.644 2.701 -1.955; 6 CD2 1.155 1.579 -2.673; 6 NE1 -0.145 3.339 -3.052
6 CE2 0.964 2.683 -3.504; 6 CE3 2.252 0.722 -2.917; 6 CZ2 1.790 2.948 -4.603; 6 CZ3 3.077 0.975 -3.986
6 CH2 2.826 2.082 -4.828; 6 C -0.674 -1.838 0.010; 6 O 0.196 -2.703 -0.123; 7 N -1.601 -1.897 0.956
7 CA -1.641 -2.932 1.963; 7 CB -2.729 -2.677 3.007; 7 CG -2.477 -1.497 3.943; 7 CD1 -3.713 -1.147 4.769
7 CD2 -1.323 -1.828 4.897; 7 C -1.826 -4.338 1.353; 7 O -1.313 -5.345 1.838; 8 N -2.632 -4.379 0.292
8 CA -3.024 -5.791 -0.269; 8 CB -4.170 -5.631 -1.255; 8 CG -4.759 -6.960 -1.661; 8 CD -5.903 -6.739 -2.643
8 CE -5.955 -7.864 -3.671; 8 NZ -6.850 -7.501 -4.846; 8 C -1.687 -6.385 -0.784; 8 O -1.465 -7.592 -0.701
9 N -0.820 -5.527 -1.319; 9 CA 0.466 -6.016 -1.905; 9 CB 1.217 -4.843 -2.546; 9 CG 0.613 -4.399 -3.865
9 OD1 -0.226 -5.130 -4.434; 9 OD2 0.996 -3.306 -4.337; 9 C 1.444 -6.353 -0.788; 9 O 2.628 -6.434 -1.098
10 N 1.021 -6.414 0.496; 10 CA 2.060 -6.618 1.593; 10 C 2.739 -5.347 1.911; 10 O 3.884 -5.263 1.501
11 N 2.096 -4.397 2.565; 11 CA 2.626 -2.967 2.723; 11 C 3.959 -3.142 3.584; 11 O 3.890 -3.740 4.660
12 N 5.052 -2.547 3.131; 12 CA 6.333 -2.533 3.806; 12 CB 7.222 -1.865 4.839; 12 CG 7.061 -2.791 5.977
12 CD 5.574 -3.011 6.021; 12 C 7.009 -3.729 3.214; 12 O 8.093 -3.550 2.655; 13 N 6.416 -4.918 3.245
13 CA 7.049 -6.179 2.704; 13 CB 6.397 -7.424 3.248; 13 OG 5.129 -7.689 2.672; 13 C 6.846 -6.306 1.168
13 O 6.880 -7.434 0.669; 14 N 6.639 -5.212 0.446; 14 CA 6.389 -5.315 -1.015; 14 CB 5.018 -5.230 -1.685
14 OG 4.249 -6.394 -1.364; 14 C 7.294 -4.291 -1.728; 14 O 6.929 -3.857 -2.832; 15 N 8.409 -3.879 -1.123
15 CA 9.451 -3.116 -1.870; 15 C 9.051 -1.620 -1.826; 15 O 9.935 -0.782 -1.734; 16 N 7.778 -1.265 -1.939
16 CA 7.289 0.084 -2.054; 16 CB 6.050 -0.021 -2.937; 16 CG 6.260 -0.820 -4.213; 16 CD 4.902 -1.427 -4.523
16 NE 4.916 -2.283 -5.687; 16 CZ 4.936 -1.793 -6.926; 16 NH1 4.947 -0.477 -7.188; 16 NH2 4.945 -2.669 -7.923
16 C 7.260 1.141 -0.919; 16 O 7.042 0.581 0.155; 17 N 7.050 2.449 -1.089; 17 CA 6.782 3.088 0.345
17 CB 7.538 4.396 0.064; 17 CG 8.212 4.198 -1.273; 17 CD 8.465 2.728 -1.372; 17 C 5.229 3.253 0.438
17 O 4.530 2.960 -0.531; 18 N 4.774 3.830 1.542; 18 CA 3.287 4.031 1.686; 18 CB 3.193 4.325 3.183
18 CG 4.284 3.472 3.764; 18 CD 5.414 3.710 2.798; 18 C 2.853 5.129 0.860; 18 O 3.710 5.984 0.637
19 N 1.569 5.300 0.506; 19 CA 1.185 6.543 -0.353; 19 CB -0.169 6.197 -0.939; 19 CG -0.392 4.770 -0.792
19 CD 0.710 4.171 -0.015; 19 C 1.042 7.641 0.785; 19 O 0.974 7.356 1.980; 20 N 0.950 8.908 0.350
20 CA 0.852 10.027 1.285; 20 CB 1.230 11.365 0.707; 20 OG 0.338 11.812 -0.283; 20 C -0.719 10.215 1.708
20 O -1.641 10.116 0.899; 20 OXT -0.915 10.537 2.980
`;
const UBIQUITIN_NCO_GOLDEN = `
1 N 27.589 24.578 2.230; 1 C 27.211 26.523 3.696; 1 O 28.293 26.457 4.278; 2 N 26.601 27.677 3.444
2 C 26.199 29.371 5.068; 2 O 24.993 29.195 5.189; 3 N 26.949 29.810 6.063; 3 C 26.850 31.295 7.914
3 O 27.970 31.538 7.459; 4 N 26.217 32.117 8.761; 4 C 27.266 33.328 10.810; 4 O 26.599 32.669 11.609
5 N 28.320 34.044 11.193; 5 C 28.684 35.423 12.766; 5 O 29.420 36.163 12.118; 6 N 27.851 35.869 13.699
6 C 28.514 37.579 15.411; 6 O 28.206 36.956 16.430; 7 N 29.520 38.443 15.363; 7 C 29.508 39.803 17.443
7 O 28.554 40.463 17.008; 8 N 30.093 40.083 18.616; 8 C 29.968 42.574 18.954; 8 O 29.370 43.554 19.374
9 N 30.970 42.658 18.085; 9 C 30.718 44.446 16.341; 9 O 31.111 45.536 15.799; 10 N 29.727 43.703 15.831
10 C 29.421 42.899 13.741; 10 O 28.740 41.879 13.637; 11 N 30.580 43.092 13.109; 11 C 30.972 40.543 12.247
11 O 31.410 39.659 12.974; 12 N 30.100 40.298 11.273; 12 C 30.575 38.352 9.609; 12 O 31.040 38.943 8.639
13 N 30.929 37.119 9.948; 13 C 30.786 35.052 8.787; 13 O 29.907 34.484 9.426; 14 N 31.018 34.821 7.501
14 C 31.299 32.690 6.507; 14 O 32.517 32.815 6.406; 15 N 30.718 31.510 6.624; 15 C 30.931 29.217 5.893
15 O 29.701 29.135 5.899; 16 N 31.688 28.408 5.163; 16 C 31.511 26.091 4.996; 16 O 32.639 25.745 5.368
17 N 30.422 25.366 5.228; 17 C 28.958 23.430 5.779; 17 O 28.084 23.739 4.964; 18 N 29.195 22.173 6.181
18 C 28.002 20.492 7.354; 18 O 28.702 20.322 8.352; 19 N 26.685 20.279 7.346; 19 C 26.582 18.770 9.287
19 O 26.621 18.960 10.513; 20 N 27.222 17.768 8.680; 20 C 29.315 17.225 9.929; 20 O 30.030 16.390 10.561
21 N 29.566 18.501 9.873; 21 C 30.529 19.437 12.076; 21 O 29.671 20.213 12.500; 22 N 31.369 18.781 12.893
22 C 31.524 20.481 14.753; 22 O 32.026 21.283 13.965; 23 N 31.113 20.841 15.965; 23 C 32.779 22.524 16.626
23 O 33.130 23.658 16.317; 24 N 33.651 21.594 17.010; 24 C 35.615 22.177 15.749; 24 O 36.431 23.096 15.753
25 N 35.174 21.609 14.635; 25 C 35.240 23.402 12.952; 25 O 36.037 24.156 12.415; 26 N 33.985 23.753 13.232
26 C 34.406 26.140 13.740; 26 O 34.787 27.154 13.165; 27 N 34.757 25.865 14.995; 27 C 37.007 26.837 15.053
27 O 37.545 27.936 14.964; 28 N 37.564 25.735 14.555; 28 C 38.724 26.647 12.601; 28 O 39.639 27.424 12.350
29 N 37.636 26.566 11.838; 29 C 37.467 28.899 11.103; 29 O 38.105 29.725 10.458; 30 N 36.768 29.236 12.185
30 C 38.178 31.031 13.052; 30 O 38.599 32.136 12.726; 31 N 38.923 30.166 13.738; 31 C 41.159 30.721 12.836
31 O 41.948 31.659 12.778; 32 N 41.007 29.862 11.830; 32 C 41.430 31.453 9.993; 32 O 42.307 32.224 9.608
33 N 40.128 31.730 9.893; 33 C 40.178 34.330 10.084; 33 O 40.887 35.298 9.798; 34 N 39.483 34.248 11.210
34 C 40.564 35.401 13.275; 34 O 40.619 36.314 14.114; 35 N 41.358 34.341 13.340; 35 C 41.921 34.299 15.818
35 O 42.542 34.894 16.699; 36 N 40.770 33.670 16.051; 36 C 40.419 32.255 17.872; 36 O 39.994 31.346 17.150
37 N 41.116 32.038 18.985; 37 C 40.139 30.134 20.135; 37 O 39.338 30.682 20.895; 38 N 39.965 28.903 19.648
38 C 38.308 28.069 21.345; 38 O 37.125 28.066 21.687; 39 N 39.326 28.083 22.200; 39 C 38.181 29.246 24.044
39 O 37.270 29.084 24.846; 40 N 38.485 30.421 23.505; 40 C 36.339 31.705 23.133; 40 O 35.549 32.545 23.563
41 N 36.027 30.837 22.171; 41 C 33.340 30.505 22.242; 41 O 33.462 29.359 22.678; 42 N 32.169 31.135 22.235
42 C 30.232 30.618 21.586; 42 O 30.035 31.757 21.172; 43 N 29.707 29.540 21.015; 43 C 27.461 29.455 20.360
43 O 27.228 28.697 21.298; 44 N 26.517 30.179 19.786; 44 C 24.222 30.305 18.965; 44 O 24.588 30.737 17.863
45 N 23.074 29.657 19.163; 45 C 20.876 28.791 18.882; 45 O 20.940 28.037 19.876; 46 N 19.740 29.334 18.457
46 C 18.445 29.066 20.564; 46 O 17.749 28.222 21.120; 47 N 19.283 29.878 21.210; 47 C 20.244 28.784 23.260
47 O 20.467 28.785 24.486; 48 N 20.788 27.854 22.492; 48 C 23.045 26.919 22.954; 48 O 23.382 27.048 21.775
49 N 23.924 26.738 23.937; 49 C 25.826 25.573 22.849; 49 O 25.674 24.468 23.363; 50 N 26.479 25.737 21.680
50 C 28.306 24.005 21.857; 50 O 29.197 24.634 22.418; 51 N 28.229 22.681 21.871; 51 C 29.899 21.093 21.360
51 O 29.405 20.576 20.373; 52 N 31.206 21.196 21.518; 52 C 31.947 19.353 19.675; 52 O 32.108 19.297 18.454
53 N 31.715 18.283 20.431; 53 C 30.316 16.659 18.997; 53 O 30.198 15.574 18.418; 54 N 29.338 17.565 19.073
54 C 28.361 18.094 16.850; 54 O 29.283 18.884 16.636; 55 N 27.484 17.730 15.906; 55 C 26.479 19.221 14.412
55 O 25.562 19.281 15.232; 56 N 26.583 20.050 13.377; 56 C 24.226 20.484 12.856; 56 O 23.219 21.055 13.270
57 N 24.187 19.299 12.251; 57 C 22.205 18.250 13.305; 57 O 20.963 18.242 13.360; 58 N 22.989 18.024 14.358
58 C 21.581 18.779 16.312; 58 O 20.857 18.603 17.290; 59 N 21.772 19.975 15.761; 59 C 19.829 21.482 15.371
59 O 18.838 22.119 15.746; 60 N 19.989 21.100 14.100; 60 C 19.181 22.835 12.631; 60 O 18.272 23.657 12.508
61 N 20.440 23.101 12.287; 61 C 20.841 24.300 10.100; 61 O 21.519 23.538 9.412; 62 N 19.959 25.149 9.563
62 C 20.832 25.907 7.434; 62 O 21.328 26.849 8.075; 63 N 20.984 25.818 6.121; 63 C 21.201 28.201 5.555
63 O 20.014 28.432 5.792; 64 N 22.156 29.129 5.614; 64 C 21.239 30.747 7.233; 64 O 20.214 31.418 7.326
65 N 21.868 30.197 8.266; 65 C 22.516 30.896 10.484; 65 O 23.609 30.354 10.655; 66 N 22.217 32.092 11.018
66 C 23.335 32.363 13.339; 66 O 22.368 32.198 14.075; 67 N 24.606 32.148 13.719; 67 C 25.976 32.620 15.658
67 O 27.056 33.005 15.207; 68 N 25.453 33.117 16.791; 68 C 27.417 33.499 18.434; 68 O 27.449 32.425 19.026
69 N 28.474 34.291 18.304; 69 C 30.016 34.947 20.048; 69 O 30.015 36.159 19.856; 70 N 30.202 34.398 21.241
70 C 31.837 34.939 22.932; 70 O 32.003 33.805 23.384; 71 N 32.820 35.853 22.908; 71 C 34.421 35.491 24.871
71 O 34.120 36.473 25.547; 72 N 34.870 34.360 25.398; 72 C 36.780 34.289 26.863; 72 O 37.470 33.511 26.211
73 N 37.289 35.303 27.554; 73 C 39.250 35.072 28.926; 73 O 38.666 35.396 29.944; 74 N 40.314 34.288 28.944
74 C 41.568 34.462 31.424; 74 O 42.303 33.815 32.189; 75 N 41.274 35.749 31.629; 75 C 41.286 37.935 32.741
75 O 41.169 38.560 31.672; 76 N 40.836 38.411 33.909; 76 C 40.914 41.233 33.572; 76 O 40.306 42.295 33.727
76 OXT 42.241 41.159 33.535
`;
const DIGESTS = {
  trpDefault: '154:77635cee',
  // Upstream C 3.06 bug fix: optimize_backbone() previously allocated the spatial grid before
  // sidechains existed, leaving fixClashes() with a stale backbone-only grid. Rebuilding the
  // grid before fixClashes() resolves real sidechain clashes when optimizeHbonds is true.
  trpHbonds: '154:3b5429b4',
  trpNoClashFix: '154:3b4cc5ff',
  trpBackboneOnly: '81:a16f249b',
  trpNoBackbone: '93:cb994610',
  trpCaOptimized: '154:9245224d',
  trpThreaded: '154:6511bc61',
  trpSidechainCenters: '154:a228cf91',
  ubqDefault: '602:02dc8684',
  // Updated for the same clash-grid rebuild fix under optimizeHbonds: true (0 remaining clashes).
  ubqHbonds: '602:2f3e6745',
  insulinDefault: '403:3d771e34',
  rnaseCaOptimized: '85:14b8aae6',
  rnaseCisPro: '85:7e3b2458',
  't4lDefault': '114:5f8f1027',
  't4lNoChirality': '114:6ea9d290'
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

const rows = (block: string) =>
  block
    .split(/[;\n]/)
    .map((r) => r.trim())
    .filter(Boolean);

/** Cα (or Cα + CM pseudo-atom) trace PDB from "RES num [name] x y z" rows. */
function tracePdb(chains: Array<[string, string]>): string {
  const out: string[] = [];
  let serial = 1;
  for (const [chainId, block] of chains) {
    for (const row of rows(block)) {
      const f = row.split(/\s+/);
      const [res, num] = f;
      const name = f.length === 6 ? f[2] : 'CA';
      const [x, y, z] = f.slice(-3).map(Number);
      out.push(
        `ATOM  ${String(serial++).padStart(5)}  ${name.padEnd(3)} ${res} ${chainId}${num.padStart(4)}    ` +
          `${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00  0.00           ${name[0]}`,
      );
    }
    out.push('TER');
  }
  out.push('END');
  return out.join('\n') + '\n';
}

interface PdbAtom {
  chain: string;
  resSeq: number;
  resName: string;
  name: string;
  xyz: [number, number, number];
}

function pdbAtoms(pdb: string): PdbAtom[] {
  return pdb
    .split('\n')
    .filter((l) => l.startsWith('ATOM'))
    .map((l) => ({
      chain: l[21],
      resSeq: Number.parseInt(l.substring(22, 26), 10),
      resName: l.substring(17, 20),
      name: l.substring(12, 16).trim(),
      xyz: [30, 38, 46].map((o) => Number.parseFloat(l.substring(o, o + 8))) as [number, number, number],
    }));
}

/** Order-independent digest of printed coordinates (milli-Å), as pinned from the C binary. */
function digest(pdb: string): string {
  const lines = pdbAtoms(pdb)
    .map((a) => `${a.chain}|${a.resSeq}|${a.resName}|${a.name}|${a.xyz.map((v) => Math.round(v * 1000) + 0).join(' ')}`)
    .sort();
  return `${lines.length}:${fnv1a(lines.join('\n'))}`;
}

const dist = (a: readonly number[], b: readonly number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function atomMap(pdb: string): Map<string, [number, number, number]> {
  return new Map(pdbAtoms(pdb).map((a) => [`${a.chain}${a.resSeq}:${a.name}`, a.xyz]));
}

function goldenMap(block: string, chain = 'A'): Map<string, [number, number, number]> {
  return new Map(
    rows(block).map((r) => {
      const [num, name, x, y, z] = r.split(/\s+/);
      return [`${chain}${num}:${name}`, [Number(x), Number(y), Number(z)] as [number, number, number]];
    }),
  );
}

/** Deterministic ±amp Å jitter (LCG) for distorted-trace tests. */
function jitter(pdb: string, amp: number, seed = 7): string {
  let s = seed >>> 0;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32 - 0.5) * 2 * amp;
  return pdb
    .split('\n')
    .map((l) =>
      l.startsWith('ATOM')
        ? l.substring(0, 30) +
          [30, 38, 46].map((o) => (Number.parseFloat(l.substring(o, o + 8)) + rnd()).toFixed(3).padStart(8)).join('') +
          l.substring(54)
        : l,
    )
    .join('\n');
}

const tables = decodePulchraTables(readFileSync(resolve(process.cwd(), 'static/vendor/pulchra/pulchra-tables.bin')));
const TRP = tracePdb([['A', TRP_CAGE_CA]]);
const UBQ = tracePdb([['A', UBIQUITIN_CA]]);
const INSULIN = tracePdb([
  ['A', INSULIN_A_CA],
  ['B', INSULIN_B_CA],
]);

const HEAVY_SIDECHAIN: Record<string, string[]> = {
  GLY: [], ALA: ['CB'], SER: ['CB', 'OG'], CYS: ['CB', 'SG'], VAL: ['CB', 'CG1', 'CG2'], THR: ['CB', 'OG1', 'CG2'],
  ILE: ['CB', 'CG1', 'CG2', 'CD1'], PRO: ['CB', 'CG', 'CD'], MET: ['CB', 'CG', 'SD', 'CE'], ASP: ['CB', 'CG', 'OD1', 'OD2'],
  ASN: ['CB', 'CG', 'OD1', 'ND2'], LEU: ['CB', 'CG', 'CD1', 'CD2'], LYS: ['CB', 'CG', 'CD', 'CE', 'NZ'],
  GLU: ['CB', 'CG', 'CD', 'OE1', 'OE2'], GLN: ['CB', 'CG', 'CD', 'OE1', 'NE2'], ARG: ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
  HIS: ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'], PHE: ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
  TYR: ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
  TRP: ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
};

function structureFromPdb(pdbText: string, id: string, extra: Partial<StructureArtifact> = {}): StructureArtifact {
  const parsed = parseStructureText(pdbText);
  return {
    kind: 'StructureArtifact',
    id,
    label: id,
    sourceModelId: 'genie2',
    length: parsed.nResidues,
    sequence: parsed.sequence,
    caCoords: parsed.caCoords,
    backbone4Coords: parsed.backbone4Coords,
    pdbText: parsed.pdbText,
    chainIds: parsed.chainIds,
    atomGranularity: 'ca-only',
    geometry: computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes),
    ...extra,
  };
}

const pulchraNode = (params: Record<string, unknown> = {}): WorkflowNode => ({
  id: 'pu_1',
  adapterId: 'pulchra',
  params: { ...defaultParamsForAdapter('pulchra'), ...params },
});

// ─── Tables ──────────────────────────────────────────────────────────────────

describe('PULCHRA lookup tables (static/vendor/pulchra/pulchra-tables.bin)', () => {
  it('decodes the packed nco_stat / rot_stat tables with the exact C array shapes and values', () => {
    const f = (q: number) => Math.fround(q / tables.scale);
    expect(tables.byteLength).toBeGreaterThan(1_000_000);
    expect(tables.byteLength).toBeLessThan(3_000_000);
    expect(tables.ncoBins.length / 3).toBe(2363);
    expect(tables.ncoProBins.length / 3).toBe(1432);
    expect(tables.rotPos.length).toBe(19897);
    expect(tables.rotCoords.length / 3).toBe(217372);
    // nco_stat[0] = {{0, 0, 12}, {-1.590, -3.016, 1.326, ...}} and its {-1,-1,-1} terminator
    expect([...tables.ncoBins.slice(0, 3)]).toEqual([0, 0, 12]);
    expect([0, 1, 2, 23].map((k) => f(tables.ncoData[k]))).toEqual([-1.59, -3.016, 1.326, 2.306].map(Math.fround));
    expect([...tables.ncoBins.slice(-3)]).toEqual([-1, -1, -1]);
    expect(f(tables.ncoData[2362 * 24])).toBe(Math.fround(3.084));
    expect([...tables.ncoProBins.slice(-3)]).toEqual([-1, -1, -1]);
    // rot_stat_idx[1] = {1, 0, 1, 53, 1, 2}; rot_stat_idx[19895] = {19, 9, 9, 73, 1, 217361}
    expect([...tables.rotIdx.slice(5, 10), tables.rotPos[1]]).toEqual([1, 0, 1, 53, 1, 2]);
    expect([...tables.rotIdx.slice(19895 * 5, 19896 * 5), tables.rotPos[19895]]).toEqual([19, 9, 9, 73, 1, 217361]);
    expect(tables.rotIdx[19896 * 5]).toBe(-1);
    // rot_stat_coords[1] = {0.332, 1.054, -1.023}; last row {2.334, -4.531, -3.151}
    expect([3, 4, 5].map((k) => f(tables.rotCoords[k]))).toEqual([0.332, 1.054, -1.023].map(Math.fround));
    expect([-3, -2, -1].map((k) => f(tables.rotCoords.at(k)!))).toEqual([2.334, -4.531, -3.151].map(Math.fround));
    expect(tables.rotRowsByType[0]).toHaveLength(0); // no glycine rotamers
    expect(tables.rotRowsByType.reduce((n, r) => n + r.length, 0)).toBe(19896);
  });

  it('rejects truncated or foreign files and accepts Node Buffer views at any byte offset', () => {
    const bytes = new Uint8Array(readFileSync(resolve(process.cwd(), 'static/vendor/pulchra/pulchra-tables.bin')));
    expect(() => decodePulchraTables(bytes.slice(0, 32))).toThrow(/truncated/);
    expect(() => decodePulchraTables(bytes.slice(0, bytes.length - 8))).toThrow(/expected/);
    const foreign = bytes.slice();
    foreign[0] = 0x58;
    expect(() => decodePulchraTables(foreign)).toThrow(/magic/);
    const future = bytes.slice();
    new DataView(future.buffer).setUint32(8, 99, true);
    expect(() => decodePulchraTables(future)).toThrow(/version 99/);
    const padded = new Uint8Array(bytes.length + 3);
    padded.set(bytes, 3);
    const t = decodePulchraTables(padded.subarray(3));
    expect(t.rotCoords).toEqual(tables.rotCoords);
  });

  it('memoizes the lazily fetched tables and retries after a failed download', async () => {
    const body = readFileSync(resolve(process.cwd(), 'static/vendor/pulchra/pulchra-tables.bin'));
    const fail = vi.fn(async () => new Response(null, { status: 503 }));
    await expect(loadPulchraTables({ fetchFn: fail as unknown as typeof fetch })).rejects.toThrow(/HTTP 503/);
    expect(fail).toHaveBeenCalledWith(PULCHRA_TABLES_URL);
    const ok = vi.fn(async () => new Response(new Uint8Array(body)));
    const first = await loadPulchraTables({ fetchFn: ok as unknown as typeof fetch });
    const again = vi.fn(async () => new Response(new Uint8Array(body)));
    const second = await loadPulchraTables({ fetchFn: again as unknown as typeof fetch });
    expect(second).toBe(first);
    expect(ok).toHaveBeenCalledTimes(1);
    expect(again).not.toHaveBeenCalled();
    expect(await loadPulchraTables({ tablesBuffer: body })).toBe(await loadPulchraTables({ tablesBuffer: body }));
  });
});

// ─── Parity with the native PULCHRA 3.06 binary ─────────────────────────────

describe('PULCHRA 3.06 parity with the native C reference (pinned goldens)', () => {
  it('rebuilds trp-cage (1L2Y) from its Cα trace with the same atoms and coordinates as the C binary', () => {
    const out = reconstructPdb(TRP, tables).pdbText;
    const got = atomMap(out);
    const want = goldenMap(TRP_CAGE_ALLATOM_GOLDEN);
    expect([...got.keys()].sort()).toEqual([...want.keys()].sort());
    let maxErr = 0;
    let bbSq = 0;
    let bbN = 0;
    for (const [key, xyz] of want) {
      const d = dist(xyz, got.get(key)!);
      maxErr = Math.max(maxErr, ...xyz.map((v, i) => Math.abs(v - got.get(key)![i])));
      if (/:(N|CA|C|O)$/.test(key)) {
        bbSq += d * d;
        bbN++;
      }
    }
    expect(maxErr).toBeLessThanOrEqual(0.0015);
    expect(Math.sqrt(bbSq / bbN)).toBeLessThan(0.01);
    expect(digest(out)).toBe(DIGESTS.trpDefault);
  });

  it.each([
    ['trpHbonds', TRP, { optimizeHbonds: true }, '-q'],
    ['trpNoClashFix', TRP, { fixClashes: false }, '-o'],
    ['trpBackboneOnly', TRP, { reconstructSidechains: false }, '-s'],
    ['trpNoBackbone', TRP, { reconstructBackbone: false }, '-b'],
    ['trpCaOptimized', TRP, { optimizeCa: true, maxCaShift: 0.5 }, 'classic -u 0.5'],
    ['trpThreaded', TRP, { sequence: THREAD_SEQ }, 'renamed input'],
    ['trpSidechainCenters', tracePdb([['A', TRP_CAGE_SG]]), { useSidechainCenters: true }, '-g'],
    ['ubqDefault', UBQ, {}, 'default'],
    ['ubqHbonds', UBQ, { optimizeHbonds: true }, '-q'],
    ['insulinDefault', INSULIN, {}, 'per chain'],
    ['rnaseCaOptimized', tracePdb([['A', RNASE_CISPRO_CA]]), { optimizeCa: true }, 'classic'],
    ['rnaseCisPro', tracePdb([['A', RNASE_CISPRO_CA]]), { optimizeCa: true, detectCisProline: true }, 'classic -p'],
    ['t4lDefault', tracePdb([['A', T4L_CHIRAL_CA]]), {}, 'default'],
    ['t4lNoChirality', tracePdb([['A', T4L_CHIRAL_CA]]), { checkChirality: false }, '-z'],
  ] as Array<[keyof typeof DIGESTS, string, PulchraOptions, string]>)(
    'matches the C reference golden %s (pulchra %s)',
    (key, input, opts) => {
      expect(digest(reconstructPdb(input, tables, opts).pdbText)).toBe(DIGESTS[key]);
    },
  );

  it('places ubiquitin N/C/O/OXT within 0.01 Å RMSD of the C binary and keeps every input Cα exactly', () => {
    const got = atomMap(reconstructPdb(UBQ, tables).pdbText);
    const want = goldenMap(UBIQUITIN_NCO_GOLDEN);
    let sq = 0;
    for (const [key, xyz] of want) sq += dist(xyz, got.get(key)!) ** 2;
    expect(Math.sqrt(sq / want.size)).toBeLessThan(0.01);
    for (const row of rows(UBIQUITIN_CA)) {
      const [, num, x, y, z] = row.split(/\s+/);
      expect(got.get(`A${num}:CA`)).toEqual([Number(x), Number(y), Number(z)]);
    }
    expect([...got.keys()].filter((k) => k.endsWith(':OXT'))).toEqual(['A76:OXT']);
  });
});

// ─── Reconstruction behaviour ───────────────────────────────────────────────

describe('PULCHRA reconstruction', () => {
  it('turns a Cα-only trace into a complete heavy-atom model with ideal backbone geometry', () => {
    const res = reconstructPdb(UBQ, tables);
    const residues = res.chains[0].residues;
    expect(residues).toHaveLength(76);
    const bonds: Record<'nca' | 'cac' | 'co' | 'cn', number[]> = { nca: [], cac: [], co: [], cn: [] };
    expect(res.sequence).toBe('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG');
    for (const [i, r] of residues.entries()) {
      const names = r.atoms.map((a) => a.name);
      const expected = ['N', 'CA', 'C', 'O', ...HEAVY_SIDECHAIN[r.resName], ...(i === 75 ? ['OXT'] : [])];
      expect(names.slice().sort()).toEqual(expected.slice().sort());
      const at = (n: string, rr = r) => {
        const a = rr.atoms.find((x) => x.name === n)!;
        return [a.x, a.y, a.z];
      };
      bonds.nca.push(dist(at('N'), at('CA')));
      bonds.cac.push(dist(at('CA'), at('C')));
      bonds.co.push(dist(at('C'), at('O')));
      if (i > 0) bonds.cn.push(dist(at('C', residues[i - 1]), at('N')));
    }
    // N/C/O come from Cα-frame templates: peptide groups are near-ideal per residue,
    // N–Cα / Cα–C spread with the template fit but average to Engh–Huber values.
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    const within = (values: number[], lo: number, hi: number) => {
      for (const v of values) {
        expect(v).toBeGreaterThan(lo);
        expect(v).toBeLessThan(hi);
      }
    };
    within(bonds.co, 1.19, 1.33);
    within(bonds.cn, 1.28, 1.38);
    within([...bonds.nca, ...bonds.cac], 1.15, 2.0);
    expect(mean(bonds.nca)).toBeGreaterThan(1.43);
    expect(mean(bonds.nca)).toBeLessThan(1.5);
    expect(mean(bonds.cac)).toBeGreaterThan(1.5);
    expect(mean(bonds.cac)).toBeLessThan(1.56);
    expect(mean(bonds.co)).toBeCloseTo(1.23, 1);
    expect(mean(bonds.cn)).toBeCloseTo(1.33, 1);
    const parsed = parseStructureText(res.pdbText);
    const geom = computeGeometryStats(parsed.caCoords, parsed.chainIds, parsed.resNums, parsed.insCodes);
    expect(geom.chainBreaks).toBe(0);
    expect(geom.length).toBe(76);
    expect(geom.helixPct + geom.sheetPct).toBeGreaterThan(30);
    expect(parsed.backbone4Coords.every(Number.isFinite)).toBe(true);
    expect(res.chains[0].stats.backboneRmsdMax).toBeLessThan(0.6);
  });

  it('returns a backbone-only model when side chains are skipped', () => {
    const res = reconstructPdb(TRP, tables, { reconstructSidechains: false });
    const names = new Set(res.chains[0].residues.flatMap((r) => r.atoms.map((a) => a.name)));
    expect([...names].sort()).toEqual(['C', 'CA', 'N', 'O', 'OXT']);
    expect(res.atomCount).toBe(20 * 4 + 1);
  });

  it('keeps input backbone and side-chain atoms of partial traces (preserve mode)', () => {
    const full = reconstructPdb(UBQ, tables).pdbText;
    const field = (l: string) => l.substring(12, 16);
    const bbOnly = full.split('\n').filter((l) => l.startsWith('ATOM') && [' N  ', ' CA ', ' C  ', ' O  '].includes(field(l))).join('\n') + '\nEND\n';
    expect(digest(reconstructPdb(bbOnly, tables).pdbText)).toBe(digest(full));
    // A displaced input CB (Gln2) is kept verbatim...
    const withCb = (xyz: (l: string) => string) =>
      full
        .split('\n')
        .filter((l) => l.startsWith('ATOM') && [' CA ', ' CB '].includes(field(l)))
        .map((l) => (field(l) === ' CB ' && l.substring(22, 26) === '   2' ? l.substring(0, 30) + xyz(l) + l.substring(54) : l))
        .join('\n');
    const nudged = withCb((l) => [30, 38, 46].map((o) => (Number.parseFloat(l.substring(o, o + 8)) + 0.25).toFixed(3).padStart(8)).join(''));
    const kept = atomMap(reconstructPdb(nudged, tables).pdbText);
    expect(kept.get('A2:CB')).toEqual([26.924, 30.46, 3.192]);
    expect(kept.get('A3:CB')).toEqual(atomMap(full).get('A3:CB'));
    // ...unless it makes the residue D-configured: like pulchra.c's chirality_check(),
    // the mirror ignores preservation (C reference: 6.367 16.162 -132.345).
    const far = withCb(() => '  99.000  99.000  99.000');
    expect(atomMap(reconstructPdb(far, tables, { checkChirality: false }).pdbText).get('A2:CB')).toEqual([99, 99, 99]);
    expect(atomMap(reconstructPdb(far, tables).pdbText).get('A2:CB')).toEqual([6.367, 16.162, -132.345]);
  });

  it('optimizes backbone H-bonds and removes side-chain clashes', () => {
    const hb = reconstructPdb(UBQ, tables, { optimizeHbonds: true }).chains[0].stats;
    expect(hb.hbondEnergyAfter!).toBeLessThan(hb.hbondEnergyBefore!);
    const trp = reconstructPdb(TRP, tables).chains[0].stats;
    expect(trp.clashesBefore).toBeGreaterThan(0);
    expect(trp.clashesAfter).toBe(0);
    expect(reconstructPdb(TRP, tables, { fixClashes: false }).chains[0].stats.clashesBefore).toBeNull();
  });

  it('refines a distorted Cα trace toward 3.8 Å virtual bonds within the shift restraint', () => {
    const noisy = jitter(UBQ, 0.5);
    const caCa = (pdb: string) => {
      const ca = pdbAtoms(pdb).filter((a) => a.name === 'CA');
      return ca.slice(1).map((a, i) => Math.abs(dist(a.xyz, ca[i].xyz) - 3.8));
    };
    const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
    const refined = reconstructPdb(noisy, tables, { optimizeCa: true, maxCaShift: 0.5 });
    expect(refined.chains[0].stats.caOptimizationSteps).toBeGreaterThan(1);
    expect(mean(caCa(refined.pdbText))).toBeLessThan(mean(caCa(noisy)) / 3);
    const before = pdbAtoms(noisy).filter((a) => a.name === 'CA');
    const after = pdbAtoms(refined.pdbText).filter((a) => a.name === 'CA');
    expect(Math.max(...after.map((a, i) => dist(a.xyz, before[i].xyz)))).toBeLessThan(1.5);
    // Without refinement the noisy Cα are kept verbatim.
    const kept = pdbAtoms(reconstructPdb(noisy, tables).pdbText).filter((a) => a.name === 'CA');
    expect(kept.map((a) => a.xyz)).toEqual(before.map((a) => a.xyz));
  });

  it('detects cis-prolines (RNase A Pro93) and keeps their 2.9 Å Cα–Cα distance', () => {
    const rnase = tracePdb([['A', RNASE_CISPRO_CA]]);
    const caDist = (pdb: string) => {
      const m = atomMap(pdb);
      return dist(m.get('A92:CA')!, m.get('A93:CA')!);
    };
    const cis = reconstructPdb(rnase, tables, { optimizeCa: true, detectCisProline: true });
    expect(cis.chains[0].stats.cisProlines).toBe(1);
    expect(caDist(cis.pdbText)).toBeCloseTo(2.9, 2);
    expect(caDist(reconstructPdb(rnase, tables, { optimizeCa: true }).pdbText)).toBeCloseTo(3.8, 2);
    // detectCisProline only acts during Cα refinement.
    expect(digest(reconstructPdb(rnase, tables, { detectCisProline: true }).pdbText)).toBe(
      digest(reconstructPdb(rnase, tables).pdbText),
    );
  });

  it('repairs D-configured side chains (T4 lysozyme Trp21) unless chirality checking is off', () => {
    const t4l = tracePdb([['A', T4L_CHIRAL_CA]]);
    const improper = (pdb: string, resSeq: number) => {
      const m = atomMap(pdb);
      const [ca, n, c, cb] = ['CA', 'N', 'C', 'CB'].map((k) => m.get(`A${resSeq}:${k}`)!);
      const v = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
      const [p, q, r] = [v(n, ca), v(c, ca), v(cb, ca)];
      return p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0]) + p[2] * (q[0] * r[1] - q[1] * r[0]);
    };
    const fixed = reconstructPdb(t4l, tables);
    expect(fixed.chains[0].stats.dAminoAcidsFixed).toBe(1);
    const raw = reconstructPdb(t4l, tables, { checkChirality: false }).pdbText;
    expect(Math.sign(improper(fixed.pdbText, 21))).toBe(-Math.sign(improper(raw, 21)));
    const lHand = Math.sign(improper(fixed.pdbText, 16));
    for (const r of fixed.chains[0].residues.filter((x) => x.resName !== 'GLY')) {
      expect(Math.sign(improper(fixed.pdbText, r.resSeq))).toBe(lHand);
    }
  });

  it('adds amide H 1.081 Å from N on non-proline residues without moving heavy atoms', () => {
    const plain = reconstructPdb(TRP, tables, { optimizeHbonds: true });
    const withH = reconstructPdb(TRP, tables, { optimizeHbonds: true, outputHydrogens: true });
    expect(plain.pdbText.split('\n').some((l) => / H {3}/.test(l.substring(12, 16)))).toBe(false);
    const hRes = withH.chains[0].residues.filter((r) => r.atoms.some((a) => a.name === 'H'));
    const expectedRes = withH.chains[0].residues.slice(1).filter((r) => r.resName !== 'PRO');
    expect(hRes.map((r) => r.resSeq)).toEqual(expectedRes.map((r) => r.resSeq));
    const m = atomMap(withH.pdbText);
    for (const r of hRes) {
      // 1.081 Å, up to the 3-decimal PDB rounding of both atoms
      expect(Math.abs(dist(m.get(`A${r.resSeq}:H`)!, m.get(`A${r.resSeq}:N`)!) - 1.081)).toBeLessThan(0.0015);
    }
    expect(withH.atomCount).toBe(plain.atomCount + hRes.length);
    const heavy = (pdb: string) => pdbAtoms(pdb).filter((a) => a.name !== 'H').map((a) => `${a.resSeq}${a.name}${a.xyz}`);
    expect(heavy(withH.pdbText)).toEqual(heavy(plain.pdbText));
  });

  it('threads a new sequence onto the trace before side-chain reconstruction', () => {
    const threaded = reconstructPdb(TRP, tables, { sequence: THREAD_SEQ });
    expect(threaded.sequence).toBe(THREAD_SEQ);
    expect(threaded.chains[0].residues.map((r) => r.resName).slice(0, 3)).toEqual(['MET', 'LYS', 'VAL']);
    expect(threaded.chains[0].residues[17].atoms.map((a) => a.name)).toContain('CZ2'); // P18W
    expect(digest(reconstructPdb(TRP, tables, { sequence: 'NLYIQWLKDGGPSSGRPPPS' }).pdbText)).toBe(DIGESTS.trpDefault);
    // Multi-chain sequences: '/'-separated, per-chain array, or concatenated.
    const insSeq = reconstructPdb(INSULIN, tables).chains.map((c) => c.residues.map((r) => r.resName));
    expect(insSeq.map((c) => c.length)).toEqual([21, 30]);
    const polyA = ['A'.repeat(21), 'A'.repeat(30)];
    for (const seq of [polyA.join('/'), polyA, polyA.join('')]) {
      const r = reconstructPdb(INSULIN, tables, { sequence: seq });
      expect(r.sequence).toBe('A'.repeat(51));
      expect(r.atomCount).toBe(51 * 5 + 2);
    }
    expect(() => reconstructPdb(TRP, tables, { sequence: 'ACDE' })).toThrow(/4 residues but the trace has 20/);
    expect(() => reconstructPdb(TRP, tables, { sequence: 'X'.repeat(20) })).toThrow(/unsupported residue code 'X'/);
    expect(() => reconstructPdb(INSULIN, tables, { sequence: ['A'.repeat(21)] })).toThrow(/1 chain/);
  });

  it('rebuilds every chain independently while preserving chain IDs and residue numbering', () => {
    const res = reconstructPdb(INSULIN, tables);
    expect(res.chains.map((c) => [c.chainId, c.residues.length])).toEqual([
      ['A', 21],
      ['B', 30],
    ]);
    expect(res.chains[1].residues.map((r) => r.resSeq)).toEqual(Array.from({ length: 30 }, (_, i) => i + 1));
    const atoms = pdbAtoms(res.pdbText);
    expect(atoms.filter((a) => a.name === 'OXT').map((a) => `${a.chain}${a.resSeq}`)).toEqual(['A21', 'B30']);
    expect(res.pdbText.split('\n').filter((l) => l.startsWith('TER'))).toHaveLength(2);
    // Same chain rebuilt alone gives identical coordinates.
    const chainB = atomMap(tracePdb([['B', INSULIN_B_CA]]));
    const alone = atomMap(reconstructPdb(tracePdb([['B', INSULIN_B_CA]]), tables).pdbText);
    const inComplex = atomMap(res.pdbText);
    for (const [k, v] of alone) expect(inComplex.get(k)).toEqual(v);
    expect(chainB.size).toBe(30);
  });

  it('centers the model on its heavy-atom centroid without changing internal geometry', () => {
    const plain = pdbAtoms(reconstructPdb(TRP, tables).pdbText);
    const centered = pdbAtoms(reconstructPdb(TRP, tables, { centerChain: true }).pdbText);
    const centroid = [0, 1, 2].map((k) => centered.reduce((s, a) => s + a.xyz[k], 0) / centered.length);
    for (const c of centroid) expect(Math.abs(c)).toBeLessThan(0.002);
    expect(dist(centered[3].xyz, centered[40].xyz)).toBeCloseTo(dist(plain[3].xyz, plain[40].xyz), 2);
  });

  it('reads only the first model and first altloc, skips hydrogens and waters, and preserves ligands', () => {
    const base = TRP.split('\n').filter((l) => l.startsWith('ATOM'));
    const alt = base[2].substring(0, 16) + 'B' + base[2].substring(17, 30) + '  50.000  50.000  50.000' + base[2].substring(54);
    const messy = [
      'MODEL        1',
      ...base.slice(0, 3).map((l, i) => (i === 2 ? l.substring(0, 16) + 'A' + l.substring(17) : l)),
      alt,
      'ATOM     99  H   TYR A   3      10.000  10.000  10.000  1.00  0.00           H',
      ...base.slice(3),
      'HETATM  500  O   HOH A 101       1.000   2.000   3.000  1.00  0.00           O',
      'HETATM  501  C1  NAG A 201       4.000   5.000   6.000  1.00  0.00           C',
      'ENDMDL',
      'MODEL        2',
      'ATOM      1  CA  GLY A 500       0.000   0.000   0.000  1.00  0.00           C',
      'ENDMDL',
    ].join('\n');
    const res = reconstructPdb(messy, tables);
    expect(digest(res.pdbText)).toBe(DIGESTS.trpDefault);
    expect(res.warnings.join(' ')).toMatch(/Skipped 1 water HETATM/);
    expect(res.pdbText).toContain('HETATM');
    expect(res.pdbText).toContain('NAG A 201');
    expect(res.pdbText).not.toContain('HOH');
    expect(() => reconstructPdb('HETATM    1  O   HOH A   1       0.000   0.000   0.000\nEND\n', tables)).toThrow(/no protein residues/);
    expect(() => reconstructPdb(TRP, tables, { maxCaShift: -1 })).toThrow(/maxCaShift/);
  });
});

// ─── Plugin runner ───────────────────────────────────────────────────────────

describe('PULCHRA plugin runner', () => {
  const opts = { tablesBuffer: readFileSync(resolve(process.cwd(), 'static/vendor/pulchra/pulchra-tables.bin')) };

  it('emits an all-atom StructureArtifact with coordinates, geometry and lineage', async () => {
    const parent = structureFromPdb(TRP, 'genie2-trp', { designKey: 'design-0', designIndex: 0, bfactorSemantics: 'plddt-0-100' });
    const events: string[] = [];
    const out = await runPulchraCpu(pulchraNode(), { structure: parent }, (ev) => events.push(ev.phase ?? ''), opts);
    const s = out.structure as StructureArtifact;
    expect(s.kind).toBe('StructureArtifact');
    expect(s.sourceModelId).toBe('pulchra');
    expect(s.parentStructureId).toBe('genie2-trp');
    expect(s.atomGranularity).toBe('all-atom');
    expect(s.length).toBe(20);
    expect(s.sequence).toBe('NLYIQWLKDGGPSSGRPPPS');
    expect(s.caCoords).toHaveLength(60);
    expect(s.backbone4Coords).toHaveLength(240);
    expect(s.backbone4Coords.every(Number.isFinite)).toBe(true);
    expect(s.geometry.chainBreaks).toBe(0);
    expect(s.geometry.length).toBe(20);
    expect(s.provenance).toEqual({});
    expect(s.designKey).toBe('design-0');
    expect(s.bfactorSemantics).toBe('plddt-0-100');
    expect(digest(s.pdbText)).toBe(DIGESTS.trpDefault);
    expect(events).toContain('load');
    expect(events).toContain('rebuild');
    expect(s.collection).toBeUndefined();
  });

  it('maps node params onto PULCHRA stages and reports backbone4 granularity without side chains', async () => {
    const node = pulchraNode({ reconstructSidechains: false });
    expect(pulchraOptionsFromParams(node)).toMatchObject({ reconstructSidechains: false, reconstructBackbone: true, maxCaShift: 0.5 });
    const out = await runPulchraCpu(node, { structure: structureFromPdb(TRP, 'trp') }, undefined, opts);
    expect((out.structure as StructureArtifact).atomGranularity).toBe('backbone4');
    expect(digest((out.structure as StructureArtifact).pdbText)).toBe(DIGESTS.trpBackboneOnly);
    const hb = await runPulchraCpu(pulchraNode({ optimizeHbonds: true }), { structure: structureFromPdb(TRP, 'trp') }, undefined, opts);
    expect(digest((hb.structure as StructureArtifact).pdbText)).toBe(DIGESTS.trpHbonds);
  });

  it('fans out over structure collections, one all-atom model per member', async () => {
    const a = structureFromPdb(TRP, 'trp', { designKey: 'k0' });
    const b = structureFromPdb(UBQ, 'ubq', { designKey: 'k1' });
    const collection: StructureArtifact = {
      ...a,
      id: 'batch',
      collection: {
        kind: 'Collection', schema: 'denovo.collection/v1', id: 'col', collectionKind: 'independent-candidates',
        requestedCount: 2, complete: true, items: [a, b], keys: ['k0', 'k1'],
      },
    };
    const out = (await runPulchraCpu(pulchraNode(), { structure: collection }, undefined, opts)).structure as StructureArtifact;
    expect(out.collection?.items.map((it) => it.parentStructureId)).toEqual(['trp', 'ubq']);
    expect(out.collection?.keys).toEqual(['k0', 'k1']);
    expect(out.collection?.items.map((it) => digest(it.pdbText))).toEqual([DIGESTS.trpDefault, DIGESTS.ubqDefault]);
    expect(out.collection?.items.every((it) => !it.collection)).toBe(true);
    expect(digest(out.pdbText)).toBe(DIGESTS.trpDefault);
  });

  it('threads each SequenceSet candidate onto its parent backbone', async () => {
    const trp = structureFromPdb(TRP, 'trp', { designKey: 'k0' });
    const ubq = structureFromPdb(UBQ, 'ubq', { designKey: 'k1' });
    const cand = (id: string, sequence: string, extra: Record<string, unknown> = {}) => ({
      id, sequence, score: null, globalScore: null, sourceModelId: 'web-protein-mpnn', ...extra,
    });
    const seqs: SequenceSet = {
      kind: 'SequenceSet', id: 'seqs', parentStructureId: 'trp', sourceModelId: 'web-protein-mpnn',
      candidates: [
        cand('s0', 'NLYIQWLKDGGPSSGRPPPS', { sourceStructureId: 'trp' }),
        cand('s1', THREAD_SEQ, { designKey: 'k0' }),
        cand('s2', 'A'.repeat(76), { sourceStructureId: 'ubq' }),
      ],
    };
    const both: StructureArtifact = {
      ...trp,
      id: 'batch',
      collection: {
        kind: 'Collection', schema: 'denovo.collection/v1', id: 'col', collectionKind: 'independent-candidates',
        requestedCount: 2, complete: true, items: [trp, ubq],
      },
    };
    const out = (await runPulchraCpu(pulchraNode(), { structure: both, sequences: seqs }, undefined, opts)).structure as StructureArtifact;
    const items = out.collection!.items;
    expect(items.map((it) => it.parentStructureId)).toEqual(['trp', 'trp', 'ubq']);
    expect(items.map((it) => digest(it.pdbText)).slice(0, 2)).toEqual([DIGESTS.trpDefault, DIGESTS.trpThreaded]);
    expect(items[2].sequence).toBe('A'.repeat(76));
    expect(new Set(out.collection!.keys).size).toBe(3);
    expect(new Set(items.map((it) => it.id)).size).toBe(3);

    const orphan: SequenceSet = { ...seqs, parentStructureId: 'missing', candidates: [cand('s9', THREAD_SEQ)] };
    await expect(runPulchraCpu(pulchraNode(), { structure: both, sequences: orphan }, undefined, opts)).rejects.toThrow(/cannot match sequence s9/);
    const tooShort: SequenceSet = { ...seqs, candidates: [cand('s3', 'ACDE', { sourceStructureId: 'trp' })] };
    await expect(runPulchraCpu(pulchraNode(), { structure: trp, sequences: tooShort }, undefined, opts)).rejects.toThrow(/4 residues/);
  });

  it('rejects a missing structure input and honours abort signals', async () => {
    await expect(runPulchraCpu(pulchraNode(), {}, undefined, opts)).rejects.toThrow(/requires an upstream/);
    const ac = new AbortController();
    ac.abort();
    await expect(
      runPulchraCpu(pulchraNode(), { structure: structureFromPdb(TRP, 'trp') }, undefined, { ...opts, signal: ac.signal }),
    ).rejects.toThrow();
  });
});

// ─── Registration ────────────────────────────────────────────────────────────

describe('PULCHRA registration (native plugin #28)', () => {
  it('declares a cpu-ts structure→structure manifest with the PULCHRA stage toggles', () => {
    const m = ADAPTER_REGISTRY.pulchra;
    expect(m).toMatchObject({ id: 'pulchra', runtime: 'cpu-ts', stage: 'score', estimatedVramMB: 0 });
    expect(m.inputs).toEqual({
      structure: { type: 'StructureArtifact' },
      sequences: { type: 'SequenceSet', required: false },
    });
    expect(m.outputs).toEqual({ structure: { type: 'StructureArtifact' } });
    expect(defaultParamsForAdapter('pulchra')).toEqual({
      reconstructBackbone: true,
      reconstructSidechains: true,
      optimizeCa: false,
      optimizeHbonds: false,
      fixClashes: true,
      maxCaShift: 0.5,
      detectCisProline: false,
      outputHydrogens: false,
    });
  });

  it('registers in the plugin registry, flow DSL, share codes and catalog without a model card', () => {
    const reg = createDefaultPluginRegistry();
    expect(reg.getPlugin('pulchra')).toBe(pulchraPlugin);
    expect(reg.listPlugins()).toHaveLength(31);
    expect(reg.resolveShortCode('pu')).toBe('pulchra');
    expect(ADAPTER_TO_SHORT.pulchra).toBe('pu');
    expect(SHORT_TO_ADAPTER.pu).toBe('pulchra');
    expect(getModelCardForAdapter('pulchra')).toBeNull();
    expect(CATALOG_ITEMS.find((c) => c.id === 'pulchra')?.group).toBe('Refinement / Backmapping');
    expect(formatNodeParamSummary('pulchra', { optimizeHbonds: true })).toBe('All-atom · H-bonds');

    const flow = parseFlowDsl('src:p:pdbId=1ubq~pu:optimizeHbonds=true');
    expect(flow.ok).toBe(true);
    const node = flow.graph!.nodes.find((n) => n.adapterId === 'pulchra')!;
    expect(node.params.optimizeHbonds).toBe(true);
    expect(flow.graph!.edges).toContainEqual(expect.objectContaining({ toNode: node.id, toPort: 'structure', fromPort: 'structure' }));
    expect(parseFlowDsl('src:p:pdbId=1ubq~allatom').graph!.nodes.map((n) => n.adapterId)).toContain('pulchra');
  });

  it('estimates one model per upstream structure, or per sequence when threading', () => {
    const base = { params: {}, upstreamProfiles: { structure: { itemCount: 4 }, sequences: { itemCount: 12 } } };
    const est = (connected: string[]) =>
      pulchraPlugin.preflight({ ...base, connectedInputs: new Set(connected) } as unknown as Parameters<typeof pulchraPlugin.preflight>[0]);
    expect(est(['structure']).outputCardinality).toEqual({ structure: 4 });
    expect(est(['structure', 'sequences']).outputCardinality).toEqual({ structure: 12 });
    expect(est(['structure']).estimatedDownloadMB).toBeGreaterThan(1);
  });
});

// ─── QA regression tests (P0 & P1 fixes) ─────────────────────────────────────

describe('PULCHRA QA regressions (P0 & P1)', () => {
  const runnerOpts = { tablesBuffer: readFileSync(resolve(process.cwd(), 'static/vendor/pulchra/pulchra-tables.bin')) };

  it('P0-1: optimizeHbonds rebuilds the clash grid before fixClashes and leaves zero heavy-atom clashes < 1.6 Å', () => {
    for (const input of [TRP, UBQ]) {
      const res = reconstructPdb(input, tables, { optimizeHbonds: true });
      const stats = res.chains[0].stats;
      expect(stats.clashesBefore).toBeGreaterThan(0);
      expect(stats.clashesAfter).toBe(0);
      const atoms = res.chains[0].residues.flatMap((r, ri) => r.atoms.map((a) => ({ ...a, ri })));
      const bb = new Set(['N', 'CA', 'C', 'O', 'OXT']);
      let minNonBonded = Infinity;
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const a = atoms[i], b = atoms[j];
          if (a.ri === b.ri) continue;
          if (Math.abs(a.ri - b.ri) === 1 && ((a.name === 'C' && b.name === 'N') || (a.name === 'N' && b.name === 'C'))) continue;
          if (bb.has(a.name) && bb.has(b.name)) continue;
          const d = dist([a.x, a.y, a.z], [b.x, b.y, b.z]);
          if (d < minNonBonded) minNonBonded = d;
        }
      }
      expect(minNonBonded).toBeGreaterThanOrEqual(1.6);
    }
  });

  it('P0-2: splits chains at Cα–Cα breaks (> 4.5 Å) without bridging gaps or collapsing them during Cα refinement', () => {
    const seg1 = rows(UBIQUITIN_CA).slice(0, 8);
    const seg2 = rows(UBIQUITIN_CA).slice(0, 8);
    // Shift segment 2 so Cα(8)–Cα(15) is exactly 9.0 Å (missing loop 9..14)
    const last1 = seg1[seg1.length - 1].split(/\s+/).slice(-3).map(Number);
    const first2 = seg2[0].split(/\s+/).slice(-3).map(Number);
    const dx = last1[0] + 9.0 - first2[0];
    const dy = last1[1] - first2[1];
    const dz = last1[2] - first2[2];
    const lines: string[] = [];
    let serial = 1;
    seg1.forEach((r, idx) => {
      const [res, num, x, y, z] = r.split(/\s+/);
      const bf = (70 + idx).toFixed(2).padStart(6);
      lines.push(
        `ATOM  ${String(serial++).padStart(5)}  CA  ${res} A${num.padStart(4)}    ` +
          `${Number(x).toFixed(3).padStart(8)}${Number(y).toFixed(3).padStart(8)}${Number(z).toFixed(3).padStart(8)}  1.00${bf}           C`,
      );
    });
    seg2.forEach((r, idx) => {
      const [res, , x, y, z] = r.split(/\s+/);
      const num = String(15 + idx);
      const bf = (85 + idx).toFixed(2).padStart(6);
      lines.push(
        `ATOM  ${String(serial++).padStart(5)}  CA  ${res} A${num.padStart(4)}    ` +
          `${(Number(x) + dx).toFixed(3).padStart(8)}${(Number(y) + dy).toFixed(3).padStart(8)}${(Number(z) + dz).toFixed(3).padStart(8)}  1.00${bf}           C`,
      );
    });
    lines.push('TER', 'END');
    const gappedPdb = lines.join('\n') + '\n';

    for (const opts of [{}, { optimizeCa: true, maxCaShift: 0.5 }] as PulchraOptions[]) {
      const out = reconstructPdb(gappedPdb, tables, opts);
      expect(out.chains).toHaveLength(1);
      expect(out.chains[0].chainId).toBe('A');
      expect(out.chains[0].residues.map((r) => r.resSeq)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22]);
      expect(out.chains[0].residues[0].bfactor).toBeCloseTo(70, 2);
      expect(out.chains[0].residues[8].bfactor).toBeCloseTo(85, 2);
      expect(out.warnings.join(' ')).toMatch(/contiguous segments/i);

      const m = atomMap(out.pdbText);
      const gapCa = dist(m.get('A8:CA')!, m.get('A15:CA')!);
      expect(gapCa).toBeGreaterThan(7.5);
      if (!opts.optimizeCa) expect(gapCa).toBeCloseTo(9.0, 2);
      const gapCN = dist(m.get('A8:C')!, m.get('A15:N')!);
      expect(gapCN).toBeGreaterThan(5.0);
      expect(dist(m.get('A8:CA')!, m.get('A8:C')!)).toBeLessThan(2.0);
      expect(m.has('A8:OXT')).toBe(false);
      expect(m.has('A22:OXT')).toBe(true);
    }
  });

  it('P0-3: preserves L < 5 segments as-is with exact +100 Å translation invariance and provenance warning', async () => {
    const fourResCa = rows(UBIQUITIN_CA).slice(0, 4);
    const makeFourPdb = (shift: number) =>
      fourResCa
        .map((r, i) => {
          const [res, num, x, y, z] = r.split(/\s+/);
          return (
            `ATOM  ${String(i + 1).padStart(5)}  CA  ${res} A${num.padStart(4)}    ` +
            `${(Number(x) + shift).toFixed(3).padStart(8)}${(Number(y) + shift).toFixed(3).padStart(8)}${(Number(z) + shift).toFixed(3).padStart(8)}  1.00  0.00           C`
          );
        })
        .join('\n') + '\nTER\nEND\n';

    const baseRes = reconstructPdb(makeFourPdb(0), tables, { optimizeCa: true });
    const shiftedRes = reconstructPdb(makeFourPdb(100), tables, { optimizeCa: true });
    expect(baseRes.warnings.join(' ')).toMatch(/L < 5.*preserved as-is/);
    expect(baseRes.chains[0].residues.flatMap((r) => r.atoms.map((a) => a.name))).toEqual(['CA', 'CA', 'CA', 'CA']);

    const pairDists = (res: ReturnType<typeof reconstructPdb>) => {
      const atoms = res.chains[0].residues.flatMap((r) => r.atoms.map((a) => [a.x, a.y, a.z] as const));
      const d: number[] = [];
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) d.push(dist(atoms[i], atoms[j]));
      }
      return d;
    };
    const d0 = pairDists(baseRes);
    const d100 = pairDists(shiftedRes);
    const maxShiftDiff = Math.max(...d0.map((v, i) => Math.abs(v - d100[i])));
    expect(maxShiftDiff).toBeLessThan(1e-6);

    const artifactOut = await runPulchraCpu(pulchraNode(), { structure: structureFromPdb(makeFourPdb(0), 'short4') }, undefined, runnerOpts);
    const s = artifactOut.structure as StructureArtifact;
    expect(s.atomGranularity).toBe('ca-only');
    expect(s.provenance?.warnings?.join(' ')).toMatch(/L < 5/);
  });

  it('P1-4: derives accurate atomGranularity and omits H-bonds/+H in canvas summary when reconstructBackbone is false', async () => {
    const trpStruct = structureFromPdb(TRP, 'trp');
    const caOnlyOut = (
      await runPulchraCpu(
        pulchraNode({ reconstructBackbone: false, reconstructSidechains: false }),
        { structure: trpStruct },
        undefined,
        runnerOpts,
      )
    ).structure as StructureArtifact;
    expect(caOnlyOut.atomGranularity).toBe('ca-only');

    // Sidechains on a CA-only input without N/C/O backbone must NOT claim all-atom
    const scOnlyOut = (
      await runPulchraCpu(
        pulchraNode({ reconstructBackbone: false, reconstructSidechains: true }),
        { structure: trpStruct },
        undefined,
        runnerOpts,
      )
    ).structure as StructureArtifact;
    expect(scOnlyOut.atomGranularity).toBe('ca-only');

    // When input already has N/CA/C/O backbone, reconstructBackbone: false + sidechains yields all-atom
    const bb4Struct = (
      await runPulchraCpu(
        pulchraNode({ reconstructSidechains: false }),
        { structure: trpStruct },
        undefined,
        runnerOpts,
      )
    ).structure as StructureArtifact;
    const rebuiltFromBb4 = (
      await runPulchraCpu(
        pulchraNode({ reconstructBackbone: false, reconstructSidechains: true }),
        { structure: bb4Struct },
        undefined,
        runnerOpts,
      )
    ).structure as StructureArtifact;
    expect(rebuiltFromBb4.atomGranularity).toBe('all-atom');

    const summary = pulchraPlugin.display!.formatCanvasSummary!;
    expect(summary({ reconstructBackbone: false, optimizeHbonds: true, outputHydrogens: true })).toBe('Sidechains');
    expect(summary({ reconstructBackbone: false, reconstructSidechains: false, optimizeHbonds: true, outputHydrogens: true })).toBe('Cα only');
    expect(formatNodeParamSummary('pulchra', { reconstructBackbone: false, optimizeHbonds: true, outputHydrogens: true })).toBe('Sidechains');
    expect(summary({ reconstructBackbone: true, optimizeHbonds: true, outputHydrogens: true })).toBe('All-atom · H-bonds · +H');
  });

  it('P1-5: parses 2-char MSE SE atoms without truncating to E and never emits both SE and SD', () => {
    const mseCa = rows(TRP_CAGE_CA)
      .slice(0, 6)
      .map((r, i) => {
        const [res, num, x, y, z] = r.split(/\s+/);
        if (i === 2) {
          const [nx, ny, nz] = [Number(x), Number(y), Number(z)];
          return [
            `HETATM    3  N   MSE A   3    ${(nx - 1.2).toFixed(3).padStart(8)}${ny.toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           N`,
            `HETATM    4  CA  MSE A   3    ${nx.toFixed(3).padStart(8)}${ny.toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           C`,
            `HETATM    5  C   MSE A   3    ${(nx + 1.2).toFixed(3).padStart(8)}${ny.toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           C`,
            `HETATM    6  O   MSE A   3    ${(nx + 1.5).toFixed(3).padStart(8)}${(ny + 1.1).toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           O`,
            `HETATM    7  CB  MSE A   3    ${nx.toFixed(3).padStart(8)}${(ny - 1.5).toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           C`,
            `HETATM    8  CG  MSE A   3    ${nx.toFixed(3).padStart(8)}${(ny - 2.9).toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           C`,
            `HETATM    9 SE   MSE A   3    ${(nx + 1.1).toFixed(3).padStart(8)}${(ny - 4.1).toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00          SE`,
            `HETATM   10  CE  MSE A   3    ${(nx + 2.4).toFixed(3).padStart(8)}${(ny - 3.6).toFixed(3).padStart(8)}${nz.toFixed(3).padStart(8)}  1.00 20.00           C`,
          ].join('\n');
        }
        return `ATOM  ${String(i + 1).padStart(5)}  CA  ${res} A${num.padStart(4)}    ${Number(x).toFixed(3).padStart(8)}${Number(y).toFixed(3).padStart(8)}${Number(z).toFixed(3).padStart(8)}  1.00  0.00           C`;
      })
      .join('\n') + '\nTER\nEND\n';

    // When sidechains are NOT rebuilt, both left- and right-aligned SE are kept intact as 'SE' (not 'E').
    for (const seLine of [
      `HETATM    9 SE   MSE A   3    `,
      `HETATM    9  SE  MSE A   3    `,
    ]) {
      const variant = mseCa.replace(`HETATM    9 SE   MSE A   3    `, seLine);
      const bbOnly = reconstructPdb(variant, tables, { reconstructSidechains: false });
      const mseResKept = bbOnly.chains[0].residues[2];
      expect(mseResKept.atoms.map((a) => a.name)).toContain('SE');
      expect(mseResKept.atoms.map((a) => a.name)).not.toContain('E');
      expect(mseResKept.atoms.find((a) => a.name === 'SE')?.element).toBe('SE');
      expect(bbOnly.pdbText).toMatch(/^ATOM\s+\d+\sSE\s+MSE A\s+3.*\sSE$/m);
    }

    // When sidechains ARE rebuilt (both preserve and optimizeCa), SE maps cleanly to SD without duplicating SE+SD.
    for (const opts of [{}, { optimizeCa: true }] as PulchraOptions[]) {
      const rebuilt = reconstructPdb(mseCa, tables, opts);
      const mseAtoms = rebuilt.chains[0].residues[2].atoms.map((a) => a.name);
      expect(mseAtoms).toContain('SD');
      expect(mseAtoms).not.toContain('SE');
      expect(mseAtoms).not.toContain('E');
    }
  });

  it('P1-6: keeps B-only and C-only altloc residues aligned with parsePdbBackbone', () => {
    const base = TRP.split('\n').filter((l) => l.startsWith('ATOM'));
    const bOnly = base.map((l, i) => (i === 2 ? l.substring(0, 16) + 'B' + l.substring(17) : l)).join('\n') + '\nTER\nEND\n';
    const cOnly = base.map((l, i) => (i === 4 ? l.substring(0, 16) + 'C' + l.substring(17) : l)).join('\n') + '\nTER\nEND\n';
    // Out-of-order B then A on residue 3: A must supersede B without creating a duplicate residue
    const bThenA = [
      ...base.slice(0, 2),
      base[2].substring(0, 16) + 'B' + base[2].substring(17, 30) + '  50.000  50.000  50.000' + base[2].substring(54),
      base[2].substring(0, 16) + 'A' + base[2].substring(17),
      ...base.slice(3),
      'TER',
      'END',
    ].join('\n') + '\n';

    const resB = reconstructPdb(bOnly, tables);
    expect(resB.residueCount).toBe(20);
    expect(resB.residueCount).toBe(parsePdbBackbone(bOnly).nResidues);
    expect(resB.sequence).toBe('NLYIQWLKDGGPSSGRPPPS');

    const resC = reconstructPdb(cOnly, tables);
    expect(resC.residueCount).toBe(20);
    expect(resC.sequence).toBe('NLYIQWLKDGGPSSGRPPPS');

    const resBA = reconstructPdb(bThenA, tables);
    expect(resBA.residueCount).toBe(20);
    expect(digest(resBA.pdbText)).toBe(DIGESTS.trpDefault);
  });

  it('P1-7: preserves non-water HETATM ligands/ions, nucleic chains, molecularContext, and HETATM SEP length parity', async () => {
    const holoInput = [
      ...TRP.split('\n').filter((l) => l.startsWith('ATOM')),
      'TER',
      'ATOM    101  P    DG D   1      20.000  20.000  20.000  1.00  0.00           P',
      "ATOM    102  O5'  DG D   1      21.000  20.500  20.000  1.00  0.00           O",
      'TER',
      'HETATM  103 ZN    ZN Z   1      10.000  10.000  10.000  1.00  0.00          ZN',
      'HETATM  104  C1  SAM Z   2      12.000  12.000  12.000  1.00  0.00           C',
      'HETATM  105  O   HOH W   1      30.000  30.000  30.000  1.00  0.00           O',
      'END',
    ].join('\n');

    const parent = structureFromPdb(holoInput, 'holo-trp');
    const out = (await runPulchraCpu(pulchraNode(), { structure: parent }, undefined, runnerOpts)).structure as StructureArtifact;
    expect(out.length).toBe(20);
    expect(out.pdbText).toContain('HETATM');
    expect(out.pdbText).toContain('ZN Z   1');
    expect(out.pdbText).toContain('SAM Z   2');
    expect(out.pdbText).toContain('DG D   1');
    expect(out.pdbText).not.toContain('HOH');
    expect(out.diagnostics?.hasLigands).toBe(true);
    expect(out.diagnostics?.ligandNames).toEqual(expect.arrayContaining(['ZN', 'SAM']));
    expect(out.diagnostics?.hasNucleicAcids).toBe(true);
    expect(out.diagnostics?.nucleicAcidChains).toEqual(['D']);
    expect(out.molecularContext?.ligands).toEqual(expect.arrayContaining(['ZN', 'SAM']));
    expect(out.molecularContext?.nucleicAcidChains).toEqual(['D']);

    // Modified-residue HETATM SEP must not cause a protein length mismatch vs parsePdbBackbone
    const baseLines = TRP.split('\n').filter((l) => l.startsWith('ATOM'));
    const withSep = [
      ...baseLines.slice(0, 5),
      'HETATM   50  N   SEP A   6      -0.716  -0.631  -0.993  1.00  0.00           N',
      'HETATM   51  CA  SEP A   6      -0.716  -0.631  -0.993  1.00  0.00           C',
      'HETATM   52  P   SEP A   6       1.000   1.000   1.000  1.00  0.00           P',
      ...baseLines.slice(6),
      'TER',
      'END',
    ].join('\n');
    const parsedIn = parsePdbBackbone(withSep);
    const sepParent = structureFromPdb(withSep, 'sep-trp');
    const sepOut = (await runPulchraCpu(pulchraNode(), { structure: sepParent }, undefined, runnerOpts)).structure as StructureArtifact;
    expect(sepOut.length).toBe(parsedIn.nResidues);
    expect(sepOut.sequence).toBe(parsedIn.sequence);
    expect(sepOut.pdbText).toContain('HETATM');
    expect(sepOut.pdbText).toContain('SEP A   6');
  });

  it('P1-8: aborts pre-aborted and stalled pulchra-tables.bin fetches promptly without poisoning subsequent callers', async () => {
    const body = readFileSync(resolve(process.cwd(), 'static/vendor/pulchra/pulchra-tables.bin'));
    const testUrl = '/test/abortable-pulchra-tables.bin';

    // 1. Pre-aborted signal never invokes fetchFn
    const preAc = new AbortController();
    preAc.abort(new DOMException('Pre-aborted', 'AbortError'));
    const neverCalled = vi.fn(async () => new Response(new Uint8Array(body)));
    await expect(
      loadPulchraTables({ url: testUrl, fetchFn: neverCalled as unknown as typeof fetch, signal: preAc.signal }),
    ).rejects.toThrow(/Pre-aborted|abort/i);
    await expect(
      runPulchraCpu(pulchraNode(), { structure: structureFromPdb(TRP, 'trp') }, undefined, {
        url: testUrl,
        fetchFn: neverCalled as unknown as typeof fetch,
        signal: preAc.signal,
      }),
    ).rejects.toThrow(/Pre-aborted|abort/i);
    expect(neverCalled).not.toHaveBeenCalled();

    // 2. Stalled fetch aborts promptly when signal fires and propagates abort to fetchFn's signal
    let capturedSignal: AbortSignal | undefined;
    const stalledFetch = vi.fn(
      (_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>(() => {
          capturedSignal = init?.signal ?? undefined;
        }),
    );
    const ac = new AbortController();
    const pending = loadPulchraTables({
      url: testUrl,
      fetchFn: stalledFetch as unknown as typeof fetch,
      signal: ac.signal,
    });
    expect(stalledFetch).toHaveBeenCalledTimes(1);
    expect(capturedSignal?.aborted).toBe(false);
    ac.abort(new DOMException('Cancelled by user', 'AbortError'));
    await expect(pending).rejects.toThrow(/Cancelled by user|abort/i);
    expect(capturedSignal?.aborted).toBe(true);

    // 3. Subsequent non-aborted caller succeeds cleanly (cache was not poisoned)
    const okFetch = vi.fn(async () => new Response(new Uint8Array(body)));
    const recovered = await loadPulchraTables({ url: testUrl, fetchFn: okFetch as unknown as typeof fetch });
    expect(okFetch).toHaveBeenCalledTimes(1);
    expect(recovered.rotPos.length).toBe(19897);

    // 4. Concurrent aborted caller does not cancel a healthy sibling caller on a fresh URL
    const concUrl = '/test/concurrent-pulchra-tables.bin';
    let releaseSibling!: (res: Response) => void;
    const siblingFetch = vi.fn(
      () =>
        new Promise<Response>((resolveFn) => {
          releaseSibling = resolveFn;
        }),
    );
    const ac1 = new AbortController();
    const p1 = loadPulchraTables({ url: concUrl, fetchFn: siblingFetch as unknown as typeof fetch, signal: ac1.signal });
    const p2 = loadPulchraTables({ url: concUrl, fetchFn: siblingFetch as unknown as typeof fetch });
    ac1.abort();
    await expect(p1).rejects.toThrow(/abort/i);
    releaseSibling(new Response(new Uint8Array(body)));
    expect((await p2).rotPos.length).toBe(19897);
  });
});
