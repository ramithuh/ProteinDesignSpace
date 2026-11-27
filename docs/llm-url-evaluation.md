# Evaluation of LLM-Friendly Workflow URL Representations for `denovo.run`

## 1. Problem Statement & Design Constraints

`denovo.run` executes protein design DAGs locally in the browser (WebGPU/WASM) and renders dynamic Open Graph previews on Cloudflare Pages Edge Workers (`_worker.js`). When an LLM without browser or code-execution tools (e.g., ChatGPT, Claude, or API agents reading `/llms.txt`) recommends a runnable protein-design pipeline, it must emit a **single URL** directly in Markdown prose.

Any candidate URL format must satisfy five constraints:
1. **Toolless First-Try Generation:** Constructible from plain ASCII text using `/llms.txt` alone—no `zlib`/`deflate-raw`, no Base64 encoder, no headless browser.
2. **Markdown Autolinker Safety:** Survives bare URL autolinking and `[text](url)` links in GitHub Flavored Markdown (GFM), ChatGPT, and Slack without truncation or escaping bugs (no unencoded `( ) [ ] < >`, no spaces/newlines, no trailing `_ - . , ; :`).
3. **Arbitrary DAG Expressiveness:** Represents any topology `denovo.run` executes—linear chains, multi-branch comparisons (e.g., RFD3 vs. RFD4), fan-out/fan-in merges (`structure-compare`, `designability-scorer` dual-input lineage), explicit `from.port~to.port`, custom node labels, and all 27 `ADAPTER_REGISTRY` models/checkpoints.
4. **Character & Token Efficiency:** Minimizes context-window output tokens and stays well below URL length limits (`MAX_HASH_CHARS = 65,536`, social unfurl limits ~2 KB).
5. **Deterministic Edge & Client Parity:** Produces identical graphs and structured validation diagnostics on both the Cloudflare Edge Worker (`esbuild --platform=browser`, zero `$lib/` aliases) and the browser client (`share.ts`), with zero runtime LLM inference cost.

---

## 2. Candidate Representations Evaluated

| Format | Example Shape | Synthesis Mechanism |
|---|---|---|
| **1. `?flow=<dsl>` (Chosen Primary)** | `?flow=src:p:pdbId=1UBQ~r3:length=80~cmp;src~r:length=80~cmp` | Plain ASCII DAG DSL using `~ : = , ; .` + short codes/aliases + auto-wiring |
| **2. Compact Shorthands (`?g=` / `?p=`)** | `?g=<b64url(deflate-raw(v2_json))>` or `?p=<presetId>&node.k=v` | `?g=` compresses compact tuple JSON via `deflate-raw`; `?p=` diffs against 8 fixed `WORKFLOW_PRESETS` |
| **3. `?recipe=<urlencoded>` (Fallback)** | `?recipe=denovo%3A%20v1%0Anodes%3A...` (YAML) or `?recipe=%7B...%7D` (JSON) | URL-encoded `denovo: v1` YAML recipe or canonical `WorkflowGraph` JSON |
| **4. `?prompt=<natural language>`** | `?prompt=Design%20an%2080-residue%20protein...` | Server/edge LLM re-interpretation on every page load / OG unfurl |

---

## 3. Quantitative Benchmark Across 4 Representative Topologies

All character lengths below are measured directly against `encodeGraphToFlow`, `encodeGraphToHash`, `graphToYamlRecipe`, and `JSON.stringify` in `web/src/lib/engine/flow-dsl.ts` and `web/src/lib/engine/share.ts` (including the 19-character `https://denovo.run/` origin prefix).

### Benchmark Workflows
1. **W1 — Linear 3-step (3 nodes, 2 edges):** `rfd3` (`length=80`, schema default) → `web-protein-mpnn` → `localfold-af2`.
2. **W2 — RFD3 vs. RFD4 branch comparison (4 nodes, 4 edges):** `pdb-input` (`1UBQ`) → parallel `rfd3` (`length=80`) and `rfd4-proteina` (`length=80`) → `structure-compare` (`structures` fan-in).
3. **W3 — Fan-out / Fan-in self-consistency (4 nodes, 4 edges):** `rfd3` (`length=70`) → `web-protein-mpnn` (`numSequences=4`) → `localfold-af2` → `designability-scorer`, with `rfd3.structure` auto-wired to `designability-scorer.structure`.
4. **W4 — Conditioned motif + LASErMPNN + RF3 + explicit ports & custom labels (6 nodes, 7 edges):** `pdb-input` (`1PRW`) → `rfd4-proteina` (`contig=10-20,A16-35,10-20`, `_label=Scaffold`) → `lasermpnn` (`numSequences=2`) → `localfold-af2` (`model=rosettafold3`, `_label=RF3`) → `designability-scorer`, plus `lasermpnn.structure` → `structure-compare.reference` and `localfold-af2.structure` → `structure-compare.structures`.

### Exact URL Character Length Comparison (`https://denovo.run/` prefix included)

| Workflow | `?flow=` (Short / Hand) | `?flow=` (Aliases) | `?flow=` (`encodeGraphToFlow`) | `?g=` (`deflate-raw`) | `?p=` (`preset-diff`) | `?recipe=` (YAML) | `?recipe=` (Compact JSON) | `?prompt=` (NL) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **W1. Linear 3-step** | **41** | 48 | **31** | 142 (4.6×) | N/A | 353 (11.4×) | 1,984 (64.0×) | 165 |
| **W2. RFD3 vs. RFD4 compare** | **78** | 93 | **82** | 221 (2.8×) | N/A | 585 (7.5×) | 2,960 (37.9×) | 198 |
| **W3. Fan-out/in + scRMSD** | **58** | 70 | **63** | 205 (3.5×) | N/A* | 543 (9.4×) | 2,707 (46.7×) | 239 |
| **W4. Motif + LASErMPNN + RF3 + ports** | **209** | 215 | **219** | 349 (1.7×) | N/A | 1,029 (4.9×) | 3,814 (18.2×) | 335 |

*\*When a graph exactly matches one of the 8 built-in `WORKFLOW_PRESETS` (for example `rfd4-mpnn-refold` with `gen.length=90` and `mpnn.numSequences=4`), `?p=` achieves **72 chars** (`https://denovo.run/?p=rfd4-mpnn-refold&gen.length=90&mpnn.numSequences=4`), whereas canonical `encodeGraphToFlow` is **150 chars** because the preset carries explicit non-default `earlyStop=true` and custom labels. However, `?p=` cannot express any custom topology outside those 8 presets.*

### Exact URLs Measured Above (All Verified by Unit Tests)

- **W1 (`?flow=` short, 41 chars):**
  `https://denovo.run/?flow=r3:length=80~m~f`
- **W1 (`?flow=` aliases, 48 chars):**
  `https://denovo.run/?flow=rfd3:length=80~mpnn~af2`
- **W1 (`?flow=` canonical `encodeGraphToFlow`, 31 chars):**
  `https://denovo.run/?flow=r3~m~f`
- **W2 (`?flow=` short, 78 chars):**
  `https://denovo.run/?flow=src:p:pdbId=1UBQ~r3:length=80~cmp;src~r:length=80~cmp`
- **W2 (`?flow=` aliases, 93 chars):**
  `https://denovo.run/?flow=src:pdb:pdbId=1UBQ~rfd3:length=80~compare;src~rfd4:length=80~compare`
- **W3 (`?flow=` short, 58 chars):**
  `https://denovo.run/?flow=r3:length=70~m:numSequences=4~f~s`
- **W3 (`?flow=` aliases, 70 chars):**
  `https://denovo.run/?flow=rfd3:length=70~mpnn:numSequences=4~af2~scorer`
- **W4 (`?flow=` short, 209 chars):**
  `https://denovo.run/?flow=src:p:pdbId=1PRW~motif:r:contig=10-20,A16-35,10-20,_label=Scaffold~pack:la:numSequences=2~fold:f:model=rosettafold3,_label=RF3~score:s;pack.structure~cmp:c.reference;fold.structure~cmp`
- **W4 (`?flow=` aliases, 215 chars):**
  `https://denovo.run/?flow=src:pdb:pdbId=1PRW~motif:rfd4:contig=10-20,A16-35,10-20,_label=Scaffold~pack:lasermpnn:numSequences=2~fold:rf3,_label=RF3~score:scorer;pack.structure~cmp:compare.reference;fold.structure~cmp`

### Structural Token & BPE Subword Efficiency Analysis

Because `tiktoken` is not bundled in `web/node_modules`, we measure exact structural token counts (splitting on DSL/URL delimiters) and analyze documented GPT/Claude BPE behavior (`cl100k_base` / `o200k_base`):

| Metric | W1 `?flow=` (`r3:length=80~m~f`) | W1 `?g=` (`TYw7...`) | W1 `?recipe=` (YAML) |
|---|---:|---:|---:|
| **Total URL characters** | **41** (31 canonical) | 142 | 353 |
| **Query payload chars (after `?`)** | **22** (`flow=r3:length=80~m~f`) | 123 | 334 |
| **Structural identifiers/operators** | 7 (`flow`, `r3`, `length`, `80`, `m`, `f`, delimiters) | High-entropy opaque Base64url string (121 chars) | 22+ YAML keys/JSON tokens + 38 `%XX` escapes |
| **Why BPE magnifies `?flow=` savings** | Lowercase English words (`length`, `pdbId`, `numSequences`) and short alphanumeric tokens (`r3`, `80`, `1UBQ`) match common BPE merges (~3.0–4.5 chars/token), and `~ : = , ; .` add only single-char delimiters. | Random-looking `deflate-raw` + Base64url characters (`TYw7CsAgEETvMvUWEoOF...`) rarely appear in natural-language merges and fragment near ~1.3–1.8 chars/token. | Percent-encoding turns every space (`%20`), newline (`%0A`), `{` (`%7B`), `}` (`%7D`), `"` (`%22`), and `:` (`%3A`) into 3 ASCII characters (`%` + 2 hex digits), multiplying both character count and BPE token splits. |

Thus, `?flow=` is **1.7×–4.6× shorter than `?g=`** and **4.9×–11.4× shorter than `?recipe=` YAML** (and **18×–64× shorter than `?recipe=` JSON**) by raw character count, while keeping human-readable tokens that avoid percent-encoding inflation.

---

## 4. Qualitative & Functional Evaluation Matrix

| Criterion | `?flow=` (Primary DSL) | `?g=` (Compressed DAG) | `?p=` (Preset Diff) | `?recipe=` (YAML/JSON) | `?prompt=` (Natural Lang.) |
|---|---|---|---|---|---|
| **Toolless LLM First-Try Validity** | **High** — Plain ASCII, friendly aliases (`rfd3`, `mpnn`, `af2`, `rf3`, `scorer`, `compare`), auto-wired ports, tolerates `->` and `(...)` | **Impossible** — Requires `deflate-raw` binary compression + Base64url encoding | **Partial** — Works only for the 8 built-in presets; fails on any custom graph | **Low–Medium** — Requires strict JSON quoting inside YAML + exact percent-encoding of spaces/newlines/braces | **Unpredictable** — Valid URL syntax, but topology/params depend on runtime LLM interpretation |
| **GFM / ChatGPT / Slack Autolinker Safety** | **100% Safe** — Primary output emits only `[A-Za-z0-9\-._~:;=,]` + `%XX` for unsafe values; zero `( ) [ ] < > \s`; never ends in `[.,;:!?_~-]` | **Safe (with padding)** — Base64url requires space-padding loop to avoid trailing `_`/`-` truncation | **Safe** — Standard `URLSearchParams` query string | **Fragile if hand-built** — Unencoded spaces, newlines, `{`, `}`, `->` (`>` becomes HTML/Markdown break) break bare autolinkers | **Fragile if hand-built** — Unencoded spaces/parens break bare URLs |
| **Arbitrary DAG Expressiveness** | **100%** — Linear, multi-branch forks, fan-in merges, explicit `from.port~to.port`, custom `_label`, all 27 adapters & 20 LocalFold models | **100%** — Full graph fidelity | **Low** — Fixed to 8 preset topologies only | **100%** — Full graph fidelity | **Ambiguous** — Complex port wiring and exact param values often drift |
| **Determinism & Latency** | **100% deterministic, <1 ms** on Edge Worker and browser | **100% deterministic, <1 ms** (async stream) | **100% deterministic, <1 ms** | **100% deterministic, <1 ms** | **Non-deterministic, 500–3,000 ms** + external AI API cost/failure risk |
| **Error Recovery & Diagnostics** | **Structured** — Per-node/token `{ code, message, token, nodeId }`; strips trailing `.,;!?]}>'"` and unmatched `)` | **Opaque** — Corrupted byte stream yields generic inflate/JSON error | **Moderate** — Unknown preset returns `null` | **Moderate** — Line/JSON error, harder to pinpoint inside `%XX` blobs | **None** — Silent misinterpretation |

---

## 5. Markdown Autolinker Safety Proof

Common Markdown parsers (GitHub Flavored Markdown, CommonMark, ChatGPT, Slack) break bare URLs under four failure modes:
1. **Unencoded parentheses `(` `)`:** Breaks `[label](https://denovo.run/?flow=r3(length=80))` when the first `)` closes the Markdown link prematurely, and GFM strips unmatched trailing `)`.
2. **Brackets & angle brackets `[` `]` `<` `>`:** Terminates `[text](url)` and `<url>` autolinks; `->` in raw HTML/Markdown contexts triggers `&gt;` escaping.
3. **Unencoded whitespace / newlines:** Immediately terminates bare URL autolinking.
4. **Trailing punctuation `. , ; : ! ? _ - ~`:** Autolinkers exclude trailing `.` / `,` / `;` / `_` / `-` as sentence punctuation.

**`?flow=` guarantees:**
- Primary grammar uses `~` (edge), `:` (id/model/params), `=` (key=val), `,` (param separator), `;` (branch separator), `.` (`node.port`).
- `encodeGraphToFlow` emits **zero** `( ) [ ] < > { } " ' \s`, percent-encodes any non-safe character inside values, and percent-encodes any trailing `[._,;-:]` so every generated URL ends in `[A-Za-z0-9]` and passes `isMarkdownSafeFlowUrl`.
- On decode, `parseFlowDsl` **tolerates** LLM habits (`->` as `~`, `node(k=v)` as `node:k=v`, trailing prose punctuation `.,;!?]}>'"` and unmatched trailing `)`) so even imperfect LLM output loads cleanly.

---

## 6. Architecture Decision

We select **`?flow=<dsl>`** as the primary LLM-facing URL format (`web/src/lib/engine/flow-dsl.ts`), while:
- Retaining **`?recipe=`** (`denovo: v1` YAML & JSON) as a supported fallback in `decodeHashToGraph` and `/api/link`,
- Retaining **`?p=` / `?g=` / `#p=` / `#g=`** as canonical stateless compact share links (and `shortUrl` output of `/api/link`), and
- Rejecting **`?prompt=`** as a primary representation due to non-determinism, latency, and Cloudflare Worker AI dependency.
