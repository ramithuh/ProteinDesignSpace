# Included Models, Licenses & Credits

`denovo.run` stands on the shoulders of extraordinary open-source protein science and WebGPU engineering work. Please cite the original authors and upstream tools when using any model from this workbench.

---

## Included Models & Browser Engines

All neural inference executes locally in sequential, VRAM-isolated Dedicated Web Workers with OPFS / CacheStorage weight persistence. Weights stream on demand from Hugging Face and are never bundled into the static build.

| Stage | Model | Capabilities | Weights / Port | Upstream License |
| --- | --- | --- | --- | --- |
| **Generation** | **RFD4-Proteina** | All-atom sequence-structure co-design, motif, binder, partial diffusion | [`sokrypton/rfd4-proteina-webgpu`](https://github.com/sokrypton/rfd4-proteina-webgpu) · [`nvidia/RFD4-Proteina`](https://huggingface.co/nvidia/RFD4-Proteina) | NVIDIA Open / MIT |
| **Generation** | **La-Proteina (LD1 + AE1)** | All-atom latent diffusion (LD1) + atom37/sequence decoder (AE1) | [`ramithuh/la-proteina-webgpu`](https://huggingface.co/ramithuh/la-proteina-webgpu) | Apache-2.0 / NVIDIA Open |
| **Generation** | **Proteina v1.2 (200M)** | Scalable 200M pair-biased transformer flow matching (Cα backbones) | [`ramithuh/proteina-webgpu`](https://huggingface.co/ramithuh/proteina-webgpu) | NVIDIA Research (NC) |
| **Generation** | **SiD-Protein (16-Step)** | Score-identity distilled 200M Proteina generator (16-step Cα sampling) | [`ramithuh/sid-protein-webgpu`](https://huggingface.co/ramithuh/sid-protein-webgpu) | SiD-Protein / Proteina |
| **Generation** | **Genie 3** | Fast 23.3M SE(3) DDIM backbone diffusion + SequenceNet | [`ramithuh/genie3-webgpu`](https://huggingface.co/ramithuh/genie3-webgpu) | Apache-2.0 |
| **Generation** | **Genie 2** | Compact 15.7M Cα diffusion & soft motif scaffolding | [`ramithuh/genie2-webgpu`](https://huggingface.co/ramithuh/genie2-webgpu) | Apache-2.0 |
| **Generation** | **RFdiffusion (Base)** | Original SE(3) backbone diffusion, motif contigs, Cn/Dn symmetry | [`ramithuh/rfdiffusion-webgpu`](https://huggingface.co/ramithuh/rfdiffusion-webgpu) | BSD |
| **Generation** | **FrameFlow** | Fast 16.8M SE(3) rigid-frame flow matching (5-atom N/CA/C/O/CB) | [`ramithuh/frameflow-webgpu`](https://huggingface.co/ramithuh/frameflow-webgpu) | MIT / Apache-2.0 |
| **Generation** | **FoldFlow 2 (ReFT)** | SE(3) flow matching + frozen ESM-2 650M (unconditional & seq-conditioned) | [`ramithuh/foldflow2-webgpu`](https://huggingface.co/ramithuh/foldflow2-webgpu) | CC BY-NC 4.0 / MIT |
| **Generation** | **BoltzGen-1** | All-atom unconditional, binder & motif diffusion (`diverse`, `adherence`) | [`ramithuh/boltzgen-webgpu`](https://huggingface.co/ramithuh/boltzgen-webgpu) | MIT |
| **Generation** | **ProtPardelle-1c** | All-atom & backbone EDM diffusion, tip-atom motif, MiniMPNN co-design | [`ramithuh/protpardelle-webgpu`](https://huggingface.co/ramithuh/protpardelle-webgpu) | MIT / CC-BY-4.0 |
| **Sequence** | **ProteinMPNN / SolubleMPNN / Cα** | Backbone-conditioned inverse folding (ONNX WebGPU / WASM) | [`MurrellGroup/webProteinMPNN`](https://github.com/MurrellGroup/webProteinMPNN) | MIT |
| **Sequence** | **LigandMPNN** | Ligand-, metal- & nucleic-acid-aware sequence design | [`ramithuh/ligandmpnn-webgpu`](https://huggingface.co/ramithuh/ligandmpnn-webgpu) | MIT |
| **Sequence** | **LASErMPNN** | Ligand-aware sequence & all-atom sidechain rotamer co-design | [`ramithuh/lasermpnn-webgpu`](https://huggingface.co/ramithuh/lasermpnn-webgpu) | MIT |
| **Sequence** | **AntiFold & ESM-IF1** | Antibody CDR inverse folding & GVP-Transformer inverse folding | [`ramithuh/antifold-esmif1-webgpu`](https://huggingface.co/ramithuh/antifold-esmif1-webgpu) | BSD-3 / MIT |
| **Sequence** | **EvoDiff-Seq (38M)** | MSA-free order-agnostic discrete sequence diffusion & inpainting | [`ramithuh/evodiff-webgpu`](https://huggingface.co/ramithuh/evodiff-webgpu) | MIT / CC-BY-4.0 |
| **Prediction** | **LocalFold Suite** | AF2, AF2-Multimer, AF3*, Boltz-2, RF3, Protenix-v2, OpenFold 3, IntelliFold 2, OpenDDE, Mol1, ESMFold2 | [`sokrypton/localfold`](https://github.com/sokrypton/localfold) | Beer-Ware (code) · Apache-2.0 / MIT / CC-BY-4.0 (*AF3 non-commercial) |
| **Property** | **ThermoMPNN** | Full L×20 site-saturation ΔΔG stability scan & 3D B-factor painting | [`ramithuh/thermompnn-webgpu`](https://huggingface.co/ramithuh/thermompnn-webgpu) | MIT |
| **Property** | **ESM-C 300M** | Zero-shot L×20 LLR fitness heatmap, pseudo-perplexity & masked fill | [`ramithuh/esmc-webgpu`](https://huggingface.co/ramithuh/esmc-webgpu) | MIT |
| **Property** | **Dyna-1** | Millisecond conformational exchange & flexibility prediction | [`ramithuh/dyna1-webgpu`](https://huggingface.co/ramithuh/dyna1-webgpu) | MIT / ESM-3 Open |
| **Evaluation** | **Designability / 3D Compare** | Kabsch SVD `scRMSD`, `scTM`, Motif RMSD, pLDDT, ipSAE, up to 6-way 3D superposition | Built-in CPU | MIT |

---

## Credits, Upstream Repositories & Citations

### Pioneering Browser WebGPU Engines
- **AlphaFold2 WebGPU** — Martin Steinegger: [`martin-steinegger/alphafold2-webgpu`](https://github.com/martin-steinegger/alphafold2-webgpu)
- **LocalFold & RFD4-Proteina WebGPU** — Sergey Ovchinnikov et al.: [`sokrypton/localfold`](https://github.com/sokrypton/localfold) · [`sokrypton/rfd4-proteina-webgpu`](https://github.com/sokrypton/rfd4-proteina-webgpu)
- **webProteinMPNN** — Ben Murrell Lab: [`MurrellGroup/webProteinMPNN`](https://github.com/MurrellGroup/webProteinMPNN)

### Generative, Inverse-Folding, Folding & Property Models
- **RFD4-Proteina** — Baker Lab / Institute for Protein Design & NVIDIA · [`nvidia/RFD4-Proteina`](https://huggingface.co/nvidia/RFD4-Proteina)
- **RFdiffusion** — Watson et al. (2023), *De novo design of protein structure and function with RFdiffusion*, Nature 620, 1089–1100 · [`RosettaCommons/RFdiffusion`](https://github.com/RosettaCommons/RFdiffusion)
- **La-Proteina & Proteina** — Geffner et al. (2025), *Proteina: Scaling Flow-based Protein Structure Generative Models* & *La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching* · [`NVIDIA-BioNeMo/la-proteina`](https://github.com/NVIDIA-BioNeMo/la-proteina) · [`NVIDIA-BioNeMo/proteina`](https://github.com/NVIDIA-BioNeMo/proteina)
- **SiD-Protein (SPID)** — Xie et al. (2025), *SPID: Distilled Protein Backbone Generation*, arXiv:2510.03095 · [`LY-Xie/SiD_Protein`](https://github.com/LY-Xie/SiD_Protein)
- **Genie 2 & Genie 3** — Lin et al. (2024, 2026), *Genie 2* (arXiv:2405.15489) & *Genie 3* (bioRxiv) · [`aqlaboratory/genie2`](https://github.com/aqlaboratory/genie2) · [`aqlaboratory/genie3`](https://github.com/aqlaboratory/genie3)
- **FrameFlow** — Yim et al. (2023), *Fast protein backbone generation with SE(3) flow matching*, arXiv:2310.05297 · [`microsoft/protein-frame-flow`](https://github.com/microsoft/protein-frame-flow)
- **FoldFlow 2 & ESM-2** — Huguet et al. (2024), *Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Generation*, NeurIPS 2024 · [`DreamFold/FoldFlow`](https://github.com/DreamFold/FoldFlow); frozen 650M encoder from Lin et al. (2023), *Evolutionary-scale prediction of atomic-level protein structure with a language model*, Science (Meta FAIR) · [`facebookresearch/esm`](https://github.com/facebookresearch/esm)
- **BoltzGen** — Stärk et al. (2025), *BoltzGen: Toward Universal Binder Design* · [`HannesStark/boltzgen`](https://github.com/HannesStark/boltzgen)
- **ProtPardelle-1c** — Chu et al. (2024), *An all-atom protein generative model*, PNAS · [`ProteinDesignLab/protpardelle-1c`](https://github.com/ProteinDesignLab/protpardelle-1c)
- **EvoDiff** — Alamdari et al. (2023), *Protein generation with evolutionary diffusion: sequence is all you need*, bioRxiv · [`microsoft/evodiff`](https://github.com/microsoft/evodiff)
- **ProteinMPNN & SolubleMPNN** — Dauparas et al. (2022), *Robust deep learning–based protein sequence design using ProteinMPNN*, Science · [`dauparas/ProteinMPNN`](https://github.com/dauparas/ProteinMPNN); Goverde et al. (2024), *Computational design of soluble and functional membrane protein analogues*, Nature (SolubleMPNN)
- **LigandMPNN** — Dauparas et al. (2025), *Atomic context-conditioned protein sequence design using LigandMPNN*, Nature Methods · [`dauparas/LigandMPNN`](https://github.com/dauparas/LigandMPNN)
- **LASErMPNN** — Polizzi Lab · [`polizzilab/LASErMPNN`](https://github.com/polizzilab/LASErMPNN)
- **AntiFold & ESM-IF1** — Høie et al. (2024), *AntiFold: Improved antibody structure-based design using inverse folding*, Bioinformatics · [`oxpig/AntiFold`](https://github.com/oxpig/AntiFold); Hsu et al. (2022), *Learning inverse folding from millions of predicted structures*, ICML · [`facebookresearch/esm`](https://github.com/facebookresearch/esm)
- **ThermoMPNN** — Dieckhaus et al. (2024), *Transfer learning to leverage larger datasets for improving prediction of protein stability changes*, PNAS · [`Kuhlman-Lab/ThermoMPNN`](https://github.com/Kuhlman-Lab/ThermoMPNN)
- **ESM-C & ESM-3** — EvolutionaryScale / Chan Zuckerberg Biohub · [`Biohub/esm`](https://github.com/Biohub/esm)
- **Dyna-1** — Wayment-Steele et al. (2026), *Learning millisecond protein dynamics from what is missing in NMR spectra*, Nature · [`WaymentSteeleLab/Dyna-1`](https://github.com/WaymentSteeleLab/Dyna-1)
- **Structure Predictors (via LocalFold)**:
  - **AlphaFold 2 & AlphaFold 3** — Jumper et al. (2021), *Nature*; Abramson et al. (2024), *Nature* (Google DeepMind)
  - **Boltz-2** — Passaro et al. (2025) (MIT / Recursion)
  - **RoseTTAFold 3** — Baker Lab / Institute for Protein Design
  - **Protenix v2** — ByteDance
  - **OpenBind-0 / OpenFold 3** — OpenFold Consortium
  - **OpenDDE** — Aureka Research
  - **Mol-1** — Molence Bio
  - **IntelliFold 2** — intelligenAI
  - **ESMFold2** — Chan Zuckerberg Biohub / EvolutionaryScale (folding trunk & ESM-C) with confidence head trained by **Synthyra**

### Evaluation Metrics, Data Services & Upstream Components
- **ipSAE** — Dunbrack (2025), *What's wrong with AlphaFold's ipTM score and how to fix it*, bioRxiv
- **TM-score & Kabsch Superposition** — Zhang & Skolnick (2004), *Proteins*; Kabsch (1976), *Acta Crystallographica*
- **Optional MSA Server** — MMseqs2 / ColabFold API (Mirdita et al., 2022, *ColabFold: making protein folding accessible to all*, Nature Methods) — disabled by default and only used after explicit user confirmation
- **RCSB PDB** — live experimental structure (PDB/mmCIF) and CCD ligand definition lookups
- **Upstream Model Code Components** (see vendored `THIRD_PARTY_LICENSES`) — [`aqlaboratory/openfold`](https://github.com/aqlaboratory/openfold), [`lucidrains/alphafold3-pytorch`](https://github.com/lucidrains/alphafold3-pytorch), [`MattMcPartlon/protein-docking`](https://github.com/MattMcPartlon/protein-docking), and NVIDIA SE(3)-Transformer

### Frontend & Visualization Infrastructure
- [**Mol\***](https://molstar.org/) (3D molecular viewer) · [**XYFlow (`@xyflow/svelte`)**](https://svelteflow.dev/) (visual node-graph canvas) · [**SvelteKit 5**](https://svelte.dev/) · [**ONNX Runtime Web**](https://onnxruntime.ai/)
