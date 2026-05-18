# EBMol: Generating Physically Consistent Molecules with Energy-Based Models

Official implementation of **EBMol**, a time-unconditional energy-based model for 3D molecular generation.

 
EBMol learns an atom-additive scalar energy landscape whose local minima correspond to stable molecular structures. It combines two components:
- **Restoring Field Matching (RFM):** A simulation-free, flow-matching-inspired training objective that shapes the energy landscape with data points as local minima.
- **Mirror-Langevin Algorithm (MLA):** A sampling framework that unifies Euclidean coordinate updates and simplex-constrained atom-type updates, combined with parallel tempering for inference-time compute scaling.

EBMol is the first energy-based model for 3D molecular generation to achieve state-of-the-art performance.

> **Paper:** *Generating Physically Consistent Molecules with Energy-Based Models*


## Installation

```bash
git clone https://github.com/griesbchr/EBMol.git
cd EBMol
conda env create -f ebmol.yaml
conda activate ebmol
```

Preprocessed datasets, trained model checkpoints, and generated samples are available as `.tar.xz` archives on Google Drive:
 
**[Download link](https://drive.google.com/drive/folders/18RXIRDgD-GRrGUR2Y7eABTRF7oNqjh1L?usp=drive_link)**
 
After downloading, extract into the project root:
 
```bash
tar -xJf qm9.tar.xz        # → data/qm9/
tar -xJf geom.tar.xz        # → data/geom/
tar -xJf experiments.tar.xz  # → experiments/
tar -xJf samples.tar.xz      # → samples/
```
 
---
 
## Training
 
```bash
python train_ebm.py --dataset_name qm9
python train_ebm.py --dataset_name geom
```
 
Training takes approximately 4 days on a single RTX 3090 Ti (QM9) or 3 days on 2× L40 (GEOM-Drugs).
 
---
## Sampling and Evaluation
 
```bash
python sample_and_analyze_qm9.py
python sample_and_analyze_geom.py
```
 
These scripts handle both sampling and evaluation. To configure:
 
- **Sampling from a trained model:** Set the `exp_name` variable in the script to select the experiment directory under `experiments/`.
- **Evaluating existing samples:** Set the `load_samples_name` variable to point to an `.xyz` file under `samples/`. This skips sampling and runs evaluation only.
- **GEOM-Drugs Revisited evaluation:** In `run_geomr_evals.sh` set the `SDF_PATH` variable to the samples to be evaluated then run bash file. 
---

## Acknowledgements
 
This codebase builds on several open-source projects including:
 
- [E3 Diffusion for Molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules) — EGNN implementation, evaluation code, and code snippets
- [GEOM-Drugs 3D Generation Evaluation](https://github.com/isayevlab/geom-drugs-3dgen-evaluation) — revised GEOM-Drugs evaluation protocol
- [MolFM](https://github.com/GenSI-THUAIR/MolFM) — SE(3)+permutation optimal transport solver

We thank the authors of these projects for making their code publicly available.

