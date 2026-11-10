# BioHyperGLOW

[![CI](https://github.com/wajdidhifli/BioHyperGLOW/actions/workflows/ci.yml/badge.svg)](https://github.com/wajdidhifli/BioHyperGLOW/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/biohyperglow.svg)](https://pypi.org/project/biohyperglow/)
[![Python versions](https://img.shields.io/pypi/pyversions/biohyperglow.svg)](https://pypi.org/project/biohyperglow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BioHyperGLOW** is an adaptive dual-space representation learning framework for interpretable disease-gene association prediction. It reconciles uneven network topology with incomplete biological ontology annotation, while providing an exact, intrinsic decomposition of every predicted link strength without relying on surrogate explainers or gradient approximations.

---

## Key Features

1. **Adaptive Dual-Space Architecture**:
   - **Semantic Branch**: Combines soft memberships over hierarchical clusters of [MONDO](https://mondo.monarchinitiative.org/) concepts with fixed graph structural targets derived from the training graph.
   - **Association-Only Structural Branch**: Learns unconstrained latent association groups directly from network connectivity.

2. **Evidence-Dependent Degree Gate ($\kappa=10$)**:
   - Disease-specific intensity gate $\alpha_d = m_d \frac{\kappa}{\kappa + k_d}$, where $m_d \in \{0, 1\}$ indicates accepted ontology mapping and $k_d$ is the disease's positive training degree.
   - Sparsely connected or training-isolated diseases with ontology mappings leverage semantic guidance ($\alpha_d \to 1$); well-connected diseases rely primarily on observed association structure ($\alpha_d \to 0$); unmapped diseases rely exclusively on the free branch ($\alpha_d = 0$).

3. **Intensity-Level Bernoulli-Poisson Fusion**:
   - Combines predicted Poisson association intensities $\lambda_{dg} = \alpha_d \lambda^S_{dg} + (1 - \alpha_d) \lambda^F_{dg}$, rather than post-link probability averaging. Marginal association probability is decoded as $p_{dg} = 1 - \exp(-\lambda_{dg})$.

4. **Exact Intrinsic Explanations (Proposition 1)**:
   - Decomposes every predicted intensity into an exact sum of $2H$ nonnegative gene-hypernode-disease path contributions:
     $$\lambda_{dg} = \sum_{h=1}^H \phi^S_{dgh} + \sum_{h=1}^H \phi^F_{dgh}$$
   - Provably equivalent to evaluating normalized concatenated representations on a $2H$-coordinate simplex ($\sum \phi_j = \lambda_{dg}$ within machine precision).
   - *Scientific Note*: Semantic hypernodes correspond to MONDO concept clusters. Latent structural hypernodes capture topological association regularities and are not presented as established biological mechanisms.

5. **Validation-Based Selection**:
   - Checkpoints selected by maximum validation Average Precision (AP); classification cutoffs selected by maximum validation binary $F_1$.

---

## Installation

### From PyPI
```bash
pip install biohyperglow
```

### From Source
```bash
git clone https://github.com/wajdidhifli/BioHyperGLOW.git
cd BioHyperGLOW
pip install -e ".[dev]"
```

### Hardware Support
BioHyperGLOW runs out-of-the-box on CPU across Linux, macOS, and Windows. Apple Silicon (MPS) and NVIDIA GPU (CUDA) acceleration can be enabled using `--device mps` or `--device cuda`.

---

## Quick Start

### 1. Run the Built-In Demo (CLI)
Verify installation and run a self-contained synthetic benchmark end-to-end:
```bash
biohyperglow demo --out-dir my_demo --epochs 25 --H 8
```

### 2. Python API Quick Start
```python
from biohyperglow import train_dual_space, BioHyperGLOWModel

# 1. Train dual-space model from network associations and ontology
model = train_dual_space(
    edges_csv="data/associations.csv",
    ontology_obo="ontology/mondo.obo",
    H=512,            # 512 coordinates per branch (1024 total)
    kappa=10.0,       # Reference degree gate scale
    epochs=100,
    device="cpu",     # 'cpu', 'cuda', or 'mps'
)

# 2. Save and reload checkpoint
model.save("checkpoints/biohyperglow_model.npz")
loaded_model = BioHyperGLOWModel.load("checkpoints/biohyperglow_model.npz")

# 3. Predict association strength for candidate pairs
preds = loaded_model.predict(
    diseases=["C0006142"],  # UMLS CUI for Breast Carcinoma
    genes=["101"]           # Entrez gene ID
)
print("Predicted Intensity:", preds["intensity"][0])
print("Predicted Probability:", preds["probability"][0])

# 4. Rank candidate genes for a query disease
candidates = loaded_model.rank_genes("C0006142", exclude_genes=["101"])
print("Top candidate:", candidates[0])

# 5. Explain association with exact path attribution (Proposition 1)
explanation = loaded_model.explain("C0006142", "101", top_k=5)
print(explanation.summary(top_k=5))
```

---

## Input Formats

### 1. Disease-Gene Association Network (`edges.csv`)
A CSV file containing at least two columns: `disease` and `gene`.
```csv
disease,gene
C0006142,672
C0006142,675
C0009402,7157
```
- **Disease Identifiers**: UMLS Concept Unique Identifiers (CUIs), e.g., `C0006142` or `UMLS:C0006142`.
- **Gene Identifiers**: NCBI Entrez IDs (e.g. `672`) or official gene symbols (e.g. `BRCA1`).

### 2. Disease Ontology (`ontology.obo`)
Standard Open Biological Ontologies (OBO) format file (e.g. MONDO release):
```text
[Term]
id: MONDO:0007254
name: breast cancer
is_a: MONDO:0004992 ! cancer
xref: UMLS:C0006142
```
Diseases are automatically matched to MONDO concepts via their UMLS cross-references (`xref: UMLS:...`). Unmatched or ambiguous diseases are assigned to the unassigned fallback channel.

---

## Command-Line Interface (CLI)

BioHyperGLOW exposes its workflows through `biohyperglow`:

### Train Model
```bash
biohyperglow train \
    --edges data/associations.csv \
    --ontology ontology/mondo.obo \
    --out checkpoints/model.npz \
    --H 512 \
    --kappa 10.0 \
    --epochs 100 \
    --device cpu
```

### Predict on Candidate Pairs
```bash
biohyperglow predict \
    --model checkpoints/model.npz \
    --pairs candidate_pairs.csv \
    --out predictions.csv
```

### Generate Intrinsic Explanation
```bash
biohyperglow explain \
    --model checkpoints/model.npz \
    --disease C0006142 \
    --gene 672 \
    --top-k 10 \
    --out explanation.json
```

### Evaluate Test Performance
```bash
biohyperglow evaluate \
    --model checkpoints/model.npz \
    --test-pairs test_pairs.csv \
    --out evaluation_metrics.json
```

---

## Obtaining Public External Data

To train on publication benchmarks:
1. **DisGeNET**: Download disease-gene associations from the [DisGeNET downloads portal](https://www.disgenet.org/downloads).
2. **MONDO Disease Ontology**: Download the latest OBO release from the [Monarch Initiative MONDO repository](https://github.com/monarch-initiative/mondo/releases) (e.g. `mondo.obo`).

---

## Running Tests

Run the test suite including numerical equivalence, Proposition 1 checks, and integration tests:
```bash
pytest -v
```

---

## Release and Publishing Instructions (For Maintainers)

BioHyperGLOW uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) via GitHub Actions.

### 1. TestPyPI Verification
To test distribution packaging on TestPyPI:
1. Configure a GitHub Actions environment named `testpypi` with repository URL `https://test.pypi.org/p/biohyperglow`.
2. Trigger the `Publish to PyPI` workflow manually via GitHub Actions dispatch, selecting target `testpypi`.
3. Verify test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps biohyperglow
   ```

### 2. Production PyPI Release
1. Configure a GitHub Actions environment named `pypi` linked to your PyPI project `biohyperglow`.
2. Create and push a signed git release tag:
   ```bash
   git tag -a v0.1.0 -m "BioHyperGLOW release 0.1.0"
   git push origin v0.1.0
   ```
3. Publish a release in GitHub. The `.github/workflows/release.yml` workflow will automatically build wheel and sdist distributions, run Twine validation, and publish to PyPI.

---

## Citation

If you use BioHyperGLOW in your research, please cite:

```bibtex
@article{dhifli2026biohyperglow,
  title={BioHyperGLOW: Adaptive Dual-Space Embeddings for Interpretable Disease-Gene Association Prediction},
  author={Dhifli, Wajdi and Karabadji, Nour Elislem and Marcq, Gautier and Elati, Mohamed and Assi, Ali},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
