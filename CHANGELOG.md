# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-10

### Added
- Initial public implementation of `biohyperglow`.
- Single-branch mixed-membership model with activity scaling and Bernoulli-Poisson link.
- Ontology-guided semantic branch with hierarchical concept clustering over MONDO concepts.
- Fixed training-graph structural target profile computation with component bridge hubs.
- Unconstrained association-only free branch.
- Mapping-aware degree gate: `alpha_d = m_d * (kappa / (kappa + k_d))` with reference configuration `kappa = 10.0`.
- Association intensity mixture decoder: `lambda_{dg} = alpha_d * lambda^S_{dg} + (1 - alpha_d) * lambda^F_{dg}`.
- Exact additive hypernode path explanation (Proposition 1), proving equivalence to normalized concatenated embeddings.
- Validation-based classification threshold selection maximizing binary F1.
- Complete CLI (`train`, `predict`, `explain`, `evaluate`, `demo`).
- Self-contained synthetic benchmark generation for quickstart and testing.
- Comprehensive test suite and CI workflows supporting Python 3.10, 3.11, and 3.12.
- Release workflow supporting PyPI Trusted Publishing.
