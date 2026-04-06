"""
BPR ↔ The Well Validation Bridge
==================================

Connects PolymathicAI's "The Well" physics simulation datasets
(https://github.com/polymathicai/the_well) to BPR prediction validators.

Each sub-module maps one Well dataset to one or more BPR predictions,
extracting measurable statistics and returning Result objects compatible
with the main validate_all_theories.py harness.

Dataset → BPR prediction mapping
---------------------------------
gray_scott_reaction_diffusion  → P23.1  Turing pattern wavelength scaling
acoustic_scattering_inclusions → Check1 Boundary Laplacian l(l+1) spectrum
rayleigh_benard                → P4.x   Phase-transition critical exponent β
active_matter                  → P10.x  Kuramoto flocking coherence onset
MHD_64                         → P7.1   Alfvén wave propagation isotropy
"""

from .loaders import WellNotAvailable, load_well_frames  # noqa: F401
