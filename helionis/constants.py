"""Physical constants used by the Helionis screening model."""

from __future__ import annotations

E_CHARGE_C = 1.602176634e-19
MEV_TO_J = 1.0e6 * E_CHARGE_C
KEV_TO_J = 1.0e3 * E_CHARGE_C
MU_0 = 1.25663706212e-6

MODEL_LABEL = "order_of_magnitude_trade_study"

# Bremsstrahlung coefficient for a simple fully ionized plasma proxy:
# P_brem ~= C * Z_eff * n_e * n_i * sqrt(T_e[eV]) [W/m^3].
BREMSSTRAHLUNG_COEFF = 5.35e-37

# Shielding proxy: tonnes per MW of neutron power. This is intentionally a
# comparative architecture metric, not a mechanical shielding design.
DEFAULT_BASE_SHIELDING_TONNES = 1.5
DEFAULT_SHIELDING_TONNES_PER_NEUTRON_MW = 4.0
