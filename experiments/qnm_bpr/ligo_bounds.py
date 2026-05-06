"""
Published LIGO/Virgo Tests-of-GR ringdown constraints.

Numbers are 90% CI fractional bounds on (delta_omega_220 / omega_220)
and (delta_tau_220 / tau_220) from peer-reviewed publications. These
are NOT pulled from a live API; they are the static published values
referenced in each entry's `source` field. Update if newer catalogs
supersede.

The combined catalog rows are the most informative for testing a
mass-INDEPENDENT predicted shift like BPR's, because they stack many
events across the BH mass spectrum.
"""

# Each entry: (label, M_solar, |delta_f/f| 90% CI bound, |delta_tau/tau| 90% CI, source)
LIGO_TGR_RINGDOWN = [
    # Single-event bounds
    {
        "label": "GW150914",
        "M_solar": 68.0,            # remnant mass
        "df_over_f_90": 0.16,        # fractional bound
        "dtau_over_tau_90": 0.34,
        "source": "Abbott et al. (2016) PRL 116, 221101; Brito, Buonanno, Raymond (2018) PRD 98, 084038",
    },
    {
        "label": "GW170104",
        "M_solar": 49.0,
        "df_over_f_90": 0.20,
        "dtau_over_tau_90": 0.50,
        "source": "Carullo et al. (2019) PRD 99, 123029",
    },
    {
        "label": "GW190521",
        "M_solar": 142.0,
        "df_over_f_90": 0.24,
        "dtau_over_tau_90": 0.60,
        "source": "Abbott et al. (2020) PRL 125, 101102; Abbott et al. (2021) PRD 103, 122002 (TGR-O3a)",
    },
    # Combined catalog bound (most informative for a universal shift)
    {
        "label": "GWTC-3 combined (TGR-O3 ringdown)",
        "M_solar": None,   # mass-stacked
        "df_over_f_90": 0.04,
        "dtau_over_tau_90": 0.16,
        "source": "Abbott et al. (2021) PRD 103, 122002 (O3a); Abbott et al. (2024) PRD 109, 022001 (O3-final TGR)",
    },
]

# Projected 3G / space-based detector sensitivities (single-event,
# loud-source). These are Fisher-matrix forecasts, not measurements.
PROJECTED_SENSITIVITY = [
    {
        "detector": "Cosmic Explorer (loud BBH ~1000 Mpc)",
        "df_over_f_90": 1e-3,
        "dtau_over_tau_90": 1e-2,
        "source": "Berti et al. (2016) PRD 94, 104024; Carullo et al. (2018) PRD 98, 104020",
    },
    {
        "detector": "Einstein Telescope (loud BBH)",
        "df_over_f_90": 1e-3,
        "dtau_over_tau_90": 1e-2,
        "source": "Maggiore et al. (2020) JCAP 03, 050",
    },
    {
        "detector": "LISA SMBHB (10^6 Msun, z=1)",
        "df_over_f_90": 1e-5,
        "dtau_over_tau_90": 1e-4,
        "source": "Berti, Buonanno, Will (2005) PRD 71, 084025; Berti, Sesana et al. (2016) PRL 117, 101102",
    },
    {
        "detector": "CE+ET stacked catalog (Fisher-forecast 3G)",
        "df_over_f_90": 1e-4,
        "dtau_over_tau_90": 1e-3,
        "source": "Brito, Buonanno, Raymond (2018) PRD 98, 084038; Maselli et al. (2020) PRD 102, 064056",
    },
]

# Note on combined-catalog sensitivity to a UNIVERSAL shift:
#
# A mass-independent fractional shift Δω/ω = δ stacks linearly across N
# events with effective error ~ σ_single / sqrt(N). With ~ 30 high-SNR
# BBH events in O1-O3 and σ_single ~ 0.1, the catalog floor on a
# universal δ is ~ 0.1 / sqrt(30) ≈ 0.018. Future O4-O5 should drive
# this below 0.01. 3G stacked catalogs of ~ 10^4-10^5 events should
# reach 10^-4 to 10^-5.
