# Helionis D-He3 Trade Study Output

Model label: `order_of_magnitude_trade_study`.

| Scenario | Fuel | T keV | Fusion MW | Neutron % | Useful MW | Net MW | Shield t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lunar_infrastructure_dhe3 | D-He3 | 100 | 928.0 | 0.33 | 725.4 | -1418.2 | 13.8 |
| orbital_data_center_dhe3 | D-He3 | 120 | 4098.4 | 0.26 | 3331.4 | -4072.7 | 44.9 |
| compact_space_reactor_dhe3 | D-He3 | 90 | 233.9 | 0.46 | 172.7 | -549.3 | 5.8 |
| terrestrial_demonstrator_dhe3 | D-He3 | 150 | 2773.1 | 0.20 | 2304.9 | -1529.7 | 23.6 |
| dt_reference_power_block | D-T | 15 | 304.4 | 80.16 | 115.7 | -88.7 | 977.4 |

Interpretation notes:

- `net_power_mw` is useful converted power minus simplified radiation and transport losses.
- `gain_proxy` is not Q-plasma; it is useful converted power divided by modeled losses.
- `required_volume_for_target_m3` is infinite when the net power proxy is negative.
- Shielding mass is a comparative neutron-load proxy, not a mechanical shield design.
