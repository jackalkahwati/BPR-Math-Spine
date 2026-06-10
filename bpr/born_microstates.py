"""Born rule from Z_p microstate counting — constructive Ω_α(x).

Addresses Gap #1 of doc/conjectures/born_rule.md: the missing constructive
definition of the microstate set Ω_α(x) ("particle of type α at x") in
terms of substrate configurations.

THE CONSTRUCTION
----------------
A substrate path γ from source to detection cell x is a chain of lattice
links, each carrying a Z_p phase increment (the discrete boundary phase
field advances in units of 2π/p). The amplitude of γ is the product of
its link phases — a single element of Z_p.

Define the microstate set for "detection at x":

    Ω(x) = { ordered pairs (γ, γ̄) : both paths terminate at x,
             γ counted with phase +φ(γ), γ̄ with −φ(γ̄) }

i.e. microstates are FORWARD-BACKWARD PATH PAIRS (the substrate analog
of the Keldysh/Schw