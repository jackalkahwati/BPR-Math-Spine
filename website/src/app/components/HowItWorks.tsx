import { ArrowRight } from 'lucide-react';
import { motion } from 'motion/react';

export function HowItWorks() {
  const steps = [
    {
      number: "1",
      title: "Substrate (discrete)",
      description: "A large lattice where each site carries variables in prime modular arithmetic (e.g., (qᵢ,πᵢ) ∈ ℤₚ × ℤₚ). Dynamics follow a Hamiltonian with neighbor interactions—this is the starting assumption of the framework, not derived from standard QFT."
    },
    {
      number: "2",
      title: "Boundaries matter",
      description: "Coherent regions have edges. BPR puts a real phase field φ on the boundary Σ = ∂M of bulk spacetime M. The idea is holographic: boundary degrees of freedom encode or drive bulk excitations."
    },
    {
      number: "3",
      title: "Boundary dynamics → effective bulk response",
      description: "The phase obeys boundary field equations (a boundary Laplacian/sourced equation), coupled to metric perturbations Δgμν. Boundary phase gradients ↔ curvature-like response in the bulk, via a controlled action principle."
    },
    {
      number: "4",
      title: "Extra terms = where information and biology enter",
      description: "The action includes information-theoretic and biological/fractal terms (IIT-style information integration, consciousness-coupling channels). In the math they are explicit terms in the variational story, not hidden tuning knobs."
    },
    {
      number: "5",
      title: "Concrete numbers instead of infinite free knobs",
      description: "The framework's selling point: push as much as possible into derivations from substrate parameters, with documented places that remain inputs, framework fits, or open questions."
    },
    {
      number: "6",
      title: "Sharp experimental handles",
      description: "Includes Casimir-style deviations (a power-law correction with a stated exponent δ) so precision force/cavity/MEMS experiments can support or falsify specific BPR claims. Phonon/collective modes are the channel where tiny couplings might become accessible."
    }
  ];

  return (
    <section className="w-full py-16 px-6 bg-background">
      <div className="max-w-5xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="mb-4"
        >
          How It Works
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          viewport={{ once: true }}
          className="text-muted-foreground mb-12 max-w-3xl"
        >
          The logical chain: <strong>discrete substrate → boundary phase → coupled bulk geometry → specific
          corrections and cross-domain predictions → compare to experiment</strong>
        </motion.p>

        <div className="space-y-6">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <div className="flex gap-6 items-start">
                <div className="flex-shrink-0 w-12 h-12 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-mono">
                  {step.number}
                </div>
                <div className="flex-1">
                  <h3 className="mb-2">{step.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{step.description}</p>
                </div>
              </div>
              {index < steps.length - 1 && (
                <div className="ml-6 mt-4 mb-2">
                  <ArrowRight className="w-5 h-5 text-primary" />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
