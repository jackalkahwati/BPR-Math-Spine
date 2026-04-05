import { motion } from 'motion/react';
import { ImageWithFallback } from './figma/ImageWithFallback';

export function ResonantPrimeSubstrate() {
  return (
    <section className="w-full py-16 px-6 bg-secondary/30 relative overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5">
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1666112835145-d79fc3f2e008?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxxdWFudHVtJTIwcGh5c2ljc3xlbnwxfHx8fDE3NzU0MzA3NTV8MA&ixlib=rb-4.1.0&q=80&w=1080"
          alt="Background pattern"
          className="w-full h-full object-cover"
        />
      </div>

      <div className="max-w-4xl mx-auto relative z-10">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="mb-6"
        >
          Resonant Prime Substrate (RPST)
        </motion.h2>

        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            viewport={{ once: true }}
            className="bg-background border border-border rounded-lg p-6"
          >
            <h3 className="mb-3">Prime Modular Structure</h3>
            <p className="text-muted-foreground leading-relaxed">
              Arithmetic on ℤₚ is used so discrete dynamics are well behaved. The framework uses{' '}
              <strong>no zero divisors</strong>—this is framed as a design axiom for the discrete model,
              ensuring clean mathematical properties at the fundamental level.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            viewport={{ once: true }}
            className="bg-background border border-border rounded-lg p-6"
          >
            <h3 className="mb-3">Resonance</h3>
            <p className="text-muted-foreground leading-relaxed">
              Stable phases and collective modes across the substrate and boundaries are what pick out{' '}
              <strong>effective constants</strong> and <strong>observable patterns</strong> (rather than
              tuning them by hand). The name "resonance" reflects how physics emerges from constructive
              interference and phase-locking in the discrete system.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            viewport={{ once: true }}
            className="bg-accent/50 border border-primary/20 rounded-lg p-6"
          >
            <p className="text-sm text-muted-foreground italic">
              <strong>Key insight:</strong> BPR is a hypothesis and mathematical framework, not established
              physics. It is built to be auditable: equations, code, tests, benchmarks, and explicit
              falsification criteria. Some predictions are claimed as BPR-unique; many matches are consistency
              checks where the Standard Model or GR already win—the tables in the repository separate these.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
