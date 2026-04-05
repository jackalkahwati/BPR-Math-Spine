import { motion } from 'motion/react';
import { ImageWithFallback } from './figma/ImageWithFallback';

export function WhatIsBPR() {
  return (
    <section className="w-full py-16 px-6 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="order-2 md:order-1 relative rounded-lg overflow-hidden shadow-2xl"
          >
            <ImageWithFallback
              src="https://images.unsplash.com/photo-1768328591729-176a94d187ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHw1fHxwYXJ0aWNsZSUyMHBoeXNpY3N8ZW58MXx8fHwxNzc1NDMwNzU2fDA&ixlib=rb-4.1.0&q=80&w=1080"
              alt="Particle physics visualization"
              className="w-full h-auto"
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="order-1 md:order-2"
          >
            <h2 className="mb-6">What is BPR?</h2>
            <div className="space-y-4 text-foreground/90 leading-relaxed">
              <p>
                <strong>Boundary Phase Resonance (BPR)</strong> is a research framework that asks:{' '}
                <em>
                  what if the quantities we treat as fundamental fields and couplings are not the bottom layer,
                  but <strong>effective</strong> descriptions of something more discrete?
                </em>
              </p>
              <p>
                In BPR, that layer is a <strong>computational substrate</strong> (in the docs it is called{' '}
                <strong>RPST — Resonant Prime Substrate Theory</strong>): a lattice whose nodes carry discrete
                phase-like degrees of freedom. <strong>Observable physics is then supposed to emerge</strong> when
                you coarse-grain that substrate—especially through <strong>boundaries</strong>—rather than being
                postulated as fields on a continuum from the start.
              </p>
              <p>
                The project's public face is not only equations on paper but a{' '}
                <strong>minimal, runnable codebase</strong> that implements core field equations, conservation
                checks, and a sharp experimental handle (including a <strong>Casimir-style deviation</strong>{' '}
                formulation with a stated critical exponent in the docs).
              </p>
              <p className="bg-accent/30 border-l-4 border-primary px-4 py-3 rounded">
                <strong>The name signals the move:</strong> physics is organized around boundary phase structure
                and how it <strong>resonates</strong> (locks in, interferes, couples) with bulk geometry and other
                sectors.
              </p>
              <p className="text-muted-foreground italic">
                This is a research program under active development; claims are accompanied by derivation
                status, benchmarks, and falsification criteria in the repository.
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
