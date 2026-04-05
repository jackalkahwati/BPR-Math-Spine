import { motion } from 'motion/react';
import { ImageWithFallback } from './figma/ImageWithFallback';

export function TheoryInBrief() {
  return (
    <section className="w-full py-16 px-6 bg-gradient-to-b from-background to-secondary/20">
      <div className="max-w-5xl mx-auto">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-6">Theory in Brief</h2>
            <div className="space-y-4 text-foreground/90 leading-relaxed">
              <p>
                BPR proposes that what we call spacetime and fields may be <strong>large-scale behavior</strong> of
                a <strong>discrete, Hamiltonian substrate</strong>. The bridge to familiar physics is{' '}
                <strong>boundary-centric</strong>: a phase field on the boundary of the bulk encodes excitations
                and couples back to <strong>geometry</strong> and <strong>stress-energy</strong>.
              </p>
              <p>
                From that single variational picture, the project derives or benchmarks{' '}
                <strong>many cross-domain consequences</strong> and names <strong>specific experiments</strong> that
                could <strong>confirm or rule out</strong> key claims—especially where the framework predicts{' '}
                <strong>small but structured deviations</strong> (for example in <strong>Casimir-type</strong>{' '}
                settings) rather than only reproducing known results.
              </p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="relative rounded-lg overflow-hidden shadow-2xl"
          >
            <ImageWithFallback
              src="https://images.unsplash.com/photo-1690683789766-93253a3422c1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHw0fHxxdWFudHVtJTIwcGh5c2ljc3xlbnwxfHx8fDE3NzU0MzA3NTV8MA&ixlib=rb-4.1.0&q=80&w=1080"
              alt="Abstract quantum physics visualization"
              className="w-full h-auto"
            />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
