import { motion } from 'motion/react';

export function PhysicsLandscape() {
  return (
    <section className="w-full py-16 px-6 bg-slate-950">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-8"
        >
          <h2 className="text-3xl md:text-4xl text-white mb-4">
            The BPR Landscape
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Every theory positioned by its energy scale and abstraction level.
            Bottom = measurable predictions. Top = mathematical foundations.
            Every edge is an equation.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="relative rounded-xl overflow-hidden border border-slate-800 shadow-2xl shadow-blue-500/10"
          style={{ aspectRatio: '16/10' }}
        >
          <iframe
            src="/viz/physics-landscape.html"
            className="w-full h-full border-0"
            title="BPR Physics Landscape"
            style={{ minHeight: '600px' }}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-6 flex flex-wrap gap-4 justify-center"
        >
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-amber-500"></span> Substrate
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-purple-500"></span> Spacetime & Gravity
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-blue-500"></span> Quantum & Information
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-green-500"></span> Particles & Forces
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-red-500"></span> Matter & Chemistry
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-pink-500"></span> Life & Consciousness
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="w-3 h-3 rounded-full bg-cyan-400"></span> Predictions
          </div>
        </motion.div>
      </div>
    </section>
  );
}
