import { motion } from 'motion/react';

export function PredictionFlow() {
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
            Two Integers &rarr; All of Physics
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            From substrate parameters <span className="text-blue-400 font-mono">p = 104,729</span> and{' '}
            <span className="text-green-400 font-mono">z = 6</span>, the framework derives 60+
            falsifiable predictions with zero free parameters.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="relative rounded-xl overflow-hidden border border-slate-800 shadow-2xl shadow-purple-500/10"
          style={{ minHeight: '800px' }}
        >
          <iframe
            src="/viz/prediction-flow.html"
            className="w-full border-0"
            title="BPR Prediction Flow"
            style={{ height: '900px' }}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-6 text-center"
        >
          <p className="text-slate-500 text-sm">
            Hover over any prediction to see its value, experimental comparison, and percent error.
            Colors: <span className="text-green-400">green</span> = matches experiment,{' '}
            <span className="text-yellow-400">yellow</span> = within 10%,{' '}
            <span className="text-red-400">red</span> = needs work.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
