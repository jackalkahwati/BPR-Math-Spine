import { Github, Mail, BookOpen } from 'lucide-react';
import { motion } from 'motion/react';
import { ImageWithFallback } from './figma/ImageWithFallback';

export function Hero() {
  return (
    <section className="relative bg-gradient-to-b from-slate-950 to-slate-900 text-white py-24 px-6 overflow-hidden">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 opacity-20">
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1771013303894-b19453bc4ac5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwzfHxhYnN0cmFjdCUyMHdhdmVzJTIwcGF0dGVybnxlbnwxfHx8fDE3NzU0MzA3NTV8MA&ixlib=rb-4.1.0&q=80&w=1080"
          alt="Abstract wave pattern"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950/50 to-slate-900/90"></div>
      </div>

      <div className="max-w-5xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8 inline-block px-4 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-300 text-sm"
        >
          Open Research Framework
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-5xl md:text-6xl mb-6 tracking-tight"
        >
          Boundary Phase Resonance
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-xl md:text-2xl text-slate-300 mb-8 max-w-3xl leading-relaxed"
        >
          Reproducible mathematics for a testable substrate framework
        </motion.p>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-lg text-slate-400 mb-12 max-w-3xl leading-relaxed"
        >
          Open-source implementation of the BPR mathematical spine: boundary fields, metric coupling,
          Casimir-scale signatures, and a broad modular library linking substrate parameters to
          predictions—documented for peer audit, with tests, benchmarks, and an experimental roadmap.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="flex flex-wrap gap-4"
        >
          <a
            href="https://github.com/jackalkahwati/BPR-Math-Spine"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors"
          >
            <Github className="w-5 h-5" />
            Explore the Framework
          </a>

          <a
            href="https://github.com/jackalkahwati/BPR-Math-Spine"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-6 py-3 rounded-lg border border-slate-700 transition-colors"
          >
            <BookOpen className="w-5 h-5" />
            Run the Code
          </a>

          <a
            href="mailto:jack@thestardrive.com"
            className="inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-6 py-3 rounded-lg border border-slate-700 transition-colors"
          >
            <Mail className="w-5 h-5" />
            Contact
          </a>
        </motion.div>
      </div>
    </section>
  );
}
