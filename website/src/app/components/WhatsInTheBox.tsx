import { Code, Layers, CheckCircle, Wrench, FileText } from 'lucide-react';
import { motion } from 'motion/react';

export function WhatsInTheBox() {
  const features = [
    {
      icon: Code,
      title: "Core equations",
      description: "Boundary Laplacian / phase field, metric perturbation, stress integration and Casimir-style sweeps—implemented with FEniCS where applicable, with fallbacks when solvers aren't installed."
    },
    {
      icon: Layers,
      title: "Theory modules",
      description: "From boundary memory and impedance to cosmology, QCD/flavor, emergent spacetime, quantum foundations, and beyond—wired through a first-principles style pipeline from substrate-style inputs."
    },
    {
      icon: CheckCircle,
      title: "Verification",
      description: "Unit tests, consistency audits, benchmark regressions, Wolfram-language smoke checks—aimed at mechanical trust in the mathematics."
    },
    {
      icon: Wrench,
      title: "Tools",
      description: "Jupyter notebooks, CLI demos, optional REST API for predictions and pipelines, Docker images for consistent environments."
    },
    {
      icon: FileText,
      title: "Evidence & papers",
      description: "Curated experiment/evidence docs and a pipeline for ingesting and auditing literature against the framework."
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
          className="mb-10"
        >
          What's in the box
        </motion.h2>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05 }}
                className="border border-border rounded-lg p-6 hover:border-primary/50 transition-colors cursor-pointer"
              >
                <div className="bg-primary/10 text-primary rounded-lg w-12 h-12 flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
