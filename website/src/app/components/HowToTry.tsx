import { Terminal } from 'lucide-react';
import { motion } from 'motion/react';

export function HowToTry() {
  const steps = [
    "Clone the repository",
    "Create the conda environment (or use Docker)",
    "Run pytest to verify installation",
    "Run the Casimir demo script",
    "Open the boundary Laplacian notebook"
  ];

  return (
    <section className="w-full py-16 px-6 bg-background">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="flex items-center gap-3 mb-6"
        >
          <Terminal className="w-8 h-8 text-primary" />
          <h2>How to try it</h2>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          viewport={{ once: true }}
          className="bg-secondary/50 border border-border rounded-lg p-8"
        >
          <ol className="space-y-3">
            {steps.map((step, index) => (
              <motion.li
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.2 + index * 0.1 }}
                viewport={{ once: true }}
                className="flex items-start gap-3"
              >
                <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0">
                  {index + 1}
                </span>
                <span className="text-foreground/90 pt-0.5">{step}</span>
              </motion.li>
            ))}
          </ol>
          <p className="text-muted-foreground mt-6 italic">
            See README sections for FEniCS vs minimal install options.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
