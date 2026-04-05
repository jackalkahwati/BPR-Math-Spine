import { Check } from 'lucide-react';
import { motion } from 'motion/react';

export function WhySection() {
  const reasons = [
    {
      title: "Reproduce the math spine",
      description: "Numbered equations tied to functions and tests so others can verify the implementation, not just read PDFs."
    },
    {
      title: "Separate prediction from poetry",
      description: "Hundreds of automated checks, scorecards against standard data, and an explicit validation/limitations story."
    },
    {
      title: "Invite experiment",
      description: "Roadmap-style tests, phonon/MEMS-relevant coupling scales discussed in README, and clear \"what would falsify this\" language."
    }
  ];

  return (
    <section className="w-full py-16 px-6 bg-secondary/30">
      <div className="max-w-4xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="mb-10"
        >
          Why this repository exists
        </motion.h2>
        <div className="grid gap-8 md:grid-cols-3">
          {reasons.map((reason, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
              viewport={{ once: true }}
              className="flex flex-col gap-3"
            >
              <div className="flex items-start gap-3">
                <div className="bg-primary text-primary-foreground rounded-full p-1 mt-1">
                  <Check className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="mb-2">{reason.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {reason.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
