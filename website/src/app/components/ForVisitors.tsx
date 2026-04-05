import { Users, FlaskConical, Terminal } from 'lucide-react';
import { motion } from 'motion/react';

export function ForVisitors() {
  const audiences = [
    {
      icon: Users,
      title: "Physicists / Reviewers",
      points: [
        "Complete framework documentation",
        "One-pager summaries",
        "Reviewer memos",
        "Experimental roadmap",
        "Limitations and falsification documentation",
        "Benchmark scorecard"
      ]
    },
    {
      icon: FlaskConical,
      title: "Experimentalists",
      points: [
        "Concrete falsification table",
        "Casimir-style signature details",
        "Neutrino and precision test timelines",
        "Primary sources in repository"
      ]
    },
    {
      icon: Terminal,
      title: "Developers / Reproducibility",
      points: [
        "MIT license",
        "pytest test suite",
        "conda/Docker environments",
        "REST API",
        "Wolfram scripts",
        "CSV prediction generation"
      ]
    }
  ];

  return (
    <section className="w-full py-16 px-6 bg-secondary/30">
      <div className="max-w-5xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="mb-10"
        >
          For different visitors
        </motion.h2>
        <div className="grid gap-8 md:grid-cols-3">
          {audiences.map((audience, index) => {
            const Icon = audience.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.15 }}
                viewport={{ once: true }}
                className="bg-background border border-border rounded-lg p-6"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-primary text-primary-foreground rounded-lg w-10 h-10 flex items-center justify-center">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3>{audience.title}</h3>
                </div>
                <ul className="space-y-2">
                  {audience.points.map((point, pointIndex) => (
                    <li key={pointIndex} className="text-muted-foreground flex items-start gap-2">
                      <span className="text-primary mt-1">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
