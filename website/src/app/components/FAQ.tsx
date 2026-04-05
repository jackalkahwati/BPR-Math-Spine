import { motion } from 'motion/react';

export function FAQ() {
  const faqs = [
    {
      question: "Is this \"standard physics\"?",
      answer: "No—it's a proposed framework. The repository is built to show what is derived, what is tested in code, and what would refute key predictions."
    },
    {
      question: "Why so many modules?",
      answer: "The codebase explores cross-domain consequences of the same substrate/boundary formalism; not every module carries the same epistemic status—see validation and derivation roadmaps."
    },
    {
      question: "Can I use the API in my own app?",
      answer: "Yes, subject to MIT license and understanding that outputs are research-grade, not a consumer product."
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
          Frequently Asked Questions
        </motion.h2>
        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-background border border-border rounded-lg p-6"
            >
              <h3 className="mb-3">{faq.question}</h3>
              <p className="text-muted-foreground leading-relaxed">{faq.answer}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
