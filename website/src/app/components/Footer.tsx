import { Github, Mail } from 'lucide-react';

export function Footer() {
  return (
    <footer className="w-full bg-primary text-primary-foreground py-12 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4 mb-8">
          <div>
            <h4 className="mb-3 text-primary-foreground">BPR-Math-Spine</h4>
            <p className="text-primary-foreground/80 leading-relaxed">
              Open-source research framework for Boundary Phase Resonance
            </p>
          </div>
          <div>
            <h4 className="mb-3 text-primary-foreground">Status</h4>
            <p className="text-primary-foreground/80">Public research draft</p>
          </div>
          <div>
            <h4 className="mb-3 text-primary-foreground">License</h4>
            <p className="text-primary-foreground/80">MIT License</p>
          </div>
          <div>
            <h4 className="mb-3 text-primary-foreground">Contact</h4>
            <div className="space-y-2">
              <a
                href="mailto:jack@thestardrive.com"
                className="flex items-center gap-2 text-primary-foreground/80 hover:text-primary-foreground transition-colors"
              >
                <Mail className="w-4 h-4" />
                jack@thestardrive.com
              </a>
              <a
                href="https://github.com/jackalkahwati/BPR-Math-Spine"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-primary-foreground/80 hover:text-primary-foreground transition-colors"
              >
                <Github className="w-4 h-4" />
                GitHub Repository
              </a>
            </div>
          </div>
        </div>
        <div className="border-t border-primary-foreground/20 pt-8 text-center text-primary-foreground/60">
          <p>© 2026 BPR-Math-Spine. Open research under MIT License.</p>
        </div>
      </div>
    </footer>
  );
}
