#!/usr/bin/env python3
"""Assemble bpr.thestardrive.com from src/pages/*.html fragments.

Each fragment starts with a small header block:
    <!-- title: ... -->
    <!-- description: ... -->
    <!-- nav: key -->        (which nav item is active)
    <!-- math: yes -->       (optional: load KaTeX)
Everything after the header is the page body. Output goes to public/.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
OUT = ROOT / "public"

NAV = [
    ("index", "index.html", "Introduction"),
    ("how", "how-it-works.html", "How it works"),
    ("math", "mathematics.html", "Mathematics"),
    ("results", "results.html", "Results"),
    ("experiments", "experiments.html", "Experiments"),
    ("status", "status.html", "Status"),
    ("calc", "calculator.html", "Calculator"),
    ("run", "run-it.html", "Run it"),
]

GITHUB = "https://github.com/jackalkahwati/BPR-Math-Spine"

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | Boundary Phase Resonance</title>
<meta name="description" content="{description}">
<meta property="og:title" content="{title} | Boundary Phase Resonance">
<meta property="og:description" content="{description}">
<meta property="og:type" content="website">
<meta property="og:url" content="https://bpr.thestardrive.com/{file}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/style.css">
{math}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <div class="wrap bar">
    <a class="brand" href="/">Boundary Phase Resonance</a>
    <button class="menu-btn" aria-label="Menu" aria-expanded="false" onclick="var n=document.getElementById('nav');var o=n.classList.toggle('open');this.setAttribute('aria-expanded',o)">Menu</button>
    <nav id="nav" class="nav">
{navitems}
      <a href="{github}" rel="noopener">GitHub</a>
    </nav>
  </div>
</header>
<main id="main">
"""

FOOT = """
</main>
<footer class="site-footer">
  <div class="wrap">
    <div class="foot-grid">
      <div>
        <div class="brand">Boundary Phase Resonance</div>
        <p class="dim">An open research framework in mathematical physics. Pre-publication work by Jack Al-Kahwati, StarDrive Inc. Not peer reviewed. Source code MIT licensed.</p>
      </div>
      <div>
        <div class="foot-h">Read</div>
        <a href="/how-it-works.html">How it works</a>
        <a href="/mathematics.html">Mathematics</a>
        <a href="/results.html">Results</a>
        <a href="/status.html">Status and negative findings</a>
        <a href="/bpr-paper.pdf">Paper (PDF, BPR 1.0 with 2.0 status note)</a>
      </div>
      <div>
        <div class="foot-h">Do</div>
        <a href="/calculator.html">Calculator</a>
        <a href="/experiments.html">Experiments that could rule it out</a>
        <a href="/run-it.html">Run the code</a>
        <a href="{github}" rel="noopener">Source on GitHub</a>
        <a href="mailto:jack@thestardrive.com">Contact</a>
      </div>
    </div>
    <p class="fine">&copy; 2026 StarDrive Inc. Results here have not been peer reviewed and should not be used as the basis for engineering, financial, or medical decisions. <a href="/privacy.html">Privacy</a> &middot; <a href="/terms.html">Terms</a></p>
  </div>
</footer>
</body>
</html>
"""

KATEX = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\\\(',right:'\\\\)',display:false}],throwOnError:false});"></script>"""


def meta(src: str, key: str, default: str = "") -> str:
    m = re.search(rf"<!--\s*{key}:\s*(.*?)\s*-->", src)
    return m.group(1) if m else default


def build_one(path: Path) -> None:
    src = path.read_text()
    title = meta(src, "title", path.stem)
    desc = meta(src, "description")
    active = meta(src, "nav")
    use_math = meta(src, "math") == "yes"
    body = re.sub(r"^(\s*<!--.*?-->\s*)+", "", src, count=1, flags=re.S)
    items = []
    for key, href, label in NAV:
        cls = ' class="active"' if key == active else ""
        items.append(f'      <a href="/{href}"{cls}>{label}</a>')
    html = HEAD.format(
        title=title, description=desc.replace('"', "&quot;"), file=path.name,
        math=KATEX if use_math else "", navitems="\n".join(items), github=GITHUB,
    ) + body.strip() + FOOT.format(github=GITHUB)
    (OUT / path.name).write_text(html)
    print("built", path.name)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "style.css").write_text((SRC / "style.css").read_text())
    for p in sorted((SRC / "pages").glob("*.html")):
        build_one(p)


if __name__ == "__main__":
    main()
