# bpr.thestardrive.com

Static site for the BPR framework. Source fragments live in `src/pages/`, the
shared stylesheet in `src/style.css`. `build.py` assembles them into `public/`,
which is what gets deployed to the Vercel project `bpr-research`.

```bash
python3 website/build.py
cd website/public && python3 -m http.server 9001   # preview
vercel deploy --prod --cwd website/public           # deploy (project linked via .vercel/)
```

Every number on the site must be traceable to a function in `bpr/` and a row in
`VALIDATION_STATUS.md` or `doc/CLOSED_AND_DEPRECATED.md`. Keep the status labels
honest: `derived`, `framework`, `consistent`, `conjectural`, `withdrawn`, `input`.
