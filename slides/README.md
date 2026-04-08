# Hackathon Pitch Deck

This directory contains a source-controlled PDF pitch deck for the Track 1 B300 hackathon story in this fork.

Generate the deck with:

```bash
python slides/hackathon_pitch.py
```

The script writes:

- `slides/hackathon_pitch.pdf`

The deck is intentionally product-facing. It pitches the actual repo deliverables that a judge can inspect quickly:

- B300 prestage, smoke, train, and eval entrypoints
- Controlled BF16 vs FP8 vs FP4 comparison flow
- Machine-readable artifacts such as `comparison.json` and `comparison.md`
