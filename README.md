# Temporal Hand Labanotation

This repository contains the code, experiment notes, and paper materials for
the Temporal Hand Labanotation project.

## Repository Layout

- `tools/`: training, evaluation, auditing, and report-generation scripts
- `experiments/`: research notes and generated experiment artifacts
- `paper/submission_ready/`: cleaned AAAI submission package
- `AGENTS.md`: local project instructions

## What Is Included

- source code for data processing, training, evaluation, and analysis
- experiment design notes and paper-writing notes
- the final anonymous submission package for the paper

## What Is Not Tracked

Large generated artifacts and local packaging outputs are excluded from git,
including:

- `experiments/generated/`
- zip archives used for local packaging
- LaTeX temporary build files

The external dataset is not stored in this repository. In the local workspace,
the dataset root is `/opt/tiger/InterHand`.

## GitHub Push

After creating a GitHub repository, push with:

```bash
git remote add origin <your-github-repo-url>
git push -u origin main
```
