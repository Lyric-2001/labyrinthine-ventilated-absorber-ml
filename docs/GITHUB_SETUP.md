# GitHub Repository Setup

## Recommended Metadata

| Field | Value |
|---|---|
| Repository name | `labyrinthine-ventilated-absorber-ml` |
| Description | `Research code and validation data for machine-learning-driven design of labyrinthine ventilated sound absorbers.` |
| Initial visibility | Private, until code ownership and the final license are confirmed |
| Default branch | `main` |
| First release tag | `v1.0.0`, after public-release approval |

Suggested GitHub topics:

```text
acoustic-metamaterials
sound-absorption
ventilated-absorber
machine-learning
inverse-design
convolutional-neural-network
particle-swarm-optimization
comsol-multiphysics
scientific-computing
```

Do not ask GitHub to add a README, `.gitignore`, or license during repository creation. Those files are already included here.

## Pre-upload Check

From this directory, run:

```bash
python tools/verify_repository.py --strict
```

The check must report zero errors. Warnings about historical absolute paths are expected and are documented in `docs/REPRODUCIBILITY.md`.

## Create the Empty GitHub Repository

On GitHub, choose **New repository**, enter the name and description above, and select **Private**. Do not initialize the repository with a README, `.gitignore`, or license because those files are already present in this folder.

## Command-line Upload

This curated folder is already initialized as a Git repository on branch `main`. First configure the commit identity if Git has not been configured on this computer. Use an email associated with the GitHub account or its GitHub-provided `noreply` address:

```bash
git config user.name "Longhui Yan"
git config user.email "YOUR-EMAIL"
```

Then run:

```bash
git add -A
git status --short
git commit -m "Release curated research code v1.0.0"
git remote add origin https://github.com/YOUR-ACCOUNT/labyrinthine-ventilated-absorber-ml.git
git push -u origin main
```

Replace `YOUR-ACCOUNT` with the GitHub account or organization that will own the repository.

## Recommended Settings After Upload

1. Keep the repository private until all authors or the institution approve public release and choose a license.
2. Enable Issues only if there is capacity to answer reproduction questions.
3. Protect `main` and require the `Repository quality checks` workflow for pull requests.
4. After public-release approval, create release `v1.0.0` from the first curated commit.
5. Publish the full data and COMSOL model on Zenodo or an institutional data repository, then add the DOI to `docs/DATASETS.md`.
6. Connect the GitHub repository to Zenodo before the first public release if a citable software DOI is wanted.

## Browser Upload Limitation

Do not upload this repository in one browser drag-and-drop operation. GitHub's web interface accepts at most 100 files per upload, while this repository contains more than 100 files. Although every included file is below both the browser's 25 MiB per-file limit and Git's 100 MiB limit, command-line Git or GitHub Desktop is the reliable way to preserve the complete directory tree. If the browser must be used, upload in batches of fewer than 100 files and verify every path afterward.

Do not drag the parent thesis folder into GitHub. It contains an 868 MiB COMSOL model, oversized training matrices, manuscripts, old Git history, IDE metadata, and third-party material that were intentionally excluded.
