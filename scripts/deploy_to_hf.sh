#!/usr/bin/env bash
#
# Deploy this repo to the Hugging Face Space at:
#   https://huggingface.co/spaces/charlesyapai/alphagenome_aging
#
# Why a separate script (not just `git push hf main`):
#   - HF Space README needs YAML frontmatter we don't want on the GitHub README
#   - HF Space repo size limits — we want to skip the same things gitignored
#   - HF auth uses a separate token (HF user/pat, not GitHub credentials)
#
# Prerequisites:
#   - You've already created the HF Space and set its SDK to "Docker"
#   - You've set ALPHAGENOME_API_KEY as a secret in the Space settings
#   - You have a Hugging Face write token (huggingface.co/settings/tokens)
#
# Usage:
#   bash scripts/deploy_to_hf.sh
#
# It will prompt for your HF token on first run. The token is sent via
# git's credential helper only; it is NOT saved into this repo.

set -euo pipefail
cd "$(dirname "$0")/.."

HF_USER="charlesyapai"
HF_SPACE="alphagenome_aging"
HF_REPO="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
SCRATCH="$(mktemp -d)"

echo "=== Cloning the HF Space repo to a scratch dir"
echo "    (auth: you'll be prompted for your HF write token if needed)"
git clone "$HF_REPO" "$SCRATCH/space"
cd "$SCRATCH/space"

echo "=== Removing all existing files except .git"
find . -mindepth 1 -maxdepth 1 ! -name ".git" -exec rm -rf {} +

echo "=== Copying repo contents (excluding gitignored / dockerignored items)"
SRC="/Users/charles/Desktop/Research Projects/AlphaGenome/Idea Testing 2"
rsync -a \
  --exclude='.git/' \
  --exclude='data/alphagenome_scores/' \
  --exclude='data/raw/' \
  --exclude='env/api_key.txt' \
  --exclude='1kGP_high_coverage_Illumina.*.tbi' \
  --exclude='*.vcf.gz' \
  --exclude='*.vcf.gz.tbi' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='.pytest_cache/' \
  --exclude='.DS_Store' \
  --exclude='README.md' \
  "$SRC/" .

echo "=== Installing HF-specific README (with required YAML frontmatter)"
cp "$SRC/scripts/hf_README.md" ./README.md

echo "=== Staging + committing"
git add -A
git -c "user.email=$(git -C "$SRC" config user.email 2>/dev/null || echo charles@charles.com)" \
    -c "user.name=$(git -C "$SRC" config user.name 2>/dev/null || echo Charles)" \
    commit -m "Deploy v2.1 inference demo" || {
      echo "  (nothing to commit — already up to date)"
    }

echo "=== Pushing to HF Space"
git push origin main

echo ""
echo "Deployed. Space will rebuild now — first build ~3-5 min."
echo "Watch the build at:"
echo "  https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}?logs=build"
echo ""
echo "Once green, the live demo is at:"
echo "  https://${HF_USER}-${HF_SPACE}.hf.space"
echo ""
echo "Cleanup:"
echo "  rm -rf $SCRATCH"
