#!/usr/bin/env bash
# =============================================================================
# download_data.sh — Download Jigsaw dataset from Kaggle
# =============================================================================
#
# Prerequisites:
#   1. Install Kaggle CLI:       pip install kaggle
#   2. Create API token at:      https://www.kaggle.com/account
#   3. Place kaggle.json at:     ~/.kaggle/kaggle.json  (chmod 600)
#
# Usage:
#   bash data/download_data.sh
# =============================================================================

set -e

DATA_DIR="$(dirname "$0")"
COMPETITION="jigsaw-unintended-bias-in-toxicity-classification"

echo "==> Downloading Jigsaw dataset..."
echo "    Competition: $COMPETITION"
echo "    Output dir:  $DATA_DIR"
echo ""

# Check kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Check credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Move the downloaded file: mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Download
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

# Unzip
echo "==> Unzipping..."
unzip -o "$DATA_DIR/${COMPETITION}.zip" -d "$DATA_DIR"
rm "$DATA_DIR/${COMPETITION}.zip"

echo ""
echo "==> Done! Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.csv


