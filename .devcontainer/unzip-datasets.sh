#!/bin/bash
# This script runs after the dev container is created to prepare datasets.

# Exit immediately if a command fails
set -e

echo "--- Preparing datasets (postCreateCommand) ---"

# 1. Ensure unzip tool is installed
if ! command -v unzip &> /dev/null; then
    echo "Installing unzip..."
    apt-get update && apt-get install -y unzip
fi

# 2. Ensure Git LFS files are downloaded
echo "Pulling Git LFS files..."
git lfs pull

# 3. Define source and destination base directories
# The script runs inside the container where your repo is at /workspace
DATA_DIR="/workspace/data"
DEST_BASE_DIR="/workspace/data/dataset"

# 4. Ensure the main destination directory exists
mkdir -p "$DEST_BASE_DIR"

echo "--- Searching for .zip files in $DATA_DIR ---"

# 5. Loop through each .zip file in the data directory
for ZIP_FILE in "$DATA_DIR"/*.zip; do
    # This check handles the case where no .zip files are found
    [ -f "$ZIP_FILE" ] || continue

    echo "Processing file: $(basename "$ZIP_FILE")"

    # Extract the base filename without the .zip extension (e.g., "dataset1.zip" -> "dataset1")
    BASENAME=$(basename "$ZIP_FILE" .zip)

    # Create the specific destination directory path (e.g., "/workspace/data/dataset/dataset1")
    DEST_DIR="$DEST_BASE_DIR/$BASENAME"

    # Unzip only if the destination directory does not already exist
    if [ ! -d "$DEST_DIR" ]; then
        echo "  -> Unzipping to ${DEST_DIR}..."
        mkdir -p "$DEST_DIR"
        unzip -q "$ZIP_FILE" -d "$DEST_DIR" # Using -q for quiet output
    else
        echo "  -> Destination ${DEST_DIR} already exists. Skipping."
    fi
done

echo "--- Dataset setup complete ---"