#!/bin/bash
# src/download_from_dropbox.sh
# 📦 Dropbox Sync Script — downloads all files from a folder and verifies audit trace

# Set environment variables from GitHub Actions secrets
APP_KEY="${APP_KEY}"
APP_SECRET="${APP_SECRET}"
REFRESH_TOKEN="${REFRESH_TOKEN}"

# Define Dropbox folder and local destination
DROPBOX_FOLDER="/engineering_simulations_pipeline"
LOCAL_FOLDER="./data/testing-input-output"
LOG_FILE="./dropbox_download_log.txt"

# Create the local folder if it doesn't exist
mkdir -p "$LOCAL_FOLDER"

# Run the Python script to call the Dropbox download function
python3 src/download_dropbox_files.py "$DROPBOX_FOLDER" "$LOCAL_FOLDER" "$REFRESH_TOKEN" "$APP_KEY" "$APP_SECRET" "$LOG_FILE"

# Verify downloaded files
if [ "$(ls -A "$LOCAL_FOLDER")" ]; then
    echo "✅ Files successfully downloaded to $LOCAL_FOLDER"
else
    echo "❌ ERROR: No files were downloaded from Dropbox. Check your credentials and folder path."
    exit 1
fi



