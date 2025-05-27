#!/bin/bash

# Set environment variables from GitHub Actions secrets
APP_KEY="${APP_KEY}"         # Dropbox App Key (Client ID)
APP_SECRET="${APP_SECRET}"   # Dropbox App Secret (Client Secret)
REFRESH_TOKEN="${REFRESH_TOKEN}" # Dropbox Refresh Token

# ✅ Updated: Corrected the local file path for fluid simulation input
LOCAL_OUTPUT_FILE_PATH="$GITHUB_WORKSPACE/data/testing-input-output/navier_stokes_results.json"

# ✅ Fixed: Changed Dropbox folder path
DROPBOX_UPLOAD_FOLDER="/engineering_simulations_pipeline"

echo "🔄 Attempting to upload ${LOCAL_OUTPUT_FILE_PATH} to Dropbox folder ${DROPBOX_UPLOAD_FOLDER}..."

# ✅ Fixed: Changed Python script name to `upload_to_dropbox.py`
python3 src/upload_to_dropbox.py \
    "$LOCAL_OUTPUT_FILE_PATH" \
    "$DROPBOX_UPLOAD_FOLDER" \
    "$REFRESH_TOKEN" \
    "$APP_KEY" \
    "$APP_SECRET"

# Verify the upload (by checking the exit code of the Python script)
if [ $? -eq 0 ]; then
    echo "✅ Successfully uploaded ${LOCAL_OUTPUT_FILE_PATH} to Dropbox."
else
    echo "❌ ERROR: Failed to upload ${LOCAL_OUTPUT_FILE_PATH} to Dropbox."
    exit 1
fi


