#!/bin/bash

# Set environment variables from GitHub Actions secrets
APP_KEY="${APP_KEY}"             # Dropbox App Key (Client ID)
APP_SECRET="${APP_SECRET}"       # Dropbox App Secret (Client Secret)
REFRESH_TOKEN="${REFRESH_TOKEN}" # Dropbox Refresh Token

# Location of archive output
BASE_OUTPUT_DIR="${GITHUB_WORKSPACE}/data/testing-input-output"
ZIP_FILE_NAME="navier_stokes_output.zip"
LOCAL_ZIP_FILE_PATH="${BASE_OUTPUT_DIR}/${ZIP_FILE_NAME}"

# Create zip archive from simulation output (contents only, no full path)
(
  cd "${GITHUB_WORKSPACE}/data/testing-input-output/navier_stokes_output" || exit 1
  echo "📦 Creating archive: ${ZIP_FILE_NAME}"
  zip -r "${LOCAL_ZIP_FILE_PATH}" ./*
)

# Confirm zip file exists
if [ ! -f "${LOCAL_ZIP_FILE_PATH}" ]; then
    echo "❌ ERROR: Archive not found at ${LOCAL_ZIP_FILE_PATH}"
    exit 1
fi

# Define Dropbox destination folder path
DROPBOX_UPLOAD_FOLDER="/engineering_simulations_pipeline"

echo "🔄 Uploading ${ZIP_FILE_NAME} to Dropbox folder: ${DROPBOX_UPLOAD_FOLDER}"

# Use absolute path for the Python uploader
python3 "${GITHUB_WORKSPACE}/src/upload_to_dropbox.py" \
    "${LOCAL_ZIP_FILE_PATH}" \
    "${DROPBOX_UPLOAD_FOLDER}" \
    "${REFRESH_TOKEN}" \
    "${APP_KEY}" \
    "${APP_SECRET}"

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Upload completed: ${ZIP_FILE_NAME} was successfully uploaded to Dropbox."
else
    echo "❌ ERROR: Dropbox upload failed."
    exit 1
fi



