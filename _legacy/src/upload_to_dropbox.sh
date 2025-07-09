#!/bin/bash

# Set environment variables from GitHub Actions secrets
APP_KEY="${APP_KEY}"             # Dropbox App Key (Client ID)
APP_SECRET="${APP_SECRET}"       # Dropbox App Secret (Client Secret)
REFRESH_TOKEN="${REFRESH_TOKEN}" # Dropbox Refresh Token

# Define the base output directory (must match OUTPUT_RESULTS_BASE_DIR in your workflow)
BASE_OUTPUT_DIR="${GITHUB_WORKSPACE}/data/testing-output-run"

# Folder inside BASE_OUTPUT_DIR to be zipped
FOLDER_TO_ZIP="navier_stokes_output"

# Archive name and full path
ZIP_FILE_NAME="navier_stokes_output.zip"
LOCAL_ZIP_FILE_PATH="${BASE_OUTPUT_DIR}/${ZIP_FILE_NAME}"

echo "📦 Creating zip archive of ${FOLDER_TO_ZIP} inside ${BASE_OUTPUT_DIR}..."

# Navigate to base directory
cd "${BASE_OUTPUT_DIR}" || { echo "❌ ERROR: Failed to enter ${BASE_OUTPUT_DIR}"; exit 1; }

# Create the zip archive relative to this directory
zip -rq "${ZIP_FILE_NAME}" "${FOLDER_TO_ZIP}"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create zip archive."
    exit 1
fi

echo "✅ Successfully created zip archive: ${LOCAL_ZIP_FILE_PATH}"

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



