#!/bin/bash

# Set environment variables from GitHub Actions secrets
APP_KEY="${APP_KEY}"         # Dropbox App Key (Client ID)
APP_SECRET="${APP_SECRET}"   # Dropbox App Secret (Client Secret)
REFRESH_TOKEN="${REFRESH_TOKEN}" # Dropbox Refresh Token

# Define the base output directory (where navier_stokes_output is located)
# This should match the OUTPUT_RESULTS_BASE_DIR from the GitHub Actions YAML
BASE_OUTPUT_DIR="${GITHUB_WORKSPACE}/data/testing-output-run"

# Define the folder to be zipped
FOLDER_TO_ZIP="${BASE_OUTPUT_DIR}/navier_stokes_output"

# Define the name and path for the output zip file
ZIP_FILE_NAME="navier_stokes_output.zip"
LOCAL_ZIP_FILE_PATH="${BASE_OUTPUT_DIR}/${ZIP_FILE_NAME}"

echo "📦 Creating zip archive of ${FOLDER_TO_ZIP}..."

# Create the zip archive
# -r: recurse into directories
# -q: quiet mode (don't print names of files added)
zip -rq "${LOCAL_ZIP_FILE_PATH}" "${FOLDER_TO_ZIP}"

# Check if zipping was successful
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create zip archive of ${FOLDER_TO_ZIP}."
    exit 1
fi

echo "✅ Successfully created zip archive: ${LOCAL_ZIP_FILE_PATH}"

# Define the Dropbox upload folder (as per previous configuration)
DROPBOX_UPLOAD_FOLDER="/engineering_simulations_pipeline"

echo "🔄 Attempting to upload ${LOCAL_ZIP_FILE_PATH} to Dropbox folder ${DROPBOX_UPLOAD_FOLDER}..."

# Call the Python script to upload the zip file
python3 src/upload_to_dropbox.py \
    "${LOCAL_ZIP_FILE_PATH}" \
    "${DROPBOX_UPLOAD_FOLDER}" \
    "${REFRESH_TOKEN}" \
    "${APP_KEY}" \
    "${APP_SECRET}"

# Verify the upload (by checking the exit code of the Python script)
if [ $? -eq 0 ]; then
    echo "✅ Successfully uploaded ${ZIP_FILE_NAME} to Dropbox."
else
    echo "❌ ERROR: Failed to upload ${ZIP_FILE_NAME} to Dropbox."
    exit 1
fi



