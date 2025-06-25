#!/bin/bash

# Set environment variables from GitHub Actions secrets
APP_KEY="${APP_KEY}"         # Dropbox App Key (Client ID)
APP_SECRET="${APP_SECRET}"   # Dropbox App Secret (Client Secret)
REFRESH_TOKEN="${REFRESH_TOKEN}" # Dropbox Refresh Token

# Define the base output directory
BASE_OUTPUT_DIR="${GITHUB_WORKSPACE}/data/testing-output-run"

# Define the folder to be zipped (target folder)
FOLDER_TO_ZIP="navier_stokes_output"

# Define the name and absolute path for the zip archive
ZIP_FILE_NAME="navier_stokes_output.zip"
LOCAL_ZIP_FILE_PATH="${BASE_OUTPUT_DIR}/${ZIP_FILE_NAME}"

echo "📦 Creating zip archive of ${FOLDER_TO_ZIP} inside ${BASE_OUTPUT_DIR}..."

# Move into the base directory to zip using relative path
cd "${BASE_OUTPUT_DIR}" || { echo "❌ ERROR: Failed to enter directory ${BASE_OUTPUT_DIR}."; exit 1; }

# Create the archive from inside the base directory
zip -rq "${ZIP_FILE_NAME}" "${FOLDER_TO_ZIP}"

# Check if zipping was successful
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create zip archive ${ZIP_FILE_NAME}."
    exit 1
fi

echo "✅ Successfully created zip archive: ${LOCAL_ZIP_FILE_PATH}"

# Define the Dropbox folder path
DROPBOX_UPLOAD_FOLDER="/engineering_simulations_pipeline"

echo "🔄 Uploading ${LOCAL_ZIP_FILE_PATH} to Dropbox at ${DROPBOX_UPLOAD_FOLDER}..."

# Call the Python Dropbox uploader script
python3 src/upload_to_dropbox.py \
    "${LOCAL_ZIP_FILE_PATH}" \
    "${DROPBOX_UPLOAD_FOLDER}" \
    "${REFRESH_TOKEN}" \
    "${APP_KEY}" \
    "${APP_SECRET}"

# Check upload success
if [ $? -eq 0 ]; then
    echo "✅ Upload completed: ${ZIP_FILE_NAME} was successfully uploaded to Dropbox."
else
    echo "❌ ERROR: Dropbox upload failed for ${ZIP_FILE_NAME}."
    exit 1
fi



