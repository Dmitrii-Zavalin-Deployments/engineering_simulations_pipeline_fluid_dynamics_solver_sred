# src/upload_to_dropbox.py
# 📤 Dropbox Uploader — uploads local files to Dropbox with audit-grade logging and token refresh

import dropbox
import os
import requests
import sys

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def refresh_access_token(refresh_token, client_id, client_secret):
    """Refreshes the Dropbox access token using the refresh token."""
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to refresh access token: Status Code {response.status_code}, Response: {response.text}")

def upload_file_to_dropbox(local_file_path, dropbox_file_path, refresh_token, client_id, client_secret):
    """Uploads a local file to a specified path on Dropbox."""
    try:
        access_token = refresh_access_token(refresh_token, client_id, client_secret)
        dbx = dropbox.Dropbox(access_token)

        with open(local_file_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)

        if debug:
            print(f"✅ Successfully uploaded file to Dropbox: {dropbox_file_path}")
        return True
    except Exception as e:
        if debug:
            print(f"❌ Failed to upload file '{local_file_path}' to Dropbox: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python src/upload_to_dropbox.py <local_file_path> <dropbox_folder> <refresh_token> <client_id> <client_secret>")
        sys.exit(1)

    local_file_to_upload = sys.argv[1]
    dropbox_folder = sys.argv[2]
    refresh_token = sys.argv[3]
    client_id = sys.argv[4]
    client_secret = sys.argv[5]

    output_file_name = os.path.basename(local_file_to_upload)
    dropbox_file_path = f"{dropbox_folder}/{output_file_name}"

    if not os.path.exists(local_file_to_upload):
        print(f"❌ Error: The output file '{local_file_to_upload}' was not found. Please ensure the preceding steps successfully generated this file.")
        sys.exit(1)

    if not upload_file_to_dropbox(local_file_to_upload, dropbox_file_path, refresh_token, client_id, client_secret):
        sys.exit(1)



