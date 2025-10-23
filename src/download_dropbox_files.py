# src/download_dropbox_files.py
# 📦 Dropbox File Downloader — downloads all files from a folder and logs audit trace

import dropbox
import os
import requests
import sys

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def refresh_access_token(refresh_token, client_id, client_secret):
    """
    Refreshes Dropbox access token using OAuth2 refresh flow.
    """
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
        raise Exception("Failed to refresh access token")

def download_files_from_dropbox(dropbox_folder, local_folder, refresh_token, client_id, client_secret, log_file_path):
    """
    Downloads all files from a specified Dropbox folder and logs each step.
    """
    access_token = refresh_access_token(refresh_token, client_id, client_secret)
    dbx = dropbox.Dropbox(access_token)

    with open(log_file_path, "a") as log_file:
        log_file.write("Starting download process...\n")
        try:
            os.makedirs(local_folder, exist_ok=True)

            has_more = True
            cursor = None
            while has_more:
                if cursor:
                    result = dbx.files_list_folder_continue(cursor)
                else:
                    result = dbx.files_list_folder(dropbox_folder)
                log_file.write(f"Listing files in Dropbox folder: {dropbox_folder}\n")

                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        local_path = os.path.join(local_folder, entry.name)
                        with open(local_path, "wb") as f:
                            _, res = dbx.files_download(path=entry.path_lower)
                            f.write(res.content)
                        log_file.write(f"Downloaded {entry.name} to {local_path}\n")
                        if debug:
                            print(entry.name)

                has_more = result.has_more
                cursor = result.cursor

            log_file.write("Download and delete process completed successfully.\n")
        except dropbox.exceptions.ApiError as err:
            log_file.write(f"Error downloading files: {err}\n")
            if debug:
                print(f"Error downloading files: {err}")
        except Exception as e:
            log_file.write(f"Unexpected error: {e}\n")
            if debug:
                print(f"Unexpected error: {e}")

if __name__ == "__main__":
    dropbox_folder = sys.argv[1]
    local_folder = sys.argv[2]
    refresh_token = sys.argv[3]
    client_id = sys.argv[4]
    client_secret = sys.argv[5]
    log_file_path = sys.argv[6]

    download_files_from_dropbox(dropbox_folder, local_folder, refresh_token, client_id, client_secret, log_file_path)



