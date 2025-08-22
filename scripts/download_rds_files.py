import requests
import os

def download_file(url, filename):
    """
    Downloads a file from a given URL and saves it to a local file.
    
    Args:
        url (str): The URL of the file to download.
        filename (str): The path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    # Directory to store the downloaded data
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)

    # List of files to download: (URL, local_filename)
    files = [
        ("https://figshare.com/ndownloader/files/44388752", "PTC.rds"),
        ("https://figshare.com/ndownloader/files/44388245", "list2_dtbl.rds"),
    ]

    for url, filename in files:
        out_path = os.path.join(data_dir, filename)
        print(f"Downloading {url} as {out_path}...")
        download_file(url, out_path)
    
    print("Download complete.")
