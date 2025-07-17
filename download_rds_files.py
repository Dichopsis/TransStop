import requests
import os

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    files = [
        ("https://figshare.com/ndownloader/files/44408420", "NTC.rds"),
        ("https://figshare.com/ndownloader/files/44388752", "PTC.rds"),
        ("https://figshare.com/ndownloader/files/44388245", "list2_dtbl.rds"),
        ("https://figshare.com/ndownloader/files/44388308", "list2.rds"),
    ]
    for url, filename in files:
        out_path = os.path.join(data_dir, filename)
        print(f"Downloading {url} as {out_path}...")
        download_file(url, out_path)
    print("Download complete.")
