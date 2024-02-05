import os
import shutil
import zipfile
import tarfile
from typing import Dict, Any, Optional, Iterable

import requests
import pandas as pd


def _download_file(url: str, directory: str) -> str:
    """
    Download a file from a given URL into the specified directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    local_filename = os.path.join(directory, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def _unzip_file(file_path: str, extract_to: str, delete_after: bool = True) -> None:
    """
    Unzip a file to a specified directory.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    if delete_after:
        os.remove(file_path)  # Remove the zip file after extraction


def _extract_tgz(tar_gz_path: str, extract_to: str, delete_after: bool = True) -> None:
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    if delete_after:
        os.remove(tar_gz_path)  # Remove the tar.gz file after extraction


def _remove_directory(directory: str) -> None:
    if os.path.exists(directory):
        shutil.rmtree(directory)


def _move_files(src_directory: str, dst_directory: str) -> None:
    for filename in os.listdir(src_directory):
        src_path = os.path.join(src_directory, filename)
        dst_path = os.path.join(dst_directory, filename)
        shutil.move(src_path, dst_path)


def _write_csv(texts: Iterable, labels: Iterable, csv_file_path: str, label_col_name: str = 'label',
               trans_dict: Optional[Dict[Any, Any]] = None):
    df = pd.DataFrame({
        'text': texts,
        label_col_name: labels
    })
    if trans_dict:
        df['trans_label'] = df[label_col_name].map(trans_dict)

    df.to_csv(csv_file_path, encoding='utf-8', index=False)
