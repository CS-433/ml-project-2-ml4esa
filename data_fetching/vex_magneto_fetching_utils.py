import asyncio
import os
from urllib.parse import urljoin
import json
import numpy as np
import pandas as pd
import re
import requests
from io import StringIO

from bs4 import BeautifulSoup
from tqdm import tqdm

datetime_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}  '

data_dir_name = '../DATA/VEX_MAGNETO/fetched_data'


def load_data(content: StringIO) -> pd.DataFrame:
    # Find first line with datetime
    first_line = ""
    for i, line in enumerate(content, start=0):
        if re.search(datetime_pattern, line):
            first_line = line
            break

    names = ['date', 'BX', 'BY', 'BZ', 'BT', 'XSC', 'YSC', 'ZSC', 'RSC']
    df1 = pd.read_csv(StringIO(first_line), header=None, delimiter='\s+', names=names)
    df = pd.read_csv(content, header=None, delimiter='\s+', names=names)
    df = pd.concat([df1, df])  # add first line to rest of the data

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%f')
    df.set_index('date', inplace=True)
    df = df[~df.isin([99999.999]).any(axis=1)]
    return df.resample('5min').mean()


def fetch_document(url: str) -> pd.DataFrame:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")
    content = response.text
    document = load_data(StringIO(content))
    return document


def fetch_documents(directory: str) -> pd.DataFrame:
    response = requests.get(directory)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    hrefs = map(lambda link: link.get("href"), links)
    hrefs = list(filter(lambda href: href.endswith(".TAB"), hrefs))
    documents = []
    for href in tqdm(hrefs, desc=directory.split('/')[-2]):
        file_url = urljoin(directory, href)
        documents.append(fetch_document(file_url))
    return pd.concat(documents)


def fetch_directories(parent_directory: str):
    response = requests.get(parent_directory)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    hrefs = map(lambda link: link.get("href"), links)
    hrefs = filter(lambda href: "ORB" in href, hrefs)
    flinks = list(map(lambda href: urljoin(parent_directory, href), hrefs))

    pdir_name = parent_directory.split('/')[-3]
    cache_name = f'{data_dir_name}/cache.json'
    for flink in reversed(flinks):
        link_name = flink.split('/')[-2]
        if not is_cached(pdir_name, cache_name, link_name):
            df = fetch_documents(flink)
            save_and_cache(df, pdir_name, cache_name, link_name)


def is_cached(pdir_name: str, cache_name: str, link_name: str):
    cache = json.load(open(cache_name, 'r'))
    if pdir_name in cache.keys():
        if link_name in cache[pdir_name]:
            return True
    return False


def save_and_cache(df: pd.DataFrame, pdir_name: str, cache_name: str, link_name: str):
    if os.path.exists(f"{data_dir_name}/{pdir_name}.csv"):
        old_df = pd.read_csv(f"{data_dir_name}/{pdir_name}.csv", index_col='date')
        old_df.index = pd.to_datetime(old_df.index)
        new_df = pd.concat([old_df, df])
    else:
        new_df = df
    new_df.to_csv(f"{data_dir_name}/{pdir_name}.csv")
    cache = json.load(open(cache_name, 'r'))
    if pdir_name not in cache.keys():
        cache[pdir_name] = []
    cache[pdir_name].append(link_name)
    json.dump(cache, open(cache_name, 'w'), indent=4)


if __name__ == '__main__':
    pdir_names = [f"VEX-V-Y-MAG-4-EXT{i}-V1.0/" for i in range(1, 5)] + [f"VEX-V-Y-MAG-4-V1.0/"]
    pdirs = [f"https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/MAG/{pdir_name}DATA/" for pdir_name in pdir_names]
    for pdir in pdirs:
        print(f"Fetching {pdir}")
        res = fetch_directories(pdir)
