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


def load_data(content: StringIO) -> pd.DataFrame:
    # Find first line with datetime
    first_line = ""
    for i, line in enumerate(content, start=0):
        if re.search(datetime_pattern, line):
            first_line = line
            break

    names = ['date', 'BX', 'BY', 'BZ', 'BT', 'XSC', 'YSC', 'ZSC', 'RSC']
    df1 = pd.read_csv(StringIO(first_line), header=None, delimiter=r'\s+', names=names)
    df = pd.read_csv(content, header=None, delimiter=r'\s+', names=names)
    df = pd.concat([df1, df])  # add first line to rest of the data

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df.resample('h').mean()


async def fetch_document(url: str) -> pd.DataFrame:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")
    content = response.text
    document = load_data(StringIO(content))
    return document


async def fetch_documents(directory: str) -> pd.DataFrame:
    response = requests.get(directory)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    hrefs = map(lambda link: link.get("href"), links)
    hrefs = list(filter(lambda href: href.endswith(".TAB"), hrefs))
    documents = []
    tasks = []
    for href in tqdm(hrefs, desc=directory.split('/')[-2]):
        file_url = urljoin(directory, href)
        tasks.append(fetch_document(file_url))
        if len(tasks) > 0:
            documents.extend(await asyncio.gather(*tasks))
            tasks = []
    # documents.extend(await asyncio.gather(*tasks))
    return pd.concat(documents)


async def fetch_directories(parent_directory: str):
    response = requests.get(parent_directory)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    hrefs = map(lambda link: link.get("href"), links)
    hrefs = filter(lambda href: "ORB" in href, hrefs)
    flinks = list(map(lambda href: urljoin(parent_directory, href), hrefs))

    pdir_name = parent_directory.split('/')[-3]
    cache_name = '../../DATA/VEX_MAGNETO/fetched_data/cache.json'
    for flink in flinks:
        link_name = flink.split('/')[-2]
        if not is_cached(pdir_name, cache_name, link_name):
            df = await fetch_documents(flink)
            save_and_cache(df, pdir_name, cache_name, link_name)


def is_cached(pdir_name: str, cache_name: str, link_name: str):
    cache = json.load(open(cache_name, 'r'))
    if pdir_name in cache.keys():
        if link_name in cache[pdir_name]:
            return True
    return False


def save_and_cache(df: pd.DataFrame, pdir_name: str, cache_name: str, link_name: str):
    if os.path.exists(f"../../DATA/VEX_MAGNETO/{pdir_name}.csv"):
        old_df = pd.read_csv(f"../../DATA/VEX_MAGNETO/fetched_data/{pdir_name}.csv", sep='\t', index_col='date')
        old_df.index = pd.to_datetime(old_df.index)
        new_df = pd.concat([old_df, df])
    else:
        new_df = df
    new_df.to_csv(f"../../DATA/VEX_MAGNETO/fetched_data/{pdir_name}.csv", sep='\t')
    cache = json.load(open(cache_name, 'r'))
    if pdir_name not in cache.keys():
        cache[pdir_name] = []
    cache[pdir_name].append(link_name)
    json.dump(cache, open(cache_name, 'w'), indent=4)


if __name__ == '__main__':
    pdir = "https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/MAG/VEX-V-Y-MAG-4-EXT1-V1.0/DATA/"
    pdir2 = "https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/MAG/VEX-V-Y-MAG-4-V1.0/DATA/"
    name = pdir.split('/')[-3]
    res = asyncio.run(fetch_directories(pdir))
    # res.to_csv(f"../../DATA/VEX_MAGNETO/{pdir_name}.csv", sep='\t')
