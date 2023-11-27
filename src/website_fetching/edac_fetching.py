import asyncio
from typing import Coroutine

import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
from urllib.parse import urljoin, urlparse


def load_data(file_path: str):
    df = pd.read_csv(file_path, sep=',', header=None)
    df.columns = ['date', 'EDAC']
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.set_index('date')
    return df


# %%
async def load_from_website(url: str) -> pd.DataFrame:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file. Status code: {response.status_code}")
    content = response.content.decode("utf-8")
    return load_data(StringIO(content))


# %%
async def load_edac_from_website_dir(directory_url: str, measures: list[str]) -> tuple[dict, dict]:
    response = requests.get(directory_url)
    datas_file_names = {m: "" for m in measures}
    datas = {m: pd.DataFrame() for m in measures}
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    hrefs = map(lambda link: link.get("href"), links)
    hrefs = filter(lambda href: href.endswith(".TAB"), hrefs)
    for href in hrefs:
        file_url = urljoin(directory_url, href)
        data = await load_from_website(file_url)
        for k in datas.keys():
            if k in href:
                datas[k] = data
                datas_file_names[k] = href
    return datas, datas_file_names


def find_measures(directory_url: str):
    response = requests.get(directory_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    measures = set()
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    for link in links:
        href = link.get("href")
        if href.endswith(".TAB"):
            measure = href.split("_")[2]
            measures.add(measure)
    return list(measures)

# %%
async def load_all_parent_directory(parent_url: str) -> tuple[list[str], list[tuple]]:
    response = requests.get(parent_url)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")
    hrefs = list(map(lambda link: link.get('href'), links))
    print("The links in the parents directory are:")
    print(hrefs)
    hrefs = filter(lambda href: href[:4].isdigit() and len(href) > 4, hrefs)
    directory_urls = list(map(lambda href: urljoin(parent_url, href), hrefs))
    measures = find_measures(directory_urls[0])
    print("The measures found for this directory are:")
    print(measures)
    return measures, await load_all_files(directory_urls, measures)


async def load_all_files(directory_urls: list[str], measures: list[str]) -> list[tuple]:
    # Get all
    tasks: list[Coroutine[tuple]] = []
    res: list[tuple] = []
    async for directory_url in tqdm(directory_urls, total=len(directory_urls)):
        tasks.append(load_edac_from_website_dir(directory_url, measures))
        if len(tasks) > 3:
            res.extend(await asyncio.gather(*tasks))
            tasks = []
    if len(tasks) > 0:
        res.extend(await asyncio.gather(*tasks))
    return res


# %%
def post_process_data(request_res: list, measures: list[str]):
    res_data = [r[0] for r in request_res]
    res_file_names = [r[1] for r in request_res]

    data = {name: [r[name].resample('D').max() for r in res_data if isinstance(r[name].index, pd.DatetimeIndex)] for name in measures}
    data_file_names = {name: [r[name] for r in res_file_names] for name in measures}
    data = {name: pd.concat(d) for name, d in data.items()}
    return data, data_file_names
