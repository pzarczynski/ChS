import asyncio
import os
import re

import aiohttp
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fire import Fire
from natsort import natsorted
from tqdm.asyncio import tqdm

load_dotenv()


def get_urls(url, pattern):
    """get all urls from a given page"""

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup.find_all("a"):
        address = tag.get("href")
        if address and re.match(pattern, address):
            yield address


async def download(url, session, path):
    """asynchronously download the file to the specified path"""

    r = await session.get(url)

    filename = url.split("/")[-1]
    filepath = os.path.join(path, filename)

    # retrieve total size of the file
    total_size = int(r.headers.get("content-length", 0))

    # iterate over chunks to display progress
    with tqdm(desc=filename, total=total_size, unit="B", unit_scale=True) as bar:
        with open(filepath, "wb") as f:
            async for data in r.content.iter_chunked(1024):
                bar.update(len(data))
                f.write(data)


def main(url=None, regex=None, path=".") -> None:
    if not url:
        url = os.environ.get("CHS_DB_URL", "")

    if not regex:
        regex = os.environ.get("CHS_DB_REGEX", "")

    urls = get_urls(url, regex)
    urls = natsorted(urls, reverse=True)

    # display sorted list of retrievable files
    for i, url in list(enumerate(urls))[::-1]:
        print(f"[{i}]\t", url, end="\n\n")

    indices = map(int, input("> ").split())

    # TODO move this out??
    async def download_all():
        session = aiohttp.ClientSession()
        tasks = []

        for idx in indices:
            url = urls[idx]
            task = asyncio.create_task(download(url, session, path))
            tasks.append(task)

        for task in tasks:
            await task

        await session.close()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_all())


if __name__ == "__main__":
    Fire(main)
