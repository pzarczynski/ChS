import asyncio
import os

import aiohttp
import requests
from bs4 import BeautifulSoup
from fire import Fire
from natsort import natsorted
from tqdm import tqdm

URL = "https://database.nikonoel.fr/"


def geturls(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup.find_all("a"):
        yield tag.get("href")


async def download(url, session, path):
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


def main(path=os.getcwd()):
    urls = filter(lambda x: x.endswith(".zip"), geturls(URL))
    urls = natsorted(urls, reverse=True)

    # display sorted list of retrievable archives
    for i, url in list(enumerate(urls))[::-1]:
        print(f"[{i}]", url, end="\n\n")

    indices = map(int, input("> ").split())

    # TODO move this out??
    async def downloadall():
        s = aiohttp.ClientSession()
        tasks = []

        for idx in indices:
            url = urls[idx]
            task = asyncio.create_task(download(url, s, path))
            tasks.append(task)

        for task in tasks:
            await task

        await s.close()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(downloadall())


if __name__ == "__main__":
    Fire(main)
