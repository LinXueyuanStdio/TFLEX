"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/17
@description: null
A rudimentary URL downloader (like wget or curl) to demonstrate Rich progress bars.
"""
import os.path
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event
from typing import Iterable
from urllib.request import urlopen

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class DownloadManager:
    def __init__(self):
        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )

        self.done_event = Event()

        def handle_sigint(signum, frame):
            self.done_event.set()

        signal.signal(signal.SIGINT, handle_sigint)

    def copy_url(self, task_id: TaskID, url: str, path: str) -> None:
        """Copy data from a url to a local file."""
        self.progress.console.log(f"Requesting {url}")
        response = urlopen(url)
        # This will break if the response doesn't contain content length
        self.progress.update(task_id, total=int(response.info()["Content-length"]))
        with open(path, "wb") as dest_file:
            self.progress.start_task(task_id)
            for data in iter(partial(response.read, 32768), b""):
                dest_file.write(data)
                self.progress.update(task_id, advance=len(data))
                if self.done_event.is_set():
                    return
        self.progress.console.log(f"Downloaded {path}")

    def download(self, urls: Iterable[str], dest_dir: str):
        """Download multuple files to the given directory."""

        with self.progress:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for url in urls:
                    filename = url.split("/")[-1]
                    dest_path = os.path.join(dest_dir, filename)
                    task_id = self.progress.add_task("download", filename=filename, start=False)
                    pool.submit(self.copy_url, task_id, url, dest_path)

    def download_to_path(self, urls: Iterable[str], dest_path: str):
        """Download multuple files to the given directory."""

        with self.progress:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for url in urls:
                    filename = url.split("/")[-1]
                    task_id = self.progress.add_task("download", filename=filename, start=False)
                    pool.submit(self.copy_url, task_id, url, dest_path)


if __name__ == "__main__":
    # Try with https://releases.ubuntu.com/20.04/ubuntu-20.04.3-desktop-amd64.iso
    if sys.argv[1:]:
        DownloadManager().download(sys.argv[1:], "./")
    else:
        print("Usage:\n\tpython downloader.py URL1 URL2 URL3 (etc)")
