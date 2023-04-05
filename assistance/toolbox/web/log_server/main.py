"""
@date: 2022/2/19
@description: 监听指定文件夹下的日志，以 web api 的形式提供服务给 log_app
```
python toolbox/web/log_server/main.py
python -m toolbox.web.log_server.main
```
"""
from typing import Optional, List

import click
import uvicorn
from fastapi import FastAPI

from .log_server import LogServer

app = FastAPI()

log_server = LogServer()


@app.get("/")
def read_root():
    return {"你好": "请到该链接下查看 api 文档 /docs"}


@app.get("/logs")
def read_logs(ignore_log_names: Optional[dict] = None) -> List[dict]:
    return log_server.read_logs(ignore_log_names)


@app.get("/certain_logs")
def read_certain_logs(log_dir_names: List[str]) -> List[dict]:
    return log_server.read_certain_logs(log_dir_names)


@click.command()
@click.option("--log_dir", type=str, default="./output", help="日志所在文件夹")
@click.option("--ip", type=str, default="0.0.0.0", help="IP 地址")
@click.option("--port", type=int, default=45666, help="端口。如果该端口不可用，会自动选一个可用的")
def main(log_dir: str, ip, port):
    app.root_path = log_dir
    log_server.set_log_dir(log_dir)
    uvicorn.run(app, host=ip, port=port)


if __name__ == '__main__':
    main()
