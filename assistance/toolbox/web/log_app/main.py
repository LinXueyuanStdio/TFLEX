"""
@date: 2022/2/20
@description: 监听远程日志服务器，显示前端
```
python toolbox/web/log_app/main.py
python -m toolbox.web.log_app.main
```
"""
import os
import shutil

import click

from .app import start_app


def create_settings(preference_dir_of_log_app="output"):
    pj_path = os.path.realpath('.')  # user project path
    tools_path = os.path.realpath(__file__)[:-len("main.py")]  # installed pkg path
    if not os.path.isdir(os.path.join(pj_path, preference_dir_of_log_app)):
        shutil.copytree(os.path.join(tools_path, "output"), os.path.join(pj_path, preference_dir_of_log_app))
    elif not os.path.exists(os.path.join(pj_path, preference_dir_of_log_app, "default.cfg")):
        shutil.copy(os.path.join(tools_path, "output", "default.cfg"), os.path.join(pj_path, preference_dir_of_log_app))


@click.command()
@click.option("--log_dir", type=str, default="output", help="app 配置文件所在文件夹.")
@click.option("--log_config_name", type=str, default="default.cfg", help="启动 app 的配置文件。app 停机后会把运行期间被修改的配置保存到该文件，方便下次运行（这个设置在配置文件中关闭）。")
@click.option("--ip", type=str, default="0.0.0.0", help="IP 地址")
@click.option("--port", type=int, default=44666, help="端口。如果该端口不可用，会自动选一个可用的")
@click.option("--standby_hours", type=int, default=24, help="空转小时数。如果超过这个时间没有任何操作，会自动停止运行，防止资源浪费")
def main(log_dir, log_config_name, ip, port, standby_hours):
    log_dir = os.path.abspath(log_dir)
    create_settings(log_dir)
    start_app(log_dir, log_config_name, standby_hours, port, ip)


if __name__ == '__main__':
    main()
