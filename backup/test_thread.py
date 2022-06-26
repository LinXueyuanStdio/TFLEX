"""
@date: 2022/3/14
@description: null
"""
import datetime
import threading
import time


def thread_func(start_time):  # 线程函数
    time.sleep(1)
    i = 0
    while (i < 11):
        print(datetime.timedelta(seconds=time.time() - start_time))
        i += 1


def many_thread():
    threads = []
    for _ in range(10):  # 循环创建500个线程
        t = threading.Thread(target=thread_func, args=[time.time()])
        threads.append(t)
        t.setDaemon(True)  # 给每个子线程添加守护线程
    for t in threads:  # 循环启动500个线程
        t.start()
    for t in threads:
        t.join()  # 阻塞线程


if __name__ == '__main__':
    many_thread()
    print("thread end")
