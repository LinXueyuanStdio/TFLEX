# 指定 gpu
```shell
CUDA_VISIBLE_DEVICES=3
```

# 内存分析，打印每一行代码执行前后的内存变化
```shell
pip install memory_profiler psutil
python -m memory_profiler main.py
```