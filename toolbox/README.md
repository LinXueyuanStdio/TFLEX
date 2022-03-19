# KGE Toolbox

知识图谱嵌入工具箱，助力快速实验出成果

1. 通用
   1. 命令行参数解析
   2. 日志
   3. 进度条
   4. 随机种子
   5. TensorBoard监控
   6. 超参数自动搜索AutoML
   7. 梯度累加（应对小内存gpu对batch_size的限制）
   8. 中断训练、恢复训练
2. 知识图谱嵌入领域专用工具
   1. 嵌入降维可视化
   2. 数据集、常用数据预处理
   3. 链接预测任务、实体对齐任务（自动生成对应的数据集并训练）
   4. 测试指标（Hit@k、MR、MMR、AUC）
   5. 经典KGE模型的PyTorch版复现


## CheatSheet

CUDA_VISIBLE_DEVICES=3

内存分析，打印每一行代码执行前后的内存变化
```shell
pip install memory_profiler psutil
python -m memory_profiler main.py
```