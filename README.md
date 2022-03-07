# pl-learn
Learning PyTorch-Lightning

学习PyTorch-Lightning

## PL的优点
1. 多gpu并行(DDP)
2. 支持混合精度训练(AMP)
3. 性能瓶颈分析(Profiling)
4. 整合了tensorboard, wandb等各种logger，方便画图

## PL的使用逻辑

一个Trainer，两个Module (Dataset和model) \
详见代码`mnist_cls.py`

## TODO
1. 模型的保存，加载
2. Transfer Learning
