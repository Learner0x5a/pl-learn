import profile
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from pytorch_lightning import Trainer, LightningDataModule
from typing import Optional
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler, PyTorchProfiler


'''https://github.com/NVIDIA/nccl/issues/342
https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container
提示shm不足的时候，改用P2P; 
或者在docker run的时候扩大shm/指定--ipc=host选项
'''
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_LEVEL"] = "SYS" 

class myplmodule(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,10)

    def forward(self,x):
        # (N, 1, 28, 28)
        batch_size, channels, height, width = x.size()
        
        # (N, 1, 28, 28) -> (N, 28*28)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 在分类问题中，softmax + CrossEntropy等价于log_softmax + nll_loss
        # 但是直接计算softmax可能会有溢出的问题，因此取个对数
        x = F.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        # loss = F.cross_entropy(out, y)
        loss = F.nll_loss(out, y)
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # log机制
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(),lr=1e-3)

class MyDataModule(LightningDataModule):
    def __init__(self, data_dir: str = os.getcwd()):
        super().__init__()
        self.data_dir = data_dir

        # Transforms for mnist
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # # Setting default dims here because we know them.
        # # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self):
        # called only on 1 GPU
        # 在此函数内进行所有一次性的数据处理操作，例如下载、分词等
        '''
        download_dataset()
        tokenize()
        build_vocab()
        '''
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        
        

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        # 在此函数内进行数据加载，分割数据集等操作
        '''
        vocab = load_vocab()
        self.vocab_size = len(vocab)

        self.train, self.val, self.test = load_datasets()
        self.train_dims = self.train.next_batch.size()
        '''
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000]) # Pytorch的random_split是针对Dataset对象做分割

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=4, persistent_workers=True, pin_memory=True)

if __name__ == '__main__':

    # instrument experiment with W&B
    wandb_logger = WandbLogger(project="PL_MNIST")
    trainer = Trainer(logger=wandb_logger)



    net = myplmodule()
    mnist_dm = MyDataModule()
    trainer = Trainer(
        strategy='ddp',
        amp_backend='apex', 
        amp_level='O2', 
        gpus=-1,
        max_epochs=10,
        logger=wandb_logger,
        # profiler=SimpleProfiler(dirpath='./',filename='profiler.log')
        profiler=SimpleProfiler()
        )
    # x = torch.randn((32, 1, 28, 28))
    # out = net(x)
    # print(out.size())

    # log gradients and model topology
    wandb_logger.watch(net)

    trainer.fit(net, mnist_dm)
