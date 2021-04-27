# Code for RIADD (ISBI-2021)
![image](https://github.com/Hanson0910/Pytorch-RIADD/blob/main/show-img/show-img.png)
This is the source code for my solution to the [RIADD (ISBI-2021)](https://riadd.grand-challenge.org/evaluation/challenge/leaderboard/)
hosted by Grand Challenge
### Requirement
- opencv_python==4.4.0.44
- tqdm==4.48.2
- pandas==1.1.1
- albumentations==0.4.6
- torchvision==0.6.0a0+35d732a
- torch==1.5.1
- visdom==0.1.8.9
- apex==0.9.10dev

### Update Log
- [2021-2-28] Release the training codes
- [2021-4-14] publish final model weights

### TODO
- publish partial model weights

### Dataset
https://www.kaggle.com/sssdey1702/riadd-rfmid

### Model Weights

https://www.kaggle.com/hanson0910/tf-efficientnet-b5-ns960

### Train Model with DDP

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_riadd_eb6_ddp.py
```
### Train Model without DDP
```
python train_riadd_eb6.py
```
### Create submission
```
python subnmit_riadd.py
```

### References
Appreciate the great work from the following repositories:

- https://github.com/rwightman/pytorch-image-models
- https://github.com/zlannnn
