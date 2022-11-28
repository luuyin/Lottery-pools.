# (AAAI 2023) Lottery Pools



<div align=center><img src="https://github.com/luuyin/Lottery-pools/blob/main/Cifar_results.png" width="900" height="200"></div>

<div align=center><b><small>Figure: Test accuracy % of the original LTs and Lottery Pools on CIFAR-10/100</small></b></div>


<div align=center><img src="https://github.com/luuyin/Lottery-pools/blob/main/Imagenet_results.png" width="520" height="200"></div>

<div align=center><b><small>Figure: Test accuracy % of the original LTs and Lottery Pools on ImageNet</small></b></div>


<br> 

**Lottery Pools: Winning More by Interpolating Tickets without Increasing Training or Inference Cost**<br>
Lu Yin, Shiwei Liu, Fang Meng, Tianjin Huang, Vlado Menkovski, Mykola Pechenizkiy<br>
https://arxiv.org/abs/2208.10842<br>

Abstract: *Lottery tickets (LTs) is able to discover accurate and sparse subnetworks that could be trained in isolation to match the performance of dense networks. Ensemble, in parallel, is one of the oldest time-proven tricks in machine learning to improve performance by combining the output of multiple independent models. However, the benefits of ensemble in the context of LTs will be diluted since ensemble does not directly lead to stronger sparse subnetworks, but leverages their predictions for a better decision. In this work, we first observe that directly averaging the weights of the adjacent learned subnetworks significantly boosts the performance of LTs. Encouraged by this observation, we further propose an alternative way to perform an 'ensemble' over the subnetworks identified by iterative magnitude pruning via a simple interpolating strategy. We call our method Lottery Pools. In contrast to the naive ensemble which brings no performance gains to each single subnetwork, Lottery Pools yields much stronger sparse subnetworks than the original LTs without requiring any extra training or inference cost. Across various modern architectures on CIFAR-10/100 and ImageNet, we show that our method achieves significant performance gains in both, in-distribution and out-of-distribution scenarios. Impressively, evaluated with VGG-16 and ResNet-18, the produced sparse subnetworks outperform the original LTs by up to 1.88% on CIFAR-100 and 2.36% on CIFAR-100-C; the resulting dense network surpasses the pre-trained dense-model up to 2.22% on CIFAR-100 and 2.38% on CIFAR-100-C.*


This code base is created by Lu Yin [l.yin@tue.nl](mailto:l.yin@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>

This repository contains implementaions of sparse training methods: [Lottery Tickets](https://arxiv.org/abs/1803.03635), [Lottery Tickets with rewinding](https://arxiv.org/abs/1912.05671), [Lottery Pools](https://arxiv.org/abs/2208.10842)


## Requirements 
The library requires Python 3.7, PyTorch v1.10.0, and CUDA v11.3.1. Other version of Pytorch should also work.

## How to Run Experiments


###  Options 

```
Options for creating lottery tickets
* --pruning_times - overall times of IMP pruning
* --rate - percentage of rate that has been pruned during each IMP pruning
* --prune_type - type of prune. Choose from lt (naive lottery tickets), rewind_lt (lottery tickets with rewinding)
* --rewind_epoch - epochs of rewinding


Options for lottery pools
* --search_num - the count of candidate lotter pools for interpolation
* --EMA_value -EMA factor for interpolation
* --interpolate_method - interpolation method, choices=['Lottery_pools','interpolate_ema', 'interpolate_swa']
* --interpolation_value_list - the candidate coefficient pools for interpolation

```
### CIFAR-10/100 Experiments
```
cd CIFAR
```
#### Create lottery tickets by IMP:
```
python -u main_imp.py --data ../data --dataset cifar100 --arch resnet18 --seed 41 --prune_type rewind_lt --rewind_epoch 9 	--pruning_times 19 
```
#### Lottery pools (average)
```
checkpoint=the path of CIFAR LTs solutions checkpoints

python Lottery_pools.py --interpolate_method Lottery_pools --rewind_epoch 9 --search_num 19 --interpolation_value_list 0.5 --arch resnet18 --data ../data --dataset cifar100  --seed 41 --inference --checkpoint  $checkpoint
```
####  Lottery pools (interpolation)
```
python Lottery_pools.py --interpolate_method Lottery_pools --rewind_epoch 9 --search_num 19 --interpolation_value_list 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 --arch resnet18 --data ../data --dataset cifar100  --seed 41 --inference --checkpoint  $checkpoint
```
####  Check  linear mode connectivity
```
python Lottery_pools.py --interpolate_method liner_inter --rewind_epoch 9 --search_num 19 --arch resnet18 --data ../data --dataset cifar100  --seed 41 --inference --checkpoint  $checkpoint
```
####  Baseline (1): Interpolation using SWA
```
python Lottery_pools.py --interpolate_method interpolate_swa --rewind_epoch 9 --search_num 19  --arch resnet18 --data ../data --dataset cifar100  --seed 41 --inference --checkpoint  $checkpoint
```
####  Baseline (2): Interpolation using EMA
```
python Lottery_pools.py --interpolate_method interpolate_ema --rewind_epoch 9 --search_num 19 --EMA_value 0.95 --arch resnet18 --data ../data --dataset cifar100  --seed 41 --inference --checkpoint  $checkpoint
```


### Imagenet Experiments

#### Create lottery tickets by IMP:
Please ref the  [OpenLTH](https://github.com/facebookresearch/open_lth) framework created by Jonathan Frankle


#### Lottery pools (average)
```
cd ImageNet

save_dir=the path of imagenet LTs solution checkpoints

data= tht path imagenet dataset 

python $1multiproc.py --nproc_per_node 2 $1Lottery_pools.py --save_dir $save_dir $2 $data --interpolation_value_list 0.5  --seed 17 --master_port 8020 -j32 -p 500 --arch imagenet_resnet_18  --interpolate_method Lottery_pools
```
####  Lottery pools (interpolation)
```
python $1multiproc.py --nproc_per_node 2 $1Lottery_pools.py --save_dir $save_dir $2 $data --interpolation_value_list 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 --seed 17 --master_port 8020 -j32 -p 500 --arch imagenet_resnet_18  --interpolate_method Lottery_pools
```
####  Baseline (1): Interpolation using SWA
```
python $1multiproc.py --nproc_per_node 2 $1Lottery_pools.py --save_dir $save_dir $2 $data --interpolation_value_list 0.5 --seed 17 --master_port 8020 -j32 -p 500 --arch imagenet_resnet_18  --interpolate_method interpolate_swa
```
####  Baseline (2): Interpolation using EMA
```
python $1multiproc.py --nproc_per_node 2 $1Lottery_pools.py --save_dir $save_dir $2 $data --interpolation_value_list 0.5 --seed 17 --master_port 8020 -j32 -p 500 --arch imagenet_resnet_18  --interpolate_method interpolate_ema
```
# Citation

if you find this repo is helpful, please cite

```bash
@article{yin2022lottery,
  title={Lottery Pools: Winning More by Interpolating Tickets without Increasing Training or Inference Cost},
  author={Yin, Lu and Liu, Shiwei and Meng, Fang and Huang, Tianjin and Menkovski, Vlado and Pechenizkiy, Mykola},
  journal={arXiv preprint arXiv:2208.10842},
  year={2022}
}


```
