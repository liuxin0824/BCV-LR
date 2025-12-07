# Videos are Sample-Efficient Supervisions: Behavior Cloning from Videos via Latent Representations 


![Example Image](BCV-LR.png)


This repository provides the continuous implementation of BCV-LR. For discrete control, see [BCV-LR-discrete](https://github.com/liuxin0824/BCV-LR-discrete/).



## Data preparation
Download the expert video data [here](https://www.modelscope.cn/datasets/lxcasia/continuous_expertvideos_1M) and place it in the expert_data directory. Please rename the .npz file and make the expert_data dir look like this:

```
expert_data
   --- finger250
      --- train
         --- 100000.npz
   --- reacherhard250
      --- train
         --- 100000.npz
   --- pointeasy250
      --- train
         --- 100000.npz
...
```

## Conda env

Enter the repository and use conda to create a environment.
```
cd BCV-LR

conda env create -f environment.yml
```

Use tmux to create a terminal (optional) and then enter the created conda environment:
```
tmux

conda activate BCV-LR
```


## Offline stage

Make the env_name in config.yaml match the expert videos name (for example, to run Finger_spin, set env_name to finger250), and then achieve offline pretraining by:

```
python run_offline.py
```

You will find the pre-trained models in the folder exp_results, like:

```
exp_results
   --- 20251206
      --- 235527finger250
         --- model.pt......
...
```

## Online stage

revise the exp_name in the config.yaml to match the offline pre-trained models. For example, to run the finger experiments above, you should revise the exp_name to:
```
###config.yaml
exp_name: 20251206-235527finger250
```

then you can run the online stage:
```
python run_online.py
```
you can find the results in eval.csv:
```
exp_results
   --- 20251206
      --- 235527finger250
         --- model.pt...
         --- 20251207
            --- 112057_seed2
               --- eval.csv
...
```





## Citation


```
@inproceedings{
liu2025videos,
title={Videos are Sample-Efficient Supervisions: Behavior Cloning from Videos via Latent Representations},
author={Xin Liu and Haoran Li and Dongbin Zhao},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=cx1KfZerNY}
}
```

## Acknowledgement
This implementation is built on [DrQV2](https://github.com/facebookresearch/drqv2) and [LAPO](https://github.com/schmidtdominik/LAPO).










