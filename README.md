# PointCloutNorm

This is the repo for point clouds normalization. It currently supports the rotation regression based on PointNet.

## Enviroment

This project is implemented based on PyTorch 1.5. Activate the training environment with the following command for the corresponding servers.

For SH40, run the following command:

```
source env40.sh
```

For SH36, run the following command:

```
source env36.sh
```

## Training

Some training parameters are set as follows by default:

|      Param      |   Value   |
| :-------------: | :-------: |
|   batch_size   |    64    |
|     epochs     |    30    |
|       LR       |   1e-3   |
|     npoints     |   30000   |
| width_threshold | [-20, 20] |

Modify the specific training parameter with training command. To start training process, run the following command:

```
sh script/slurm_training.sh train TRAINING_NODE GPU_NUM --model_tag CUSTOM_TAG

```

An example of training on SH36:

```
sh script/slurm_training.sh train ad_lidar 8 --model_tag baseline 
```

The trained model parameters are saved in ./output/model_tag/ckpt. If you want to train a model with different depth or width threshold, add the corresponding paramter like:

```
sh script/slurm_training.sh train ad_lidar 8 --model_tag baseline --depth_threshold --width_threshold
```

## Correction

To obtain the final rotation result for a point sequence, run the following command:

```
sh script/slurm_correct.sh correct TRAINING_NODE GPU_NUM --model_tag CUSTOM_TAG --pretrained_model MODEL_PATH --pkl_file PKL_FILE_PATH
```

If you want to have a detailed result about the abnormal predictions, you can set an error threshold to select the corresponding point sequence. To be specific, run the following command:

```
sh script/slurm_correct.sh correct TRAINING_NODE GPU_NUM --model_tag CUSTOM_TAG --pretrained_model MODEL_PATH --pkl_file PKL_FILE_PATH --to_file --max_threshold MAX_THRESHOLD --min_threshold MIN_THRESHOLD
```

A file named abnormal.txt wil be saved in ./output/model_tag/ with selected point sequence index and its predicted rotation which is greater than the MAX_THRESHOLD or less than the MIN_THRESHOLD.

## Pretrained Model
An avaiable pretrained model is saved in output/model_best/best.pth


