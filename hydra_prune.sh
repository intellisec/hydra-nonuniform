#!/bin/bash
ARCH="vgg16_bn" # vgg16_bn, wrn_28_4, resnet18, resnet50 (only for imagenet)
DATASET="CIFAR10"  # CIFAR10, SVHN, imagenet (only for resnet50)
P_Rate=0.1
P_Reg='weight'  # channel weight
STG_ID='010_0'  # strategy ID that you named for found strategy in .json strategy file
GPU="0"

echo "HYDRA pruning with non-uniform strategy"

if [ $ARCH == 'ResNet50' ] && [ $DATASET == 'imagenet' ]
then
  python train_imagenet.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'prune' --k $P_Rate --prune_reg $P_Reg --stg_id $STG_ID --epochs 20 --lr 0.1
  python train_imagenet.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'finetune' --k $P_Rate --prune_reg $P_Reg --stg_id $STG_ID --epochs 100 --lr 0.01
elif [ $ARCH != 'ResNet50' ] && [ $DATASET != 'imagenet' ]
then
  python train.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'prune' --k $P_Rate --prune_reg $P_Reg --stg_id $STG_ID --epochs 20 --lr 0.1
  python train.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'finetune' --k $P_Rate --prune_reg $P_Reg --stg_id $STG_ID --epochs 100 --lr 0.01
else
  echo "Model does not match dataset."
fi

if [ $ARCH == 'ResNet50' ] && [ $DATASET == 'imagenet' ]
then
  python test_imagenet.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'finetune' --k $P_Rate --prune_reg $P_Reg --stg_id $STG_ID --evaluate
else
  echo "Evaluate robustness against FGSM, PGD-20 and CW attacks"
  python test_adv.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'finetune' --k $P_Rate --prune_reg $P_Reg --stg_id $STG_ID --evaluate
fi
