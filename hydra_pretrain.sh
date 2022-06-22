#!/bin/bash
ARCH="vgg16_bn" # vgg16_bn, wrn_28_4, resnet18, resnet50 (only for imagenet)
DATASET="CIFAR10"  # CIFAR10, SVHN, imagenet (only for resnet50)
P_Rate=0.1
P_Reg='weight'  # weight, channel
GPU="0"

echo "HYDRA pruning with non-uniform strategy"

if [ $ARCH == 'ResNet50' ] && [ $DATASET == 'imagenet' ]
then
  python train.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'pretrain' --k $P_Rate --prune_reg $P_Reg --epochs 100 --lr 0.1
elif [ $ARCH != 'ResNet50' ] && [ $DATASET != 'imagenet' ]
then
  python train_imagenet.py --arch $ARCH --dataset $DATASET --gpu $GPU --exp-mode 'pretrain' --k $P_Rate --prune_reg $P_Reg --epochs 100 --lr 0.01
else
  echo "Model does not match dataset."
fi
