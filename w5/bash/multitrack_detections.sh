#!/bin/bash

result_paths=/home/pau/Documents/master/M6/project/repo/w4/results
dataset_root=/home/pau/Documents/datasets/aicity/train

weights_path=/home/pau/Documents/master/M6/project/repo/w5/results/margin02/weights/weights_5.pth
for model in $result_paths/train_models_testS03/*
do
  echo "$model"
  python3 ../multicamera_tracking.py  "$dataset_root" \
                                      "$weights_path" \
                                      "$model" \
                                      "track_purge.txt" \
                                      "average"

  python3 ../eval_multi_cam.py "$dataset_root" \
                               "$model" \
                               "$model/summary.txt"
done

weights_path=/home/pau/Documents/master/M6/project/repo/w5/results/margin02_s01/weights/weights_5.pth
for model in $result_paths/train_models_testS01/*
do
  echo "$model"
  python3 ../multicamera_tracking.py  "$dataset_root" \
                                      "$weights_path" \
                                      "$model" \
                                      "track_purge.txt" \
                                      "average"

  python3 ../eval_multi_cam.py "$dataset_root" \
                               "$model" \
                               "$model/summary.txt"
done

weights_path=/home/pau/Documents/master/M6/project/repo/w5/results/margin02_s04/weights/weights_5.pth
for model in $result_paths/train_models_testS04/*
do
  echo "$model"
  python3 ../multicamera_tracking.py  "$dataset_root" \
                                      "$weights_path" \
                                      "$model" \
                                      "track_purge.txt" \
                                      "average"

  python3 ../eval_multi_cam.py "$dataset_root" \
                               "$model" \
                               "$model/summary.txt"
done
