#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 4096 # 4GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

for arch in COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
            COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml \
            COCO-Detection/faster_rcnn_R_50_C4_3x.yaml \
            COCO-Detection/retinanet_R_50_FPN_1x.yaml
do
  for i in 0 1 2 3
  do
    echo "${i}-fold w/ arch: ${arch}"
    python3 off_shelf.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/kfold_${i} \
                         /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                         /home/pau/Documents/master/M6/project/data/results/${arch}/fold_${i} \
                         ${arch}
  done
done
