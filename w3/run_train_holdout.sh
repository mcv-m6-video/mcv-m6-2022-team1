python3 train_arch.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/faster_fpn \
                      COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

python3 train_arch.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/faster_dc5 \
                      COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml

python3 train_arch.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/faster_c4 \
                      COCO-Detection/faster_rcnn_R_50_C4_3x.yaml

python3 train_arch.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/retina_r50 \
                      COCO-Detection/retinanet_R_50_FPN_1x.yaml