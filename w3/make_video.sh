# Untrained #################

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test_cocoid.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/data/results/COCO-Detection/retinanet_R_50_FPN_1x.yaml/holdout/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/holdout_retina.mp4 \
                      540 840

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test_cocoid.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/data/results/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml/holdout/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/holdout_fasterfpn.mp4 \
                      540 840

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test_cocoid.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/data/results/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml/holdout/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/holdout_fasterc4.mp4 \
                      540 840

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test_cocoid.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/data/results/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml/holdout/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/holdout_fasterdc5.mp4 \
                      540 840

# Trained #################

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/faster_fpn/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/trained_fasterfpn.mp4 \
                      540 840

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/faster_dc5/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/trained_fasterdc5.mp4 \
                      540 840

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/faster_c4/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/trained_fasterc4.mp4 \
                      540 840

python3 make_video.py /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/gt_coco/holdout/gt_all_test.json \
                      /home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/S03/c010/vdo_frames \
                      /home/pau/Documents/master/M6/project/repo/w3/results/train_holdout/retina_r50/coco_instances_results.json \
                      /home/pau/Documents/master/M6/project/data/results/w3videos/trained_retinafpn.mp4 \
                      540 840