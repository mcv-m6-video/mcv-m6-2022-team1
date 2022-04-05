dataset_path=/home/pau/Documents/datasets/aicity/train/
output_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_fpn \
                          COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_dc5 \
                          COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_c4 \
                          COCO-Detection/faster_rcnn_R_50_C4_3x.yaml

python3 ../train_archs.py $dataset_path \
                          $output_path/retina_fpn \
                          COCO-Detection/retinanet_R_50_FPN_1x.yaml