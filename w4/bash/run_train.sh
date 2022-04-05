dataset_path=/home/pau/Documents/datasets/aicity/train/
output_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models_testS03
train="S01 S04"
test="S03"

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_fpn \
                          COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
                          --train_on $train \
                          --test_on $test

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_dc5 \
                          COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml \
                          --train_on $train \
                          --test_on $test

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_c4 \
                          COCO-Detection/faster_rcnn_R_50_C4_3x.yaml \
                          --train_on $train \
                          --test_on $test


python3 ../train_archs.py $dataset_path \
                          $output_path/retina_fpn \
                          COCO-Detection/retinanet_R_50_FPN_1x.yaml \
                          --train_on $train \
                          --test_on $test


dataset_path=/home/pau/Documents/datasets/aicity/train/
output_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models_testS01
train="S03 S04"
test="S01"

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_fpn \
                          COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
                          --train_on $train \
                          --test_on $test

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_dc5 \
                          COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml \
                          --train_on $train \
                          --test_on $test

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_c4 \
                          COCO-Detection/faster_rcnn_R_50_C4_3x.yaml \
                          --train_on $train \
                          --test_on $test


python3 ../train_archs.py $dataset_path \
                          $output_path/retina_fpn \
                          COCO-Detection/retinanet_R_50_FPN_1x.yaml \
                          --train_on $train \
                          --test_on $test

dataset_path=/home/pau/Documents/datasets/aicity/train/
output_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models_testS04
train="S01 S03"
test="S04"


python3 ../train_archs.py $dataset_path \
                          $output_path/faster_fpn \
                          COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
                          --train_on $train \
                          --test_on $test

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_dc5 \
                          COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml \
                          --train_on $train \
                          --test_on $test

python3 ../train_archs.py $dataset_path \
                          $output_path/faster_c4 \
                          COCO-Detection/faster_rcnn_R_50_C4_3x.yaml \
                          --train_on $train \
                          --test_on $test


python3 ../train_archs.py $dataset_path \
                          $output_path/retina_fpn \
                          COCO-Detection/retinanet_R_50_FPN_1x.yaml \
                          --train_on $train \
                          --test_on $test