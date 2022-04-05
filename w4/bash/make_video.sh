result_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models
data_path=/home/pau/Documents/datasets/aicity/train/S03/c010
out_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models

for model in faster_fpn faster_dc5 faster_c4 retina_fpn
do
python3 ../make_video.py $data_path/gt/gt_coco.json \
                      $data_path/vdo_frames \
                      $result_path/$model/ai_citiesS03c010/coco_instances_results.json \
                      $out_path/demo.mp4 \
                      540 840
done