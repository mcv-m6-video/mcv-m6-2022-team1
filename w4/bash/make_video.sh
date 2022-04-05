result_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models_testS03
data_path=/home/pau/Documents/datasets/aicity/train/S03

regex="(c[0-9][0-9][0-9])"

for model in faster_fpn faster_dc5 faster_c4 retina_fpn
do
  allpaths=("$result_path"/"$model"/*)
  for fullpath in "${allpaths[@]}"
  do
    echo $fullpath
    if [[ "$fullpath" =~ $regex ]]
    then
      seq="${BASH_REMATCH[1]}"
      echo $seq
      python3 ../misc/make_video.py "$data_path"/"$seq"/gt/gt_coco.json \
                                    "$data_path"/"$seq"/vdo_frames \
                                    "$fullpath"/coco_instances_results.json \
                                    "$fullpath"/demo.mp4 \
                                    540 840
    fi
  done
done