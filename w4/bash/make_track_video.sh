result_path=/home/pau/Documents/master/M6/project/repo/w4/results/train_models_testS04
data_path=/home/pau/Documents/datasets/aicity/train/S04

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
      python3 ../misc/make_track_video.py "$data_path"/"$seq"/gt/gt.txt\
                                          "$data_path"/"$seq"/vdo_frames \
                                          "$fullpath"/track_purge.txt \
                                          "$fullpath"/demo_purge.mp4 \
                                          1 221
    fi
  done
done