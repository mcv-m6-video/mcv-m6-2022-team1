#!/bin/bash

result_paths=(/home/pau/Documents/master/M6/project/repo/w4/results/train_models_test*)
dataset_root=/home/pau/Documents/datasets/aicity/train/

for testseq in "${result_paths[@]}"
do
  regx="(S[0-9][0-9])"
  if [[ "$testseq" =~ $regx ]]
  then
    seq="${BASH_REMATCH[1]}"
    for model in $testseq/*
    do
      campaths=("$model"/ai_cities"${seq}"c*)
      for campath in "${campaths[@]}"
      do
        regx="(c[0-9][0-9][0-9])"
        if [[ "$campath" =~ $regx ]]
        then
          cam="${BASH_REMATCH[1]}"
          echo $campath $cam $seq

          nframes=$(find "$dataset_root"/"$seq"/"$cam"/vdo_frames/*.jpg | wc --lines)

          echo There are $nframes frames

          python3 ../task_2.py "$campath" $nframes --purge
          python3 ../eval_track.py "$dataset_root"/"$seq"/"$cam"/gt/gt.txt \
                                   "$campath"/track_purge.txt \
                                   "$campath"/summary_purge.csv \
                                   0 "$nframes"
        fi
      done
    done
  fi
done
