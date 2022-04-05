#!/bin/bash

# Quick script to create all vdo folders with frames and convert all datasets
# into COCO format.
# Run with a single argument in double quotes with the desired path expansion
# Example:

elms=($1)

echo "* * * * * * * * * * * * * * * * * * * * * * * * * *"
echo "  -- Convert AI cities dataset into COCO format    "
echo "  -- Number of folders to explore: ${#elms}        "
echo "* * * * * * * * * * * * * * * * * * * * * * * * * *"

echo "${elms[@]}"

for path in "${elms[@]}";
do
  echo ""
  echo "Processing ${path}"
  echo "* * * * * * * * * * * * * * * * * * * * * * * * * *"
  echo ""

  mkdir "${path}/vdo_frames/"
  # ffmpeg -i "${path}/vdo.avi" "${path}/vdo_frames/%05d.jpg"
  nframes=$(find "${path}"/vdo_frames/*.jpg | wc --lines)
  python3 ../misc/cvt_coco_gt.py "${path}/gt/gt.txt" "${nframes}"
done
