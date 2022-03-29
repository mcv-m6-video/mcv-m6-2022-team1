for i in *.mp4;
do
  ffmpeg -y -i "$i"  -vf scale=720:-2,setsar=1:1 "${i%.*}.gif";
done
