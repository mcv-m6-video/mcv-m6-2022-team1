import cv2
from tqdm.auto import tqdm
from pathlib import Path
from utils import data, viz

#%%

who = "marcos"

if who == "pau":
    gt_path = Path('/home/pau/Documents/datasets/aicity')
elif who == "marcos":
    gt_path = Path('/home/cisu/PycharmProjects/mcv-m6-2022-team1/w5/DATA')
else:
    gt_path = Path('/home/pau/Documents/datasets/aicity')

for sequence in (gt_path / "train").glob("*"):
    for camera in sequence.glob("*"):
        output = camera / "cars"
        output.mkdir(exist_ok=True)
        anns = data.load_annotations(str(camera / "gt" / "gt.txt"))

        unique_frame = anns["frame"].unique()
        unique_frame.sort()

        unique_id = anns["ID"].unique()
        for ii in unique_id:
            (output / f"{ii}").mkdir(exist_ok=True)

        for ind in tqdm(unique_frame,desc=f"Building dataset for {str(camera).split('/')[-1]}"):
            slice = anns[anns["frame"] == ind]

            img = cv2.imread(str(camera / "vdo_frames" / f"{ind:05d}.jpg"))
            # viz.imshow(img[:, :, ::-1])
            for frame, track_id, left, top, width, height, _, _, _, _ \
                    in slice.itertuples(index=False):
                car = img[top:top+height, left:left+width, :]
                # viz.imshow(car)
                cv2.imwrite(str(output / f"{track_id}" / f"{frame}.png"), car)

#%%
str(camera / "vdo_frames" / f"{ind:05d}.jpg")

