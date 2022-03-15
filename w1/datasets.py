import xml.etree.ElementTree as ET


def generate_gt_from_xml(in_path: str, out_path: str) -> None:
    dataset = ET.parse(in_path).getroot()

    # Separate annotations by frames. We do not care about the classes for the
    # time being, we only grab cars

    annotations = []

    # [frame, ID, left, top, width, height, 1, -1, -1, -1]
    for track in dataset.findall("track"):
        if track.attrib["label"] == "car":
            for box in track.findall("box"):
                annot = (
                    box.attrib["frame"], track.attrib["id"],
                    box.attrib["xtl"], box.attrib["ytl"],
                    float(box.attrib["xbr"]) - float(box.attrib["xtl"]),
                    float(box.attrib["ybr"]) - float(box.attrib["ytl"]),
                    1, -1, -1, -1
                )

                # Some functional magic to convert all elements in the tuple to
                # strings of integer numbers
                annot = tuple(map(lambda x: str(int(float(x))), annot))
                annotations.append(annot)

    with open(out_path, 'w') as f_gt:
        for ii in annotations:
            f_gt.write(",".join(ii) + "\n")