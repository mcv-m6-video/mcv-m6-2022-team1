class Track(object):
    def __init__(self, id: int, detections: int):
        self.id = id  # track id
        self.detections = detections  # stores num of detections
        self.bbox = list()  # list of bbox associated to an object

    def append_bbox(self, new_bbox):
        self.append(new_bbox)
