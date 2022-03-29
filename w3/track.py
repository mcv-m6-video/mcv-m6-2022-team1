class Track(object):
    def __init__(self, id: int):
        self.id = id  # track id
        self.bbox = list()  # list of bbox associated to an object
        self.frame_id_appearence = list()

    def append_bbox(self, new_bbox):
        self.bbox.append(new_bbox)

    def append_frame_id_appearence(self, frame_id_):
        self.frame_id_appearence.append(frame_id_)
