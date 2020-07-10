import os
from PIL import Image
from face_dect_recong.align.detector import detect_faces
from face_dect_recong.align.visualization_utils import show_results

__cwd__ = os.path.dirname(os.path.abspath(__file__))

img = Image.open(os.path.join(__cwd__, '../data/other_my_face/my/my/myf112.jpg'))
bounding_boxes, landmarks = detect_faces(img)  # detect bboxes and landmarks for all faces in the image
show_results(img, bounding_boxes, landmarks)  # visualize the results