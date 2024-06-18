import onnxruntime
import cv2
import numpy as np
import time
from darknetWrapper import darknet
from nms import nms_python
# weight_path = '/home/tevel/workspace/data/create_tree_map/tinyyolo4_3l_generic/yolo4-tiny-3l_best_yolov4_1_3_480_640_static.onnxs'
# weight_path = '/home/tevel/workspace/data/weights_object_detection/fp32_fp16/yolov4_1_3_480_640_static.onnx'
# weight_path = '/home/tevel/workspace/data/weights_object_detection/fp32_fp16/yolov4_1_3_480_640_static.onnx'
# image_path = '/home/tevel/workspace/data/weights_object_detection/sample.png'
image_path = '/workspace/models/fp32_fp16/sample.png'
# weight_path = '/workspace/models/fp32_fp16/yolov4_1_3_480_640_static.onnx'
weight_path = '/workspace/models/fp32_fp16/yolov4_1_3_480_640_static_fp16.onnx'
# im = cv2.imread(image_path)
# im = im[...,::-1]
# session = onnxruntime.InferenceSession(weight_path,providers=['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider'])
# outputs = session.run(None, {'input': [np.transpose(im,(2,0,1))]})
# print(outputs)
# for i in range(len(outputs)):
#
#     print('-----------------')

def preprocess_image_16bit(image_path, width = 640, height = 480):
    image_size = (640, 480)
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size, cv2.INTER_AREA)
    img_original = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float16)
    img = (img ) / 255.
    img = np.expand_dims(img, axis=0)
    return img, img_original

def preprocess_image_32bit(image_path, width = 640, height = 480):
    image_size = (640, 480)
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size, cv2.INTER_AREA)
    img_original = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (img ) / 255.
    img = np.expand_dims(img, axis=0)
    return img, img_original


def show_inference_results(img, result, threshold=0.5,width = 640, height = 480):
    indx = np.where(result[1][0] > threshold)[0]
    

    for bbox in result:
        bbox = bbox.squeeze() #result[0][0][i][0]
        img =cv2.rectangle(img, (int(bbox[0]*width), int(bbox[1]*height)), (int(bbox[2]*width), int(bbox[3]*height)), (0, 255, 0), 2)

        # print('-----------------')
        # print(f'Output {i}:')
        # print(result[i].shape)
        # print(result[i])
        # print('-----------------')
    cv2.imshow('Frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

session = onnxruntime.InferenceSession(weight_path, None ,providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
print(f'The NN is runnning on {onnxruntime.get_device()}')
img, img_original = preprocess_image_16bit(image_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
dt =0
N_exp = 1000
detection_th=0.45
result = session.run(None, {input_name: img})
for i in range(N_exp):
    t = time.time()
    result = session.run(None, {input_name: img})
    ind_th = np.where(result[1][0]>detection_th)[0]
    indexes = nms_python(result[0][0][ind_th].squeeze(),result[1][0][ind_th].squeeze(),0.9)
    dt += time.time()-t
     

print(f'execution time = {dt/N_exp:.4f}')
show_inference_results(img_original, result[0][0][ind_th[indexes]], 0.8)
# matte = np.squeeze(result[0])
# cv2.imshow('Matte', (matte * 255.).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
