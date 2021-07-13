import sys
import os
sys.path.append(os.getcwd())
from TensorRT.Infer_Interface_TensorRT import Infer_TensorRT as Infer_Interface
import numpy as np
import cv2
from PIL import Image
class Alexnet:
    def __init__(self,preprocessor_args,postprocessor_args,model_args):
        self.pre = Preprocess(**preprocessor_args)
        self.Infer = Infer_Interface(**model_args)
        self.post = Postprocess(**postprocessor_args)
    def model_infer(self, bgr_img):
        input_data= self.pre.process(bgr_img)  #输出input_data为(1,c,h,w)
        trt_outputs = self.Infer.infer(input_data)
        mask = self.post.process(trt_outputs)
        return mask
    def model_destory(self,):
        self.Infer.destory()

class Preprocess:
    def __init__(self, scale):
        self.scale = scale
    def process(self, rgb_img):
        w, h = rgb_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img = rgb_img.resize((newW, newH))
        input_data = self._normalize(img)
        return input_data
    def _normalize(self, image_array):
        image_array = np.array(image_array, dtype=np.float32, order='C')
        image_array = image_array /255
        image_array = np.transpose(image_array, [2,0,1])
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.array(image_array, dtype=np.float32, order='C')
        return image_array

class Postprocess:
    def __init__(self, scale, class_num, out_threshold):
        self.scale = scale
        self.class_num = class_num
        self.out_threshold = out_threshold
    def process(self, output, rgb_img):
        w, h = rgb_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        output = output[0].reshape(1,self.class_num, newW, newH)
        if self.class_num > 1:
            mask = self.softmax(output, 1)
        else:
            mask = self.sigmoid(output)
        return mask > self.out_threshold
    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)
    def sigmoid(x):
        return 1/(1+np.exp(-x))
3

if __name__ == '__main__':
    Params = [
        {"scale": 1},
        { "scale": 1,
         "class_num": 1, 
         "out_threshold": 0.5},
        { "trt_path":"./net/alexnet.trt",
            "gpuID":0}]
    class_model = Alexnet(Params[0], Params[1], Params[2])
    #img = cv2.imread("./data/img.jpg")
    input_image = Image.open("./data/img2.jpg")
    output = class_model.model_infer(input_image)
    print(output)
    class_model.model_destory()

