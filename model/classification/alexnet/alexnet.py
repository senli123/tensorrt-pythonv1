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
        is_nbx = self.post.process(trt_outputs)
        return is_nbx
    def model_destory(self,):
        self.Infer.destory()

class Preprocess:
    def __init__(self, input_resolution, mean_list,std_list):
        self.input_resolution = input_resolution
        self.mean_array = np.array(mean_list, dtype= np.float32) * 255
        self.std_array = np.reciprocal(np.array(std_list, dtype= np.float32)* 255, dtype=np.float32)
    def process(self, rgb_img):
        #img = cv2.resize(rgb_img,(self.input_resolution,self.input_resolution), interpolation=cv2.INTER_LINEAR)
        img = rgb_img.resize((self.input_resolution, self.input_resolution))
        input_data = self._normalize(img)
        return input_data
    def _normalize(self, image_array):
        image_array = np.array(image_array, dtype=np.float32, order='C')
        image_array -= self.mean_array
        image_array *= self.std_array
        image_array = np.transpose(image_array, [2,0,1])
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.array(image_array, dtype=np.float32, order='C')
        return image_array

class Postprocess:
    def __init__(self, output_shape):
        self.output_shape = output_shape
    def process(self, output):
        output = output[0].reshape(self.output_shape)
        output = self.softmax(output,1)
        output = np.argmax(output)
        return output
    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

if __name__ == '__main__':
    Params = [
        {"input_resolution":224,
            "mean_list":[0.485, 0.456, 0.406],
            "std_list":[0.229, 0.224, 0.225],},
        { "output_shape":[1,1000]},
        { "trt_path":"./net/alexnet.trt",
            "gpuID":0}]
    class_model = Alexnet(Params[0], Params[1], Params[2])
    #img = cv2.imread("./data/img.jpg")
    input_image = Image.open("./data/img2.jpg")
    output = class_model.model_infer(input_image)
    print(output)
    class_model.model_destory()

