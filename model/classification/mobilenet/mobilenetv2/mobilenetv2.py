import sys
import os
sys.path.append(os.getcwd())
from TensorRT.Infer_Interface_TensorRT import Infer_TensorRT as Infer_Interface
import numpy as np
import cv2
from PIL import Image
class MobilenetV2:
    def __init__(self,preprocessor_args,postprocessor_args,model_args):
        self.pre = Preprocess(**preprocessor_args)
        self.Infer = Infer_Interface(**model_args)
        self.post = Postprocess(**postprocessor_args)
    def model_infer(self, bgr_img):
        input_data= self.pre.process(bgr_img)  #输出input_data为(1,c,h,w)
        trt_outputs = self.Infer.infer(input_data)
        category = self.post.process(trt_outputs)
        return category
    def model_destory(self,):
        self.Infer.destory()

class Preprocess:
    def __init__(self, input_resolution, mean_list,std_list):
        self.input_resolution = input_resolution
        self.mean_array = np.array(mean_list, dtype= np.float32) * 255
        self.std_array = np.reciprocal(np.array(std_list, dtype= np.float32)* 255, dtype=np.float32)
    def process(self, rgb_img):
        img = cv2.resize(rgb_img,(self.input_resolution,self.input_resolution))
        #img = rgb_img.resize((self.input_resolution, self.input_resolution))
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
    def __init__(self, output_shape,top_num,labels_map):
        self.output_shape = output_shape
        self.top_num = top_num
        self.labels_map = labels_map
    def process(self, output):
        output = output[0].reshape(self.output_shape)
        # output = self.softmax(output,1)
        self.top_info(output)
        output = np.argmax(output)
        return output
        
    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)
    def top_info(self,output):
        for i ,out in enumerate(output):
            top_ind = np.argsort(out)[-self.top_num:][::-1]
                #print("Image {}\n".format(args.input[i]))
                #print(classid_str, probability_str)
                #print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
            for id in top_ind:
                det_label = self.labels_map[id] if self.labels_map else "{}".format(id)
                label_length = len(det_label)
                space_num_before = (7 - label_length) // 2
                space_num_after = 7 - (space_num_before + label_length) + 2
                space_num_before_prob = (11 - len(str(out[id]))) // 2
                print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                            ' ' * space_num_after, ' ' * space_num_before_prob,
                                            out[id]))
            print("\n")

if __name__ == '__main__':
    Params = [
        {"input_resolution":224,
            "mean_list":[0.485, 0.456, 0.406],
            "std_list":[0.229, 0.224, 0.225],},
        { "output_shape":[1,1000],
        "top_num":5,
        "labels_map":None},
        { "trt_path":"./model_zoo/trt/mobilenet_v2.trt",
            "gpuID":0}]
    class_model = MobilenetV2(Params[0], Params[1], Params[2])
    img = cv2.imread("./data/img2.jpg")
    input_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB);
    #input_image = Image.open("./data/img2.jpg")
    output = class_model.model_infer(input_image)
    print(output)
    class_model.model_destory()
