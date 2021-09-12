import sys
import os
sys.path.append(os.getcwd())
from TensorRT.Infer_Interface_TensorRT import Infer_TensorRT as Infer_Interface
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import torch
from torchvision import transforms
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
class Unet:
    def __init__(self,preprocessor_args,postprocessor_args,model_args):
        self.pre = Preprocess(**preprocessor_args)
        self.Infer = Infer_Interface(**model_args)
        #self.post = Postprocess(**postprocessor_args)
    def model_infer(self, rgb_img):
        input_data= self.pre.process(rgb_img)  #输出input_data为(1,c,h,w)
        trt_outputs = self.Infer.infer(input_data)
        mask = np.array(trt_outputs, dtype=np.float32, order='C')
        #mask = self.post.process(trt_outputs)
        return mask
    def model_destory(self,):
        self.Infer.destory()

class Preprocess:
    def __init__(self, input_resolution,mean_list,std_list):
        self.input_resolution = input_resolution
        self.mean_array = np.array(mean_list, dtype= np.float32) * 255
        self.std_array = np.reciprocal(np.array(std_list, dtype= np.float32)* 255, dtype=np.float32)
    def process(self, rgb_img):
        img = cv2.resize(rgb_img,(self.input_resolution,self.input_resolution))
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

# class Postprocess:
#     def __init__(self, scale, class_num, out_threshold):
#         self.scale = scale
#         self.class_num = class_num
#         self.out_threshold = out_threshold
#     def process(self, output, rgb_img):
#         w, h = rgb_img.size
#         newW, newH = int(self.scale * w), int(self.scale * h)
#         assert newW > 0 and newH > 0, 'Scale is too small'
#         output = output[0].reshape(1,self.class_num, newW, newH)
#         if self.class_num > 1:
#             mask = self.softmax(output, 1)
#         else:
#             mask = self.sigmoid(output)
#         return mask > self.out_threshold
#     def softmax(self, x, axis=None):
#         x = x - x.max(axis=axis, keepdims=True)
#         y = np.exp(x)
#         return y / y.sum(axis=axis, keepdims=True)
#     def sigmoid(x):
#         return 1/(1+np.exp(-x))


if __name__ == '__main__':
    Params = [
         {"input_resolution":640,
          "mean_list":[0, 0, 0],
            "std_list":[1, 1, 1]},
             { "topk":5},
        { "trt_path":"./model_zoo/trt/unet-sim.trt",
            "gpuID":0}]
    class_model = Unet(Params[0], Params[1], Params[2])
    img = cv2.imread("./data/car.bmp")
    input_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB);
    #input_image = Image.open("./data/img2.jpg")
    output = class_model.model_infer(input_image)
    output = output.reshape((1,2,640,640))
    b = torch.from_numpy(output)
    probs = F.softmax(b, dim=1)[0]
    #print(out.shape)
    # for index, score in zip(indexes, scores):
    #     print("%d=%f" % (index, score))
    #class_model.model_destory()
    tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img.shape[0], img.shape[1])),
            transforms.ToTensor()
        ])
    full_mask = tf(probs).squeeze()
    mask = F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()
    out_filename = "./car_mask.bmp"
    result = mask_to_image(mask)
    result.save(out_filename)
    class_model.model_destory()

