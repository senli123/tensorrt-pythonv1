from utils.calibrator import Int8EntropyCalibrator
from TensorRT import common
import time
import numpy as np
import os 
import pickle
from PIL import Image
from densenet import Preprocess, Postprocess
class DenseNetEntropyCailbrator(Int8EntropyCalibrator):
    @classmethod
    def get_data(cls, training_data, cache_file, batch_size):
        cailb=cls(training_data, cache_file, batch_size)
        #返回的是一个初始化后的类
        return cailb

class loader():
    @classmethod
    def process_data(cls, file_path, data_set, label_set, input_resolution, mean_list, std_list):
        cls.preprocess = Preprocess(input_resolution, mean_list, std_list)
        data_file = os.path.join(file_path, data_set)
        label_file = os.path.join(file_path,label_set)
        with open(data_file, 'rb') as Dfile:   
            name_list = pickle.load(Dfile)
        with open(label_file, 'rb') as Lfile:   
            label_list = pickle.load(Lfile)
        cls.img_data = np.zeros([len(name_list), 3, input_resolution, input_resolution],dtype=np.float32)
        for index, img_name in enumerate(name_list):
            img = Image.open(img_name)
            img = cls.preprocess.process(img)
            cls.img_data[index] = img
        cls.label = np.array(label_list)
    def load_data(self,):
        return self.img_data
    def load_labels(self,):
        return self.label

def check_accuracy(context, batch_size, test_set, test_labels,output_size):
    print("check_accuracy")
    output_shape = [batch_size, output_size] 
    postprocess = Postprocess(output_shape)
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)

    num_corrent = 0
    num_total = 0
    batch_num = 0
    total_time = 0
    for start_idx in range(0, test_set.shape[0], batch_size):
        batch_num +=1
        if batch_num %10 ==0:
            print("Validating batch {:}".format(batch_num))
        end_idx = min(start_idx + batch_size, test_set.shape[0])
        effective_batch_size = end_idx - start_idx 

        inputs[0].host = test_set[start_idx : start_idx + effective_batch_size]
        t0 = time.time()
        output = common.do_inference(context, bindings= bindings, inputs = inputs, outputs = outputs, stream=stream, batch_size=effective_batch_size)
        total_time += time.time()-t0
        preds = postprocess(output)[0:effective_batch_size]
        labels = test_labels[start_idx:start_idx + effective_batch_size]
        num_total += effective_batch_size
        num_corrent +=np.count_nonzero(np.equal(preds,labels))
    
    percent_correct = 100 * num_corrent / float(num_total)
    print("Total Accuracy: {:}%".format(percent_correct))
    infer_time = total_time / test_set.shape[0]
    print("mean val of one batch infer time :{} s".format(infer_time))