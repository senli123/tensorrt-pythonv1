import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self,training_data, cache_file, batch_size =64):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = training_data
        self.current_index = 0
        self.device_input = cuda.men_alloc(self.data[0].nbytes * self.batch_size)
    def get_batch_size(self):
        return self.batch_size
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None
        current_batch = int(self.current_index / self.batch_size)
        if current_batch %10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        batch = self.data[self.current_index: self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]
    
    def read_cailbration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    def write_cailbration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)