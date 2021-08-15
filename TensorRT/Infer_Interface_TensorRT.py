from __future__ import print_function
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import TensorRT.common as common
class Infer_TensorRT():
    def __init__(self,trt_path, gpuID):
        self.cfx = cuda.Device(gpuID).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)
        with open(trt_path, 'rb') as f:
            buf = f.read()
            self.engine = runtime.deserialize_cuda_engine(buf)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def infer(self,input):
        self.cfx.push()
        self.inputs[0].host = input
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, 
                        inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        self.destory()
        return trt_outputs 
    def destory(self):
        self.cfx.pop()

    