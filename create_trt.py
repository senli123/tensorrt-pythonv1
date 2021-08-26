import tensorrt as trt
import os
import TensorRT.common as common
import argparse
TRT_LOGGER = trt.Logger()
def build_trx(model_path,OnnxFileName,batch_size,shape,TrtFileName,save_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network,TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = batch_size
            # Parse model file
            onnx_file_path = os.path.join(model_path,OnnxFileName)
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please first to generate it.'.format(OnnxFileName))
                exit(0)
            print('Loading ONNX file from path {}...'.format(OnnxFileName))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            network.get_input(0).shape = shape
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(OnnxFileName))
            engine = builder.build_cuda_engine(network)
            print('Completed creating Engine')
            engine_file_path = os.path.join(save_path,TrtFileName)
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_path", type=str, default="./model_zoo/onnx")
    parser.add_argument("--save_path", type=str, default="./model_zoo/trt")
    parser.add_argument("--onnx_name", type=str, default="squeezenet1_0.onnx")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shape', type=list, default=[1,3, 224,224])
    args = parser.parse_args()
    net_path = args.net_path
    onnx_name = args.onnx_name
    save_path = args.save_path
    onnx_path = os.path.join(net_path, onnx_name)
    if not os.path.exists(onnx_path):
        print("there is no model %s" %onnx_name)
    else:
        #trt_path = os.path.join(net_path, onnx_name.split('.')[0]+'.trt')
        trt_name = onnx_name.split('.')[0]+'.trt'
        # OnnxFileName = 'add_flj_no_pifeng_10cls_mix.onnx'
        # TrtFileName = 'add_flj_no_pifeng_10cls_mix.trt'
        batch_size = args.batch_size
        shape = args.shape
        build_trx(net_path,onnx_name,batch_size,shape,trt_name,save_path)
                    