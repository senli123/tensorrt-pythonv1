import struct
import torch
import argparse
import os
import numpy as np
def generate_weights(opt):
    if not opt.weights:
        print("Please provide weights file")
        return 
    #Load model
    model = torch.load(opt.weights)
    model = model.cuda()
    model = model.eval()

    f = open(opt.save_trt_weights,"w")
    print("Keys:",model.state_dict().keys())
    f.write("{}\n".format(len(model.state_dict().keys())))
    for key, val in model.state_dict().items():
        print("Key:{}, Val:{}".format(key,val.shape))
        vval = val.reshape(-1).cpu().numpy()
        f.write("{} {}".format(key,len(vval)))
        for v in vval:
            f.write(" ")
            f.write(struct.pack(">f",float(v)).hex())
        f.write("\n")

def load_weights(file):
    print(f"Loading weights:{file}")
    assert os.path.exists(file),'Unable to load weight file.'

    weight_map = {}
    with open(file,"r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1,count+1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)
    return weight_map            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./model_zoo/pth/mobilenet_v2.pth', help='pytorch weights model.pth path(s)')
    parser.add_argument('--save-trt-weights', type=str, default='./model_zoo/wts/mobilenet_v2.wts', help='save path for tensorrt weights')
    opt = parser.parse_args()
    print(opt)
    generate_weights(opt)