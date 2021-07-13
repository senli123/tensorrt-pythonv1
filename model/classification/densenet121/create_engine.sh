#!/bin/bash
curPath=$(readlink -f "$(dirname "$0")")
curFile=${curPath}/create_engine.py
#echo $curFile
/opt/conda/bin/python $curFile  --batch_size 1 \
--height 224 \
--width 224 \
--output_size 1000 \
--input_name data  \
--output_name prob \
--weight_path /workspace/lisen/_bushu/tensorrt-python/densenet121.wts \
--engine_path ./densenet121.engine 