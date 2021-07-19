
#----------------------CUDA----------------------------
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}") 

#---------------------TensorRT----------------------------

set(TENSORRT_ROOT /home/lisen/TensorRT-7.2.2.3/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_MYELIN myelin
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_NVPARSERS    nvparsers
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY    ${TENSORRT_LIBRARY_INFER} 
                        ${TENSORRT_LIBRARY_MYELIN} 
                        ${TENSORRT_LIBRARY_INFER_PLUGIN}
                        ${TENSORRT_LIBRARY_NVPARSERS}
                        ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

#---------------------OpenCV----------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")
#--------------------加入头文件--------------------------------
#底层接口相关
include_directories(${PROJECT_SOURCE_DIR}/TensorRT)
#utils相关
include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/model/detection)
#模型相关
include_directories(${PROJECT_SOURCE_DIR}/model/detection/yolov5)
# include_directories(${PROJECT_SOURCE_DIR}/model/include/segmentation)
# include_directories(${PROJECT_SOURCE_DIR}/model/include/segmentation/Unet)
#include_directories(${PROJECT_SOURCE_DIR}/model/include/classification/LJclassify)
#service 相关
include_directories(${PROJECT_SOURCE_DIR}/service/yolov5)
#--------------------第三方头文件-----------------------------
include_directories(${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#--------------------加入源文件---------------------------------
aux_source_directory(. SRC_LIST)
aux_source_directory(${PROJECT_SOURCE_DIR}/TensorRT DIR_TensorRT_SRCS)
aux_source_directory(${PROJECT_SOURCE_DIR}/model/detection DIR_DETE_SRCS)
aux_source_directory(${PROJECT_SOURCE_DIR}/model/detection/yolov5 DIR_YOLOV5_SRCS)
#aux_source_directory(${PROJECT_SOURCE_DIR}/model/src/classification/LJclassify DIR_LJ_SRCS)
# aux_source_directory(${PROJECT_SOURCE_DIR}//model/src/segmentation DIR_SEG_SRCS)
# aux_source_directory(${PROJECT_SOURCE_DIR}//model/src/segmentation/Unet DIR_UNET_SRCS)
aux_source_directory(${PROJECT_SOURCE_DIR}/service/yolov5 DIR_SERVICE_SRCS)
add_executable(demo ${SRC_LIST} ${DIR_SERVICE_SRCS} ${DIR_TensorRT_SRCS} ${DIR_DETE_SRCS}  ${DIR_YOLOV5_SRCS}  ${DIR_LJ_SRCS}
)
#-------------------链接库--------------------------------------
target_link_libraries(demo ${CUDA_LIBRARIES})
target_link_libraries(demo /usr/local/cuda/lib64/libnvrtc.so.10.2)
target_link_libraries(demo ${OpenCV_LIBRARIES}  ${TENSORRT_LIBRARY} )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
