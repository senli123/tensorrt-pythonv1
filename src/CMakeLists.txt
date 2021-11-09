
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

#--------------------glog---------------------------------
set(glog_INCLUDE_DIR "/workspace/lisen/_bushu/lisen/tools/glog/build")
set(glog_LIBRARIES "/workspace/lisen/_bushu/lisen/tools/glog/build/libglog.so")

#---------------加入所有源文件-------------------------------
file(GLOB_RECURSE SRC_MOPDEL_FILES
    ${CMAKE_SOURCE_DIR}/model/*.cpp
)
file(GLOB_RECURSE SRC_INTERFACE_FILES
    ${CMAKE_SOURCE_DIR}/TensorRT/*.cpp
)
file(GLOB_RECURSE SRC_SERVICE_FILES
    ${CMAKE_SOURCE_DIR}/service/*.cpp
)
file(GLOB_RECURSE SRC_UTILS_FILES
    ${CMAKE_SOURCE_DIR}/utils/*.cpp
)
add_library(TensorrtEngine STATIC ${SRC_INTERFACE_FILES} ${SRC_MOPDEL_FILES} ${SRC_SERVICE_FILES} ${SRC_UTILS_FILES})   #添加src路径下的所有cpp
#-------------------链接库--------------------------------------
target_link_libraries(TensorrtEngine 
                        PUBLIC ${CUDA_LIBRARIES} 
                        /usr/local/cuda/lib64/libnvrtc.so.10.2
                        ${OpenCV_LIBRARIES}  
                        ${TENSORRT_LIBRARY}
                        ${glog_LIBRARIES})
#--------------------加入头文件--------------------------------
target_include_directories(TensorrtEngine 
                        PUBLIC  ${TENSORRT_INCLUDE_DIR} 
                                ${OpenCV_INCLUDE_DIRS} 
                                ${CUDA_INCLUDE_DIRS}
                                ${glog_INCLUDE_DIR}
                                ${PROJECT_SOURCE_DIR}/TensorRT #底层接口相关
                                ${PROJECT_SOURCE_DIR}/utils #utils相关
                                ${PROJECT_SOURCE_DIR}/service
                                ${PROJECT_SOURCE_DIR}/model
                                ${PROJECT_SOURCE_DIR}/model/detection
                                ${PROJECT_SOURCE_DIR}/model/segmentation
                                ${PROJECT_SOURCE_DIR}/model/classification
                                #${PROJECT_SOURCE_DIR}/model/classification/mobilenet/mobilenetv2
                                )

# if(MIRROR_BUILD_CLASSIFIER)
#         add_executable(demo ${CMAKE_SOURCE_DIR}/src/test_classification_debug.cpp)
#         target_link_libraries(demo PUBLIC TensorrtEngine)
# endif()
if(MIRROR_BUILD_CLASSIFIER)
        add_executable(classifier ${CMAKE_SOURCE_DIR}/src/test_classification_debug.cpp)
        target_link_libraries(classifier PUBLIC TensorrtEngine)
endif()
if(MIRROR_BUILD_DETECTION)
        add_executable(detection ${CMAKE_SOURCE_DIR}/src/test_detection_debug.cpp)
        target_link_libraries(detection PUBLIC TensorrtEngine)
endif()
if(MIRROR_BUILD_SEGMENTATION)
        add_executable(segmentation ${CMAKE_SOURCE_DIR}/src/test_segmentation_debug.cpp)
        target_link_libraries(segmentation PUBLIC TensorrtEngine)
endif()
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
