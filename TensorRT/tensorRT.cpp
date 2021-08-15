#include "tensorRT.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <unistd.h>
TensorRT_Interface::TensorRT_Interface(const TensorRT_data &Tparams)
{
    this->Tparams = Tparams;
}

bool TensorRT_Interface::build()
{
    cudaSetDevice(this->Tparams.CudaID);
	nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(global_logger);
	if (!builder)
	{
		printf("Create builder fail !");
		return false;
	}
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
	if (!network)
	{
		printf("Create network fail !");
		builder->destroy();
		return false;
	}
	nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
	if (!config)
	{
		printf("Create config fail !");
		builder->destroy();
		network->destroy();
		return false;
	}
	builder->setMaxBatchSize(this->Tparams.BatchSize);
	builder->setMaxWorkspaceSize(1 << 30);
	if (this->Tparams.fp16)
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	if (this->Tparams.int8)
	{
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		setAllTensorScales(network, 127.0f, 127.0f);
	}
	// enableDLA(builder, config, 0);    // 利用DLA进行某些层的加速
	std::string binfile = this->Tparams.BinFileName;
	std::string onnxfile = this->Tparams.OnnxFileName;
	if (access(binfile.c_str(), 0) != 0)
	{
		nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, global_logger);
		if (!parser->parseFromFile(onnxfile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
		{
			std::cout << "Parse onnx file fail!" << std::endl;
			builder->destroy();
			network->destroy();
			config->destroy();
			return false;
		}
		engine = builder->buildEngineWithConfig(*network, *config);
		if (!engine)
		{
			std::cout << "Create engine fail!" << std::endl;
			builder->destroy();
			network->destroy();
			config->destroy();
			parser->destroy();
			return false;
		}
		nvinfer1::IHostMemory* trt_stream = engine->serialize();
		std::ofstream ofs(binfile, std::ios::out | std::ios::binary);
		ofs.write((char*)(trt_stream->data()), trt_stream->size());
		ofs.close();
		trt_stream->destroy();
		parser->destroy();
	}
	else
	{
		std::ifstream ifs(binfile, std::ios::in | std::ios::binary);
		std::stringstream temp_stream;
		temp_stream << ifs.rdbuf();
		ifs.close();
		temp_stream.seekg(0, std::ios::end);
		const int model_size = temp_stream.tellg();
		temp_stream.seekg(0, std::ios::beg);
		void* model_mem = malloc(model_size);
		temp_stream.read((char*)model_mem, model_size);
		nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(global_logger);
		if (!runtime)
		{
			std::cout << "Create runtime fail!" << std::endl;
			builder->destroy();
			network->destroy();
			config->destroy();
			return false;
		}
		this->engine = runtime->deserializeCudaEngine(model_mem, model_size, nullptr);
		if (!this->engine)
		{
			std::cout << "Create engine fail!" << std::endl;
			builder->destroy();
			network->destroy();
			config->destroy();
			runtime->destroy();
			return false;
		}
		runtime->destroy();
	}
	builder->destroy();
	config->destroy();
	network->destroy();
	this->buffer.createBuffer(this->engine, this->Tparams.BatchSize);
	this->hostDataBuffer = static_cast<float *>(this->buffer.getHostBuffer(this->Tparams.InputTensorNames));
	this->output = static_cast<float*>(this->buffer.getHostBuffer(this->Tparams.OutputTensorNames));
	return true;
}

bool TensorRT_Interface::infer()
{
	this->context = this->engine->createExecutionContext();
	if (!this->context)
	{
		printf("context fail !");
		return false;
	}
    cudaSetDevice(this->Tparams.CudaID);
    this->buffer.copyBuffers(true, false);
	clock_t a = clock();
    bool status = this->context->executeV2(this->buffer.getDeviceBuffer().data());
    if (!status)
    {
        printf("infer fail !");
        return false;
    }
	clock_t b = clock();
	double endtime=(double)(b-a)/CLOCKS_PER_SEC;
	std::cout<<"Total time:"<<endtime<<std::endl;		//s为单位
	std::cout<<"Total time:"<<endtime*1000<<"ms"<<std::endl;	//ms为单位
    this->buffer.copyBuffers(false, true);
	this->context->destroy();
    return true;
}

bool TensorRT_Interface::processInput(std::vector<cv::Mat> &Batch_rbg_img)
{
    try
    {
        memcpy(this->hostDataBuffer + this->img_size, Batch_rbg_img[0].data,
        this->Tparams.inputH *  this->Tparams.inputW *sizeof(float));
        memcpy(this->hostDataBuffer + this->Tparams.inputH * this->Tparams.inputW + this->img_size, Batch_rbg_img[1].data,
        this->Tparams.inputH *  this->Tparams.inputW *sizeof(float));
        memcpy(this->hostDataBuffer + this->Tparams.inputH * this->Tparams.inputW * 2 + this->img_size, Batch_rbg_img[2].data,
        this->Tparams.inputH *  this->Tparams.inputW *sizeof(float));
        this->img_size += this->Tparams.inputH * this->Tparams.inputW * 3;

    }catch (std::exception)
    {
        return false;
    }
    return true;
}
void TensorRT_Interface::img_size_clear()
{
    this->img_size = 0;
}