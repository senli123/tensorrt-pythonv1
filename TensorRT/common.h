#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cuda_runtime.h>

#include "NvInfer.h"


template <typename A, typename B>
inline A divUp(A x, B n)
{
	return (x + n - 1) / n;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32: return 4;
	case nvinfer1::DataType::kFLOAT: return 4;
	case nvinfer1::DataType::kHALF: return 2;
	case nvinfer1::DataType::kBOOL:
	case nvinfer1::DataType::kINT8: return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}

inline void setAllTensorScales(nvinfer1::INetworkDefinition* network, float in_scales = 2.0f, float out_scales = 4.0f)
{
	for (int i = 0; i < network->getNbLayers(); i++)
	{
		auto layer = network->getLayer(i);
		for (int j = 0; j < layer->getNbInputs(); j++)
		{
			nvinfer1::ITensor* input{ layer->getInput(j) };
			if (input != nullptr && !input->dynamicRangeIsSet())
			{
				input->setDynamicRange(-in_scales, in_scales);
			}
		}
	}
	for (int i = 0; i < network->getNbLayers(); i++)
	{
		auto layer = network->getLayer(i);
		for (int j = 0; j < layer->getNbOutputs(); j++)
		{
			nvinfer1::ITensor* output{ layer->getOutput(j) };
			if (output != nullptr && !output->dynamicRangeIsSet())
			{
				if (layer->getType() == nvinfer1::LayerType::kPOOLING)
				{
					output->setDynamicRange(-in_scales, in_scales);
				}
				else
				{
					output->setDynamicRange(-out_scales, out_scales);
				}
			}
		}
	}
	return;
}

inline void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
	if (useDLACore >= 0)
	{
		if (builder->getNbDLACores() == 0)
		{
			std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
				<< std::endl;
			return;
		}
		if (allowGPUFallback)
		{
			config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
		}
		if (!builder->getInt8Mode() && !config->getFlag(nvinfer1::BuilderFlag::kINT8))
		{
			builder->setFp16Mode(true);
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
		config->setDLACore(useDLACore);
		config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
	}
	return;
}

class Logger : public nvinfer1::ILogger
{
public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg)override
	{
		if (severity != nvinfer1::ILogger::Severity::kINFO)
		{
			std::cout << msg << std::endl;
		}
	}
};

class BufferControl
{
public:
	void createBuffer(nvinfer1::ICudaEngine* engine, const int batch_size)
	{
		m_engine = engine;
		for (int i = 0; i < engine->getNbBindings(); ++i)
		{
			auto dims = m_engine->getBindingDimensions(i);
			/*size_t volum = static_cast<size_t>(batch_size);*/
			size_t volum = 1;
			nvinfer1::DataType data_type = m_engine->getBindingDataType(i);
			int vec_dim = m_engine->getBindingVectorizedDim(1);
			if (-1 != vec_dim)
			{
				int scalars_perVec = m_engine->getBindingComponentsPerElement(i);
				dims.d[vec_dim] = divUp(dims.d[vec_dim], scalars_perVec);
				volum *= scalars_perVec;
			}
			volum *= accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
			int element_size = getElementSize(data_type);
			void* host_buffer, *device_buffer;
			cudaMalloc(&device_buffer, volum * element_size);
			host_buffer = malloc(volum * element_size);
			m_device_buffer.emplace_back(device_buffer);
			m_host_buffer.emplace_back(host_buffer);
			m_buffer_size.push_back(volum * element_size);
		}
	}
	void* getHostBuffer(std::string inputName)
	{
		int index = m_engine->getBindingIndex(inputName.c_str());
		if (index == -1)
		{
			std::cout << "Input name is wrong!!" << std::endl;
			return nullptr;
		}
		return m_host_buffer[index];
	}

	void copyBuffers(const bool copy_input, const bool device_to_host)
	{
		for (int i = 0; i < m_engine->getNbBindings(); ++i)
		{
			void* dst_Ptr = device_to_host ? m_host_buffer[i] : m_device_buffer[i];
			void* src_Ptr = device_to_host ? m_device_buffer[i] : m_host_buffer[i];
			const size_t byte_size = m_buffer_size[i];
			const cudaMemcpyKind memcpy_type = device_to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
			if ((copy_input && m_engine->bindingIsInput(i)) || (!copy_input && !m_engine->bindingIsInput(i)))
			{
				cudaMemcpy(dst_Ptr, src_Ptr, byte_size, memcpy_type);
			}
		}
		return;
	}

	std::vector<void*>& getDeviceBuffer()
	{
		return m_device_buffer;
	}

	std::vector<void*>& getHostBuffer()
	{
		return m_host_buffer;
	}

	~BufferControl()
	{
		for (auto ptr : m_device_buffer)
		{
			cudaFree(ptr);
		}
		for (auto ptr : m_host_buffer)
		{
			free(ptr);
		}
	}
private:
	nvinfer1::ICudaEngine* m_engine;
	std::vector<void*> m_device_buffer;
	std::vector<void*> m_host_buffer;
	std::vector<size_t> m_buffer_size;
};