#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/reduce_sum.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
class Parameter {
public:
	Parameter(float data) : initial_data{data} {}

	virtual ~Parameter() {}

	T* get_parameter(bool inference) {
		T* params = inference ? m_param_inference : m_param;

		return params;
	}

	T* get_gradient(){
		return m_param_gradient;
	}

	T* inference_mixed_precision_impl(bool use_inference_params) {
		T* param = get_parameter(use_inference_params);

		return param;
	}

	T* forward_impl(bool use_inference_params) {
		T* param = get_parameter(use_inference_params);
		return param;
	}

	void backward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& dL_doutput,
		bool use_inference_params,
		EGradientMode param_gradients_mode
	) {
		T* param = get_gradient();

		if (param_gradients_mode != EGradientMode::Accumulate) {
			T zero = 0;
			CUDA_CHECK_THROW(cudaMemcpyAsync(param, &zero, sizeof(T), cudaMemcpyHostToDevice, stream));
		}
		

		reduce_sum(dL_doutput.data(), [] __device__ (T val) { return val; }, param, dL_doutput.n_elements(), stream);
	}

	void set_params(T* params, T* inference_params, T* gradients) {
		m_param = params;
		m_param_inference = inference_params;
		m_param_gradient = gradients;
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1.0f) {
		CUDA_CHECK_THROW(cudaMemcpy(params_full_precision, &initial_data, sizeof(float), cudaMemcpyHostToDevice));
	}

	size_t n_params() const {
		return align_to_cacheline(sizeof(T)) / (8 * sizeof(T));
	}

	json hyperparams() const {
		return {
			{"otype", "Parameter"},
			{"data", initial_data}
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMemoryArena::Allocation alloc;
	};

	float initial_data;
	T* m_param{nullptr};
	T* m_param_inference{nullptr};
	T* m_param_gradient{nullptr};
};

template <typename T>
Parameter<T>* create_parameter(const json& parameter);

TCNN_NAMESPACE_END
