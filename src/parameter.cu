#include <tiny-cuda-nn/parameter.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
Parameter<T>* create_parameter(const json& parameter) {
    std::string otype = parameter["otype"];
    if (equals_case_insensitive(otype, "Parameter")) {
        return new Parameter<T>{
            parameter["data"]
        };
    }

    throw std::runtime_error{std::string{"Invalid otype: "} + otype};
}

template Parameter<network_precision_t>* create_parameter(const json& parameter);

TCNN_NAMESPACE_END
