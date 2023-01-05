#ifndef SRC_SPIKEGENERATION2DMANYIP_H_
#define SRC_SPIKEGENERATION2DMANYIP_H_

#include <unistd.h>

#include <cctype>
#include <iostream>

class SpikeGeneration2DManyIP
{
    public:
    SpikeGeneration2DManyIP();
    ~SpikeGeneration2DManyIP();

    bool spike_generation_entrypoint(
        int64_t input_pointer_addr, int64_t input_dim_0,
        int64_t input_dim_1, int64_t input_dim_2, int64_t input_dim_3,
        int64_t random_values_pointer_addr, int64_t random_values_dim_0,
        int64_t random_values_dim_1, int64_t random_values_dim_2,
        int64_t random_values_dim_3, int64_t output_pointer_addr,
        int64_t output_dim_0, int64_t output_dim_1, int64_t output_dim_2,
        int64_t output_dim_3, int64_t number_of_cpu_processes);

    bool spike_generation(
        float* input_pointer,
        size_t input_dim_c0,
        size_t input_dim_c1,
        size_t input_dim_c2,
        float* random_values_pointer,
        size_t random_values_dim_c0,
        size_t random_values_dim_c1,
        size_t random_values_dim_c2,
        int64_t* output_pointer,
        size_t output_dim_c0,
        size_t output_dim_c1,
        size_t output_dim_c2,
        size_t x_dim,
        size_t y_dim,
        size_t spike_dim,
        size_t h_dim,
        size_t pattern_id);

    bool gpu_spike_generation(
        float* input_pointer,
        size_t input_dim_c0,
        size_t input_dim_c1,
        size_t input_dim_c2,
        float* random_values_pointer,
        size_t random_values_dim_c0,
        size_t random_values_dim_c1,
        size_t random_values_dim_c2,
        int64_t* output_pointer,
        size_t output_dim_c0,
        size_t output_dim_c1,
        size_t output_dim_c2,
        size_t x_dim,
        size_t y_dim,
        size_t spike_dim,
        size_t h_dim,
        size_t number_of_pattern);

    private:
    size_t lower_bound(float* data_ptr, size_t data_length,
        size_t data_ptr_stride,
        float compare_to_value);
};

#endif /* SRC_SPIKEGENERATION2DMANYIP_H_ */