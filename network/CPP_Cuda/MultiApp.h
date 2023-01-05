#ifndef SRC_MultiApp_H_
#define SRC_MultiApp_H_

#include <unistd.h>

#include <cctype>
#include <iostream>

class MultiApp
{
public:
    MultiApp();
    ~MultiApp();

    bool update(float *np_input_pointer, float *np_weight_pointer,
                float *np_output_pointer, int64_t pattern_dim,
                int64_t feature_dim, int64_t x_dim, int64_t y_dim,
                int64_t input_channel_dim, int64_t id_pattern,
                bool approximation_enable, int64_t number_of_trunc_bits,
                int64_t number_of_frac);

    bool update_gpu(float *input_pointer, float *weight_pointer,
                    float *output_pointer, uint64_t pattern_dim,
                    uint64_t feature_dim, uint64_t x_dim, uint64_t y_dim,
                    uint64_t input_channel_dim,
                    bool approximation_enable, uint64_t number_of_trunc_bits,
                    uint64_t number_of_frac);

    bool update_with_init_vector_multi_pattern(
        int64_t np_input_pointer_addr, int64_t np_weight_pointer_addr,
        int64_t np_output_pointer_addr, int64_t pattern_dim, int64_t feature_dim,
        int64_t x_dim, int64_t y_dim, int64_t input_channel_dim,
        int64_t number_of_processes, bool approximation_enable,
        int64_t number_of_trunc_bits, int64_t number_of_frac);

private:
};

#endif /* SRC_MultiApp_H_ */