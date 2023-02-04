#ifndef HDYNAMICCNNGPU
#define HDYNAMICCNNGPU

#include <unistd.h>

#include <cctype>
#include <iostream>

class HDynamicCNNGPU
{
public:
    HDynamicCNNGPU();
    ~HDynamicCNNGPU();

    void entrypoint(
        int64_t h_pointer_addr,
        int64_t h_dim_0,
        int64_t h_dim_1,
        int64_t h_dim_2,
        int64_t h_dim_3,
        int64_t epsilon_xy_pointer_addr,
        int64_t epsilon_xy_dim_0,
        int64_t epsilon_xy_dim_1,
        int64_t epsilon_xy_dim_2,
        int64_t epsilon_t_pointer_addr,
        int64_t epsilon_t_dim_0,
        int64_t weights_pointer_addr,
        int64_t weights_dim_0,
        int64_t weights_dim_1,
        int64_t input_pointer_addr,
        int64_t input_dim_0,
        int64_t input_dim_1,
        int64_t input_dim_2,
        int64_t input_dim_3,
        int64_t init_vector_pointer_addr,
        int64_t init_vector_dim_0,
        int64_t number_of_processes,
        float forgetting_offset,
        int64_t gpu_tuning_factor);

    void gpu_occupancy_export(
        size_t dim_x,
        size_t dim_y,
        size_t number_of_pattern,
        size_t h_dim,
        int64_t setting_memory_addr,
        size_t setting_dim_0,
        size_t setting_dim_1);

    void gpu_occupancy_import(
        int64_t setting_memory_addr,
        size_t setting_dim_0,
        size_t setting_dim_1);

private:

    void gpu_update(
        float* h_init_ptr,
        float* h_pointer,
        size_t h_dim_c0,
        size_t h_dim_c1,
        size_t h_dim_c2,
        size_t h_dim,
        float* epsilon_xy_pointer,
        size_t epsilon_xy_dim_c0,
        size_t epsilon_xy_dim_c1,
        float* epsilon_t_pointer,
        float* weights_pointer,
        size_t weights_dim_c0,
        int64_t* input_pointer,
        size_t input_dim_c0,
        size_t input_dim_c1,
        size_t input_dim_c2,
        size_t number_of_spikes,
        size_t dim_x,
        size_t dim_y,
        float forgetting_offset,
        float forgetting_offset_local,
        size_t number_of_pattern,
        size_t gpu_tuning_factor);

};

#endif /* HDYNAMICCNNGPU */
