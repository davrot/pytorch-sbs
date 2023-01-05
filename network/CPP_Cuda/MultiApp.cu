#include "MultiApp.h"

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "approximation_multiplication_function.cpp"
#include "gpu_approximation_multiplication_function.cu"

MultiApp::MultiApp()
{

};

MultiApp::~MultiApp()
{

};

bool MultiApp::update(float* np_input_pointer,
    float* np_weight_pointer,
    float* np_output_pointer, int64_t pattern_dim,
    int64_t feature_dim, int64_t x_dim, int64_t y_dim,
    int64_t input_channel_dim, int64_t id_pattern,
    bool approximation_enable, int64_t number_of_trunc_bits,
    int64_t number_of_frac_bits)
{

    assert((id_pattern >= 0));
    assert((id_pattern < pattern_dim));

    float* np_input_pointer_pattern;
    float* np_output_pointer_pattern;

    float* input_ptr;
    float* output_ptr;
    float* w_ptr;

    uint64_t pattern_size = input_channel_dim;

    std::vector<float> ap_h_vector;
    ap_h_vector.resize(pattern_size);
    float* ap_h_ptr = ap_h_vector.data();

    std::vector<uint32_t> ap_x_vector;
    ap_x_vector.resize(pattern_size);
    uint32_t* ap_x_ptr = ap_x_vector.data();

    std::vector<uint32_t> ap_y_vector;
    ap_y_vector.resize(pattern_size);
    uint32_t* ap_y_ptr = ap_y_vector.data();

    std::vector<uint32_t> ap_x_exponent_vector;
    ap_x_exponent_vector.resize(pattern_size);
    uint32_t* ap_x_exponent_ptr = ap_x_exponent_vector.data();

    std::vector<uint32_t> ap_y_exponent_vector;
    ap_y_exponent_vector.resize(pattern_size);
    uint32_t* ap_y_exponent_ptr = ap_y_exponent_vector.data();

    std::vector<uint32_t> ap_h_exponent_vector;
    ap_h_exponent_vector.resize(pattern_size);
    uint32_t* ap_h_exponent_ptr = ap_h_exponent_vector.data();

    std::vector<uint64_t> ap_res_vector;
    ap_res_vector.resize(pattern_size);
    uint64_t* ap_res_ptr = ap_res_vector.data();

    uint32_t ap_mask = static_cast<uint64_t>(pow(2, number_of_trunc_bits)) - 1;

    std::vector<uint32_t> sign_temp_vector;
    sign_temp_vector.resize(pattern_size);
    uint32_t* sign_temp_ptr = sign_temp_vector.data();

    uint64_t input_pattern_size = input_channel_dim * x_dim * y_dim;
    uint64_t output_pattern_size = feature_dim * x_dim * y_dim;

    np_input_pointer_pattern = np_input_pointer + id_pattern * input_pattern_size;
    np_output_pointer_pattern =
        np_output_pointer + id_pattern * output_pattern_size;

    uint64_t counter;

    uint64_t counter_x;
    uint64_t counter_y;
    uint64_t counter_feature;
    uint64_t pos_xy;
    uint64_t pos_xy_if;

    float temp_sum;

    uint64_t pattern_c_2 = x_dim * y_dim;

    for (counter_x = 0; counter_x < x_dim; counter_x++)
    {
        for (counter_y = 0; counter_y < y_dim; counter_y++)
        {
            pos_xy = counter_y + counter_x * y_dim;
            for (counter_feature = 0; counter_feature < feature_dim;
                counter_feature++)
            {
                pos_xy_if = counter_feature * pattern_c_2 + pos_xy;

                input_ptr = np_input_pointer_pattern + pos_xy;
                output_ptr = np_output_pointer_pattern + pos_xy_if;
                w_ptr = np_weight_pointer + counter_feature * input_channel_dim;

#pragma omp simd
                for (counter = 0; counter < pattern_size; counter++)
                {
                    ap_h_ptr[counter] = input_ptr[counter * pattern_c_2];
                }

                approximation_multiplication_function(
                    ap_h_ptr, w_ptr, pattern_size, number_of_trunc_bits,
                    number_of_frac_bits, ap_x_ptr, ap_y_ptr, ap_x_exponent_ptr,
                    ap_y_exponent_ptr, ap_h_exponent_ptr, ap_mask, ap_res_ptr,
                    sign_temp_ptr, approximation_enable);

                temp_sum = 0.0;
#pragma omp simd reduction(+ \
                           : temp_sum)
                for (counter = 0; counter < pattern_size; counter++)
                {
                    temp_sum += ap_h_ptr[counter];
                }

                output_ptr[0] = temp_sum;
            }
        }
    }

    return true;
};

bool MultiApp::update_with_init_vector_multi_pattern(
    int64_t np_input_pointer_addr, int64_t np_weight_pointer_addr,
    int64_t np_output_pointer_addr, int64_t pattern_dim, int64_t feature_dim,
    int64_t x_dim, int64_t y_dim, int64_t input_channel_dim,
    int64_t number_of_processes, bool approximation_enable,
    int64_t number_of_trunc_bits, int64_t number_of_frac)
{



    int64_t number_of_pattern = pattern_dim;
    int64_t pattern_id;

    float* np_input_pointer = (float*)np_input_pointer_addr;
    float* np_weight_pointer = (float*)np_weight_pointer_addr;
    float* np_output_pointer = (float*)np_output_pointer_addr;

    assert((np_input_pointer != nullptr));
    assert((np_output_pointer != nullptr));
    assert((np_weight_pointer != nullptr));

    assert((pattern_dim > 0));
    assert((feature_dim > 0));
    assert((x_dim > 0));
    assert((y_dim > 0));
    assert((input_channel_dim > 0));

    if (number_of_processes > 0)
    {
        omp_set_num_threads(number_of_processes);
        // For debugging: Only one thread
        // omp_set_num_threads(1);

#pragma omp parallel for
        for (pattern_id = 0; pattern_id < number_of_pattern; pattern_id++)
        {

            update(np_input_pointer, np_weight_pointer,
                np_output_pointer, pattern_dim, feature_dim, x_dim, y_dim,
                input_channel_dim, pattern_id, approximation_enable,
                number_of_trunc_bits, number_of_frac);
        }
    }
    else
    {
        update_gpu(np_input_pointer, np_weight_pointer,
            np_output_pointer, pattern_dim, feature_dim, x_dim, y_dim,
            input_channel_dim, approximation_enable,
            number_of_trunc_bits, number_of_frac);
    }
    return true;
};

__global__ void kernel_approx_multiplication(float* __restrict__ input_pointer, float* __restrict__ weight_pointer,
    float* __restrict__ output_pointer, uint64_t pattern_dim,
    uint64_t feature_dim, uint64_t x_dim, uint64_t y_dim,
    uint64_t input_channel_dim, size_t max_threadable_tasks,
    uint64_t input_index_scale, uint64_t number_of_frac_bits,
    bool approximation_enable, uint64_t number_of_trunc_bits,
    uint32_t ap_mask)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < max_threadable_tasks)
    {
        int pattern_id = idx / feature_dim;
        int feature_id = idx - (pattern_id * feature_dim);
        int x_id = blockIdx.y;
        int y_id = blockIdx.z;

        float* weight_pointer_sub = weight_pointer + feature_id * input_channel_dim;
        float* input_pointer_sub = input_pointer + pattern_id * input_channel_dim * x_dim * y_dim + x_id * y_dim + y_id;
        float* output_pointer_sub = output_pointer +
            pattern_id * feature_dim * x_dim * y_dim +
            feature_id * x_dim * y_dim + x_id * y_dim + y_id;
        *output_pointer_sub = 0.0;

        size_t counter;
        for (counter = 0; counter < input_channel_dim; counter++)
        {
            *output_pointer_sub += gpu_approximation_multiplication_function(
                weight_pointer_sub[counter],
                input_pointer_sub[counter * input_index_scale],
                number_of_frac_bits, approximation_enable,
                number_of_trunc_bits, ap_mask);
        }
    }
};

bool MultiApp::update_gpu(float* np_input_pointer,
    float* np_weight_pointer,
    float* np_output_pointer, uint64_t pattern_dim,
    uint64_t feature_dim, uint64_t x_dim, uint64_t y_dim,
    uint64_t input_channel_dim,
    bool approximation_enable, uint64_t number_of_trunc_bits,
    uint64_t number_of_frac_bits)
{

    uint32_t ap_mask = static_cast<uint64_t>(pow(2, number_of_trunc_bits)) - 1;
    // std::cout << approximation_enable << std::endl;
    // std::cout << number_of_trunc_bits << std::endl;
    // std::cout << number_of_frac_bits << std::endl;

    cudaError_t status;
    assert((x_dim < 65535));
    assert((y_dim < 65535));

    // //////////////////////////////////////
    // Get infos about the device
    // //////////////////////////////////////

    int device;
    cudaDeviceProp prop;

    status = cudaGetDevice(&device);
    assert((status == cudaSuccess));
    // std::cout << "Device ID: " << device << std::endl;

    status = cudaGetDeviceProperties(&prop, device);
    assert((status == cudaSuccess));
    // std::cout << "Device name: " << prop.name << std::endl;

    // //////////////////////////////////////
    // Calculate the distribution on the GPU
    // //////////////////////////////////////

    int min_grid_size;
    int block_size;
    int grid_size;

    size_t dynamic_s_mem_size = 0;
    size_t max_threadable_tasks = pattern_dim * feature_dim * x_dim * y_dim;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=blocksize#occupancy-calculator
    status = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
        (void*)kernel_approx_multiplication,
        dynamic_s_mem_size, max_threadable_tasks);
    assert((status == cudaSuccess));

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    // Maximum dimensionality of grid of thread blocks: 3
    // Maximum x -dimension of a grid of thread blocks: (2^31)-1
    // Maximum y- or z-dimension of a grid of thread blocks: 65535

    // Round up according to array size
    grid_size = ((pattern_dim * feature_dim) + block_size - 1) / block_size;

    // std::cout << min_grid_size << std::endl;
    // std::cout << grid_size << std::endl;
    // std::cout << block_size << std::endl;
    // std::cout << max_threadable_tasks << std::endl;

    dim3 grid(grid_size, x_dim, y_dim);

    kernel_approx_multiplication<<<grid, block_size>>>(np_input_pointer,
        np_weight_pointer,
        np_output_pointer,
        pattern_dim,
        feature_dim,
        x_dim,
        y_dim,
        input_channel_dim,
        (pattern_dim * feature_dim),
        (x_dim * y_dim),
        number_of_frac_bits,
        approximation_enable,
        number_of_trunc_bits,
        ap_mask);

    cudaDeviceSynchronize();
    return true;
};
