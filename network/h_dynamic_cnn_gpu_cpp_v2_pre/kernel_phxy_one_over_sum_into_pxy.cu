#include <cassert>
#include <iostream>

#include "kernel_helper_functions.h"
#include "kernel_phxy_one_over_sum_into_pxy.h"

__global__ void kernel_phxy_one_over_sum_into_pxy(
    float* __restrict__ phxy_memory, float* __restrict__ pxy_memory,
    size_t phxy_dim_c0, size_t phxy_dim_c1, size_t phxy_dim_c2, size_t h_dim,
    size_t pxy_dim_c0, size_t pxy_dim_c1, size_t block_dim_c0,
    size_t block_dim_c1, size_t max_idx) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < max_idx) {
    size_t pattern_id = idx / block_dim_c0;
    idx -= pattern_id * block_dim_c0;
    size_t position_x = idx / block_dim_c1;
    idx -= position_x * block_dim_c1;
    size_t position_y = idx;

    size_t offset_phxy_temp =
        pattern_id * phxy_dim_c0 + position_x * phxy_dim_c2 + position_y;

    size_t offset_pxy_sum =
        pattern_id * pxy_dim_c0 + position_x * pxy_dim_c1 + position_y;

    float temp = 0.0;
    for (size_t idx_h = 0; idx_h < h_dim; idx_h++) {
      temp += phxy_memory[offset_phxy_temp + idx_h * phxy_dim_c1];
    }
    if (temp > 1E-10) {
      pxy_memory[offset_pxy_sum] = 1.0 / temp;
    } else {
      pxy_memory[offset_pxy_sum] = 0.0;
    }
  }
};

void occupancy_kernel_phxy_one_over_sum_into_pxy(size_t dim_x, size_t dim_y,
                                                 size_t number_of_pattern,
                                                 size_t h_dim,
                                                 std::vector<size_t>& output,
                                                 bool display_debug) {
  size_t max_threadable_tasks;
  cudaError_t status;

  int min_grid_size;
  int thread_block_size;
  int grid_size;

  max_threadable_tasks = number_of_pattern * dim_x * dim_y;

  status = cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size,
      (void*)kernel_phxy_one_over_sum_into_pxy, 0, max_threadable_tasks);
  assert((status == cudaSuccess));

  grid_size =
      (max_threadable_tasks + thread_block_size - 1) / thread_block_size;

  output.resize(7);
  output[0] = grid_size;
  output[1] = 1;
  output[2] = 1;
  output[3] = thread_block_size;
  output[4] = 1;
  output[5] = 1;
  output[6] = max_threadable_tasks;

  if (display_debug == true) {
    std::cout << "kernel_phxy_one_over_sum_into_pxy:" << std::endl;
    kernel_debug_plot(output, display_debug);
  }
};