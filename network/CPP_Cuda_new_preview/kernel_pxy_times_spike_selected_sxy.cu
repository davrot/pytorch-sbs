#include <cassert>
#include <iostream>

#include "kernel_helper_functions.h"
#include "kernel_pxy_times_spike_selected_sxy.h"

__global__ void kernel_pxy_times_spike_selected_sxy(
    float* __restrict__ pxy_memory, float* __restrict__ sxy_memory,
    int64_t* __restrict__ spike_memory, size_t spike_time, size_t spike_dim_c0,
    size_t spike_dim_c1, size_t spike_dim_c2, size_t pxy_dim_c0,
    size_t pxy_dim_c1, size_t sxy_dim_c0, size_t sxy_dim_c1,
    size_t block_dim_c0, size_t block_dim_c1, size_t max_idx) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < max_idx) {
    size_t pattern_id = idx / block_dim_c0;
    idx -= pattern_id * block_dim_c0;
    size_t position_x = idx / block_dim_c1;
    idx -= position_x * block_dim_c1;
    size_t position_y = idx;

    int64_t* spike = spike_memory + pattern_id * spike_dim_c0 +
                     spike_time * spike_dim_c1 + position_x * spike_dim_c2 +
                     position_y;

    if (*spike >= 0) {
      pxy_memory[pattern_id * pxy_dim_c0 + position_x * pxy_dim_c1 +
                 position_y] *=
          sxy_memory[*spike * sxy_dim_c0 + position_x * sxy_dim_c1 +
                     position_y];
    } else {
      pxy_memory[pattern_id * pxy_dim_c0 + position_x * pxy_dim_c1 +
                 position_y] = 0;
    }
  }
};

void occupancy_kernel_pxy_times_spike_selected_sxy(size_t dim_x, size_t dim_y,
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
      (void*)kernel_pxy_times_spike_selected_sxy, 0, max_threadable_tasks);
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
    std::cout << "kernel_pxy_times_spike_selected_sxy:" << std::endl;
    kernel_debug_plot(output, display_debug);
  }
};