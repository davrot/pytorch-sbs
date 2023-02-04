#include <cassert>
#include <iostream>

#include "kernel_helper_functions.h"
#include "kernel_phxy_plus_phxy.h"

__global__ void kernel_phxy_plus_phxy(float* __restrict__ phxy_memory_a,
                                      float* __restrict__ phxy_memory_b,
                                      size_t max_idx) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < max_idx) {
    phxy_memory_a[idx] += phxy_memory_b[idx];
  }
};

void occupancy_kernel_phxy_plus_phxy(size_t dim_x, size_t dim_y,
                                     size_t number_of_pattern, size_t h_dim,
                                     std::vector<size_t>& output,
                                     bool display_debug) {
  size_t max_threadable_tasks;
  cudaError_t status;

  int min_grid_size;
  int thread_block_size;
  int grid_size;

  max_threadable_tasks = number_of_pattern * h_dim * dim_x * dim_y;

  status = cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, (void*)kernel_phxy_plus_phxy, 0,
      max_threadable_tasks);
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
    std::cout << "kernel_phxy_plus_phxy:" << std::endl;
    kernel_debug_plot(output, display_debug);
  }
};