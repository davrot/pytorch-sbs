#include <cassert>
#include <iostream>

#include "kernel_helper_functions.h"
#include "kernel_pxy_time_pxy.h"

// a *= b
__global__ void kernel_pxy_time_pxy(float* __restrict__ pxy_memory_a,
                                    float* __restrict__ pxy_memory_b,
                                    size_t max_idx) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < max_idx) {
    pxy_memory_a[idx] *= pxy_memory_b[idx];
  }
};

void occupancy_kernel_pxy_time_pxy(size_t dim_x, size_t dim_y,
                                   size_t number_of_pattern, size_t h_dim,
                                   std::vector<size_t>& output,
                                   bool display_debug) {
  size_t max_threadable_tasks;
  cudaError_t status;

  int min_grid_size;
  int thread_block_size;
  int grid_size;

  max_threadable_tasks = number_of_pattern * dim_x * dim_y;

  status = cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, (void*)kernel_pxy_time_pxy, 0,
      max_threadable_tasks);
  assert((status == cudaSuccess));

  size_t gpu_tuning_factor = 5;

  if ((gpu_tuning_factor > 0) && (gpu_tuning_factor < thread_block_size)) {
    thread_block_size = int(gpu_tuning_factor);
  }

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
    std::cout << "kernel_pxy_time_pxy:" << std::endl;
    kernel_debug_plot(output, display_debug);
  }
};
