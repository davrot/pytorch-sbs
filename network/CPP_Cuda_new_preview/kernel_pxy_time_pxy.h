#ifndef KERNEL_PXY_TIME_PXY
#define KERNEL_PXY_TIME_PXY

#include <vector>

__global__ void kernel_pxy_time_pxy(float* __restrict__ pxy_memory_a,
                                    float* __restrict__ pxy_memory_b,
                                    size_t max_idx);

void occupancy_kernel_pxy_time_pxy(size_t dim_x, size_t dim_y,
                                   size_t number_of_pattern, size_t h_dim,
                                   std::vector<size_t>& output,
                                   bool display_debug);

#endif /* KERNEL_PXY_TIME_PXY */
