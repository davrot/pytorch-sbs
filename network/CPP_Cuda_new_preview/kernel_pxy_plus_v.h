#ifndef KERNEL_PXY_PLUS_V
#define KERNEL_PXY_PLUS_V
#include <vector>

__global__ void kernel_pxy_plus_v(float* __restrict__ pxy_memory, float value,
                                  size_t max_idx);

void occupancy_kernel_pxy_plus_v(size_t dim_x, size_t dim_y,
                                 size_t number_of_pattern, size_t h_dim,
                                 std::vector<size_t>& output,
                                 bool display_debug);
#endif /* KERNEL_PXY_PLUS_V */
