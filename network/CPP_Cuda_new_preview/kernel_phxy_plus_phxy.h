#ifndef KERNEL_PHXY_PLUS_PHXY
#define KERNEL_PHXY_PLUS_PHXY
#include <vector>
__global__ void kernel_phxy_plus_phxy(float* __restrict__ phxy_memory_a,
                                      float* __restrict__ phxy_memory_b,
                                      size_t max_idx);

void occupancy_kernel_phxy_plus_phxy(size_t dim_x, size_t dim_y,
                                     size_t number_of_pattern, size_t h_dim,
                                     std::vector<size_t>& output,
                                     bool display_debug);
#endif /* KERNEL_PHXY_PLUS_PHXY */
