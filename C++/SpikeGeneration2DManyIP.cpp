// MIT License
// Copyright 2022 University of Bremen
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
// THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//
// David Rotermund ( davrot@uni-bremen.de )
//
//
// Release history:
// ================
// 1.0.0 -- 01.05.2022: first release
//
//

#include "SpikeGeneration2DManyIP.h"

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

SpikeGeneration2DManyIP::SpikeGeneration2DManyIP(){

};

SpikeGeneration2DManyIP::~SpikeGeneration2DManyIP(){

};

bool SpikeGeneration2DManyIP::spike_generation_multi_pattern(
    int64_t np_input_pointer_addr, int64_t np_input_dim_0,
    int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
    int64_t np_random_values_pointer_addr, int64_t np_random_values_dim_0,
    int64_t np_random_values_dim_1, int64_t np_random_values_dim_2,
    int64_t np_random_values_dim_3, int64_t np_output_pointer_addr,
    int64_t np_output_dim_0, int64_t np_output_dim_1, int64_t np_output_dim_2,
    int64_t np_output_dim_3, int64_t number_of_cpu_processes) {
  int64_t number_of_pattern = np_input_dim_0;
  int64_t pattern_id;

  omp_set_num_threads(number_of_cpu_processes);

#pragma omp parallel for
  for (pattern_id = 0; pattern_id < number_of_pattern; pattern_id++) {
    spike_generation(
        np_input_pointer_addr, np_input_dim_0, np_input_dim_1, np_input_dim_2,
        np_input_dim_3, np_random_values_pointer_addr, np_random_values_dim_0,
        np_random_values_dim_1, np_random_values_dim_2, np_random_values_dim_3,
        np_output_pointer_addr, np_output_dim_0, np_output_dim_1,
        np_output_dim_2, np_output_dim_3, pattern_id);
  }

  return true;
};

bool SpikeGeneration2DManyIP::spike_generation(
    int64_t np_input_pointer_addr, int64_t np_input_dim_0,
    int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
    int64_t np_random_values_pointer_addr, int64_t np_random_values_dim_0,
    int64_t np_random_values_dim_1, int64_t np_random_values_dim_2,
    int64_t np_random_values_dim_3, int64_t np_output_pointer_addr,
    int64_t np_output_dim_0, int64_t np_output_dim_1, int64_t np_output_dim_2,
    int64_t np_output_dim_3, int64_t id_pattern) {
  float *np_input_pointer = (float *)np_input_pointer_addr;
  float *np_random_values_pointer = (float *)np_random_values_pointer_addr;
  int64_t *np_output_pointer = (int64_t *)np_output_pointer_addr;

  assert((id_pattern >= 0));
  assert((id_pattern < np_input_dim_0));

  // Input
  assert((np_input_pointer != nullptr));
  assert((np_input_dim_0 > 0));
  assert((np_input_dim_1 > 0));
  assert((np_input_dim_2 > 0));
  assert((np_input_dim_3 > 0));

  int64_t np_input_dim_c0 = np_input_dim_1 * np_input_dim_2 * np_input_dim_3;
  int64_t np_input_dim_c1 = np_input_dim_2 * np_input_dim_3;
  int64_t np_input_dim_c2 = np_input_dim_3;

  // Random
  assert((np_random_values_pointer != nullptr));
  assert((np_random_values_dim_0 > 0));
  assert((np_random_values_dim_1 > 0));
  assert((np_random_values_dim_2 > 0));
  assert((np_random_values_dim_3 > 0));

  int64_t np_random_values_dim_c0 =
      np_random_values_dim_1 * np_random_values_dim_2 * np_random_values_dim_3;
  int64_t np_random_values_dim_c1 =
      np_random_values_dim_2 * np_random_values_dim_3;
  int64_t np_random_values_dim_c2 = np_random_values_dim_3;

  // Output
  assert((np_output_pointer != nullptr));
  assert((np_output_dim_0 > 0));
  assert((np_output_dim_1 > 0));
  assert((np_output_dim_2 > 0));
  assert((np_output_dim_3 > 0));

  int64_t np_output_dim_c0 =
      np_output_dim_1 * np_output_dim_2 * np_output_dim_3;
  int64_t np_output_dim_c1 = np_output_dim_2 * np_output_dim_3;
  int64_t np_output_dim_c2 = np_output_dim_3;

  // -------------------------------

  int64_t h_dim = np_input_dim_1;
  int64_t spike_dim = np_output_dim_1;

  std::vector<float> temp_p;
  temp_p.resize(h_dim);
  float *temp_p_ptr = temp_p.data();

  std::vector<int64_t> temp_out;
  temp_out.resize(spike_dim);
  int64_t *temp_out_ptr = temp_out.data();

  std::vector<float> temp_rand;
  temp_rand.resize(spike_dim);
  float *temp_rand_ptr = temp_rand.data();

  int64_t counter;

  int64_t counter_x = 0;
  int64_t counter_y = 0;

  float *p_ptr = nullptr;
  int64_t *out_ptr = nullptr;
  float *rand_ptr = nullptr;

  std::vector<float>::iterator position_iterator;

  for (counter_x = 0; counter_x < np_output_dim_2; counter_x++) {
    for (counter_y = 0; counter_y < np_output_dim_3; counter_y++) {
      p_ptr = np_input_pointer + id_pattern * np_input_dim_c0 +
              counter_x * np_input_dim_c2 + counter_y;
      // + counter * np_input_dim_c1

      out_ptr = np_output_pointer + id_pattern * np_output_dim_c0 +
                counter_x * np_output_dim_c2 + counter_y;
      // + counter * np_output_dim_c1

      rand_ptr = np_random_values_pointer +
                 id_pattern * np_random_values_dim_c0 +
                 counter_x * np_random_values_dim_c2 + counter_y;
      // + counter * np_random_values_dim_c1

#pragma omp simd
      for (counter = 0; counter < h_dim; counter++) {
        temp_p_ptr[counter] = p_ptr[counter * np_input_dim_c1];
      }

#pragma omp simd
      for (counter = 0; counter < spike_dim; counter++) {
        temp_rand_ptr[counter] = rand_ptr[counter * np_random_values_dim_c1];
      }

      // ----------------------------
      for (counter = 0; counter < spike_dim; counter++) {
        position_iterator = std::lower_bound(temp_p.begin(), temp_p.end(),
                                             temp_rand_ptr[counter]);
        temp_out_ptr[counter] = position_iterator - temp_p.begin();
      }
      // ----------------------------

#pragma omp simd
      for (counter = 0; counter < spike_dim; counter++) {
        out_ptr[counter * np_output_dim_c1] = temp_out_ptr[counter];
      }
    }
  }

  return true;
};
