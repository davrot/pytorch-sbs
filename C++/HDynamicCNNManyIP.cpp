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

#include "HDynamicCNNManyIP.h"

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

HDynamicCNNManyIP::HDynamicCNNManyIP(){

};

HDynamicCNNManyIP::~HDynamicCNNManyIP(){

};

bool HDynamicCNNManyIP::update(
    int64_t np_h_pointer_addr, int64_t np_h_dim_0, int64_t np_h_dim_1,
    int64_t np_h_dim_2, int64_t np_h_dim_3, int64_t np_epsilon_xy_pointer_addr,
    int64_t np_epsilon_xy_dim_0, int64_t np_epsilon_xy_dim_1,
    int64_t np_epsilon_xy_dim_2, int64_t np_epsilon_t_pointer_addr,
    int64_t np_epsilon_t_dim_0, int64_t np_weights_pointer_addr,
    int64_t np_weights_dim_0, int64_t np_weights_dim_1,
    int64_t np_input_pointer_addr, int64_t np_input_dim_0,
    int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
    float *np_init_vector_pointer_ptr, int64_t np_init_vector_dim_0,
    int64_t id_pattern) {
  float *np_h_pointer = (float *)np_h_pointer_addr;
  float *np_epsilon_xy_pointer = (float *)np_epsilon_xy_pointer_addr;
  float *np_epsilon_t_pointer = (float *)np_epsilon_t_pointer_addr;
  float *np_weights_pointer = (float *)np_weights_pointer_addr;
  int64_t *np_input_pointer = (int64_t *)np_input_pointer_addr;

  int64_t number_of_pattern = np_input_dim_0;

  assert((id_pattern >= 0));
  assert((id_pattern < number_of_pattern));

  assert((np_h_pointer != nullptr));
  assert((np_h_dim_0 > 0));
  assert((np_h_dim_1 > 0));
  assert((np_h_dim_2 > 0));
  assert((np_h_dim_3 > 0));

  int64_t np_h_dim_c0 = np_h_dim_1 * np_h_dim_2 * np_h_dim_3;
  int64_t np_h_dim_c1 = np_h_dim_2 * np_h_dim_3;
  int64_t np_h_dim_c2 = np_h_dim_3;

  float *np_h_pointer_pattern;
  float *np_h_pointer_pattern_0;
  float *np_h_pointer_pattern_01;

  assert((np_epsilon_xy_pointer != nullptr));
  assert((np_epsilon_xy_dim_0 > 0));
  assert((np_epsilon_xy_dim_1 > 0));

  int64_t np_epsilon_xy_dim_c0 = np_epsilon_xy_dim_2 * np_epsilon_xy_dim_1;
  int64_t np_epsilon_xy_dim_c1 = np_epsilon_xy_dim_2;

  float *np_epsilon_xy_pointer_0;
  float *np_epsilon_xy_pointer_01;

  assert((np_epsilon_t_pointer != nullptr));
  assert((np_epsilon_t_dim_0 > 0));

  assert((np_weights_pointer != nullptr));
  assert((np_weights_dim_0 > 0));
  assert((np_weights_dim_1 > 0));

  int64_t np_weights_dim_c0 = np_weights_dim_1;

  float *w_ptr;

  assert((np_input_pointer != nullptr));
  assert((np_input_dim_0 > 0));
  assert((np_input_dim_1 > 0));
  assert((np_input_dim_2 > 0));
  assert((np_input_dim_3 > 0));

  int64_t np_input_dim_c0 = np_input_dim_1 * np_input_dim_2 * np_input_dim_3;
  int64_t np_input_dim_c1 = np_input_dim_2 * np_input_dim_3;
  int64_t np_input_dim_c2 = np_input_dim_3;

  int64_t *np_input_pointer_pattern;
  int64_t *np_input_pointer_pattern_0;
  int64_t *np_input_pointer_pattern_01;
  int64_t *np_input_pointer_pattern_01_spike;

  assert((np_init_vector_pointer_ptr != nullptr));
  assert((np_init_vector_dim_0 == np_weights_dim_1));

  int64_t number_of_spikes = np_input_dim_1;

  int64_t h_dim = np_weights_dim_1;

  std::vector<float> h_temp_vector;
  h_temp_vector.resize(h_dim);
  float *h_temp = h_temp_vector.data();

  std::vector<float> h_subsegment_vector;
  h_subsegment_vector.resize(h_dim);
  float *h_subsegment = h_subsegment_vector.data();

  float h_temp_sum;

  int64_t id_0;
  int64_t id_1;
  int64_t id_spike;
  int64_t counter;

  float temp_value;

  float epsilon_scale;
  float epsilon_subsegment;

  // epsilon_subsegment = np_epsilon_xy_pointer[
  //                         id_0    * np_epsilon_xy_dim_c0 +
  //                         id_1 ]
  //                 * np_epsilon_t_pointer[id_spike];

  // spike = np_input_pointer[
  //                         id_pattern  * np_input_dim_c0 +
  //                         id_spike    * np_input_dim_c1 +
  //                         id_0     * np_input_dim_c2 +
  //                         id_1];

  // w_ptr = np_weights_pointer +
  //                         spike *     np_weights_dim_c0;

  // h_ptr = np_h_pointer +
  //         id_pattern  * np_h_dim_c0 +
  //         id_0        * np_h_dim_c2 +
  //         id_1;
  //         // 0 * np_h_dim_c1 +

  np_input_pointer_pattern = np_input_pointer + id_pattern * np_input_dim_c0;
  np_h_pointer_pattern = np_h_pointer + id_pattern * np_h_dim_c0;

  for (id_0 = 0; id_0 < np_input_dim_2; id_0++) {
    np_epsilon_xy_pointer_0 =
        np_epsilon_xy_pointer + id_0 * np_epsilon_xy_dim_c1;

    np_h_pointer_pattern_0 = np_h_pointer_pattern + id_0 * np_h_dim_c2;

    np_input_pointer_pattern_0 =
        np_input_pointer_pattern + id_0 * np_input_dim_c2;

    for (id_1 = 0; id_1 < np_input_dim_3; id_1++) {
      np_epsilon_xy_pointer_01 = np_epsilon_xy_pointer_0 + id_1;

      np_h_pointer_pattern_01 = np_h_pointer_pattern_0 + id_1;

      np_input_pointer_pattern_01 = np_input_pointer_pattern_0 + id_1;

      memcpy(h_subsegment, np_init_vector_pointer_ptr, sizeof(float) * h_dim);

      epsilon_scale = 1.0;

      for (id_spike = 0; id_spike < number_of_spikes; id_spike++) {
        if (epsilon_scale > 1E10) {
          temp_value = 1.0 / epsilon_scale;

#pragma omp simd
          for (counter = 0; counter < h_dim; counter++) {
            h_subsegment[counter] *= temp_value;
          }

          epsilon_scale = 1.0;
        }

        np_input_pointer_pattern_01_spike =
            np_input_pointer_pattern_01 + id_spike * np_input_dim_c1;

        epsilon_subsegment =
            np_epsilon_xy_pointer_01[np_input_pointer_pattern_01_spike[0] *
                                     np_epsilon_xy_dim_c0] *
            np_epsilon_t_pointer[id_spike];

        w_ptr = np_weights_pointer +
                np_input_pointer_pattern_01_spike[0] * np_weights_dim_c0;

        memcpy(h_temp, h_subsegment, sizeof(float) * h_dim);

#pragma omp simd
        for (counter = 0; counter < h_dim; counter++) {
          h_temp[counter] *= w_ptr[counter];
        }

        h_temp_sum = 0.0;
#pragma omp simd reduction(+ : h_temp_sum)
        for (counter = 0; counter < h_dim; counter++) {
          h_temp_sum += h_temp[counter];
        }

        if (h_temp_sum > 1E-10) {
          temp_value = epsilon_scale * epsilon_subsegment / h_temp_sum;

#pragma omp simd
          for (counter = 0; counter < h_dim; counter++) {
            h_temp[counter] *= temp_value;
          }

#pragma omp simd
          for (counter = 0; counter < h_dim; counter++) {
            h_subsegment[counter] += h_temp[counter];
          }

          epsilon_scale *= 1.0 + epsilon_subsegment;
          // IF
        }
        // spike End
      }

      temp_value = 1.0 / epsilon_scale;
#pragma omp simd
      for (counter = 0; counter < h_dim; counter++) {
        np_h_pointer_pattern_01[counter * np_h_dim_c1] =
            h_subsegment[counter] * temp_value;
      }

      // id_1 End
    }

    // id_0 End
  }

  return true;
};

bool HDynamicCNNManyIP::update_with_init_vector_multi_pattern(
    int64_t np_h_pointer_addr, int64_t np_h_dim_0, int64_t np_h_dim_1,
    int64_t np_h_dim_2, int64_t np_h_dim_3, int64_t np_epsilon_xy_pointer_addr,
    int64_t np_epsilon_xy_dim_0, int64_t np_epsilon_xy_dim_1,
    int64_t np_epsilon_xy_dim_2, int64_t np_epsilon_t_pointer_addr,
    int64_t np_epsilon_t_dim_0, int64_t np_weights_pointer_addr,
    int64_t np_weights_dim_0, int64_t np_weights_dim_1,
    int64_t np_input_pointer_addr, int64_t np_input_dim_0,
    int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
    int64_t np_init_vector_pointer_addr, int64_t np_init_vector_dim_0,
    int64_t number_of_processes) {
  int64_t number_of_pattern = np_input_dim_0;
  int64_t pattern_id;

  int64_t h_dim = np_init_vector_dim_0;
  float *h_init_ptr = (float *)np_init_vector_pointer_addr;

  omp_set_num_threads(number_of_processes);

#pragma omp parallel for
  for (pattern_id = 0; pattern_id < number_of_pattern; pattern_id++) {
    update(np_h_pointer_addr, np_h_dim_0, np_h_dim_1, np_h_dim_2, np_h_dim_3,
           np_epsilon_xy_pointer_addr, np_epsilon_xy_dim_0, np_epsilon_xy_dim_1,
           np_epsilon_xy_dim_2, np_epsilon_t_pointer_addr, np_epsilon_t_dim_0,
           np_weights_pointer_addr, np_weights_dim_0, np_weights_dim_1,
           np_input_pointer_addr, np_input_dim_0, np_input_dim_1,
           np_input_dim_2, np_input_dim_3, h_init_ptr, h_dim, pattern_id);
  }

  return true;
};
