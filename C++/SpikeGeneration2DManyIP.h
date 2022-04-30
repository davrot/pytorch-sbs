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

#ifndef SRC_SPIKEGENERATION2DMANYIP_H_
#define SRC_SPIKEGENERATION2DMANYIP_H_

#include <unistd.h>

#include <cctype>
#include <iostream>

class SpikeGeneration2DManyIP {
 public:
  SpikeGeneration2DManyIP();
  ~SpikeGeneration2DManyIP();

  bool spike_generation_multi_pattern(
      int64_t np_input_pointer_addr, int64_t np_input_dim_0,
      int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
      int64_t np_random_values_pointer_addr, int64_t np_random_values_dim_0,
      int64_t np_random_values_dim_1, int64_t np_random_values_dim_2,
      int64_t np_random_values_dim_3, int64_t np_output_pointer_addr,
      int64_t np_output_dim_0, int64_t np_output_dim_1, int64_t np_output_dim_2,
      int64_t np_output_dim_3, int64_t number_of_cpu_processes);

  bool spike_generation(
      int64_t np_input_pointer_addr, int64_t np_input_dim_0,
      int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
      int64_t np_random_values_pointer_addr, int64_t np_random_values_dim_0,
      int64_t np_random_values_dim_1, int64_t np_random_values_dim_2,
      int64_t np_random_values_dim_3, int64_t np_output_pointer_addr,
      int64_t np_output_dim_0, int64_t np_output_dim_1, int64_t np_output_dim_2,
      int64_t np_output_dim_3, int64_t id_pattern);

 private:
};

#endif /* SRC_SPIKEGENERATION2DMANYIP_H_ */