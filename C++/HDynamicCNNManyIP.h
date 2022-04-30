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

#ifndef SRC_HDYNAMICCNNMANYIP_H_
#define SRC_HDYNAMICCNNMANYIP_H_

#include <unistd.h>

#include <cctype>
#include <iostream>

class HDynamicCNNManyIP {
 public:
  HDynamicCNNManyIP();
  ~HDynamicCNNManyIP();

  bool update(int64_t np_h_pointer_addr, int64_t np_h_dim_0, int64_t np_h_dim_1,
              int64_t np_h_dim_2, int64_t np_h_dim_3,
              int64_t np_epsilon_xy_pointer_addr, int64_t np_epsilon_xy_dim_0,
              int64_t np_epsilon_xy_dim_1, int64_t np_epsilon_t_pointer_addr,
              int64_t np_epsilon_t_dim_0, int64_t np_weights_pointer_addr,
              int64_t np_weights_dim_0, int64_t np_weights_dim_1,
              int64_t np_input_pointer_addr, int64_t np_input_dim_0,
              int64_t np_input_dim_1, int64_t np_input_dim_2,
              int64_t np_input_dim_3, float *np_init_vector_pointer_ptr,
              int64_t np_init_vector_dim_0, int64_t id_pattern);

  bool update_with_init_vector_multi_pattern(
      int64_t np_h_pointer_addr, int64_t np_h_dim_0, int64_t np_h_dim_1,
      int64_t np_h_dim_2, int64_t np_h_dim_3,
      int64_t np_epsilon_xy_pointer_addr, int64_t np_epsilon_xy_dim_0,
      int64_t np_epsilon_xy_dim_1, int64_t np_epsilon_t_pointer_addr,
      int64_t np_epsilon_t_dim_0, int64_t np_weights_pointer_addr,
      int64_t np_weights_dim_0, int64_t np_weights_dim_1,
      int64_t np_input_pointer_addr, int64_t np_input_dim_0,
      int64_t np_input_dim_1, int64_t np_input_dim_2, int64_t np_input_dim_3,
      int64_t np_init_vector_pointer_addr, int64_t np_init_vector_dim_0,
      int64_t number_of_processes);

 private:
};

#endif /* SRC_HDYNAMICCNNMANYIP_H_ */