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

#include <pybind11/pybind11.h>

#include "HDynamicCNNManyIP.h"

namespace py = pybind11;

PYBIND11_MODULE(PyHDynamicCNNManyIP, m) {
  m.doc() = "HDynamicCNNManyIP Module";
  py::class_<HDynamicCNNManyIP>(m, "HDynamicCNNManyIP")
      .def(py::init<>())
      .def("update_with_init_vector_multi_pattern",
           &HDynamicCNNManyIP::update_with_init_vector_multi_pattern);
}