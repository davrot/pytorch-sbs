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

#include "SpikeGeneration2DManyIP.h"

namespace py = pybind11;

PYBIND11_MODULE(PySpikeGeneration2DManyIP, m) {
  m.doc() = "SpikeGeneration2DManyIP Module";
  py::class_<SpikeGeneration2DManyIP>(m, "SpikeGeneration2DManyIP")
      .def(py::init<>())
      .def("spike_generation_multi_pattern",
           &SpikeGeneration2DManyIP::spike_generation_multi_pattern);
}
