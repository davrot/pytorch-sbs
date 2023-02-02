
#include <pybind11/pybind11.h>

#include "SpikeGenerationCPU.h"

namespace py = pybind11;

PYBIND11_MODULE(PySpikeGenerationCPU, m) {
  m.doc() = "SpikeGenerationCPU Module";
  py::class_<SpikeGenerationCPU>(m, "SpikeGenerationCPU")
    .def(py::init<>())
    .def("spike_generation",
      &SpikeGenerationCPU::entrypoint);
}
