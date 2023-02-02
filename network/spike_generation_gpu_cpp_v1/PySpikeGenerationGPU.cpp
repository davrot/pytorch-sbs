
#include <pybind11/pybind11.h>

#include "SpikeGenerationGPU.h"

namespace py = pybind11;

PYBIND11_MODULE(PySpikeGenerationGPU, m)
{
  m.doc() = "SpikeGenerationGPU Module";
  py::class_<SpikeGenerationGPU>(m, "SpikeGenerationGPU")
    .def(py::init<>())
    .def("gpu_occupancy_export",
      &SpikeGenerationGPU::gpu_occupancy_export)
    .def("gpu_occupancy_import",
      &SpikeGenerationGPU::gpu_occupancy_import)
    .def("spike_generation",
      &SpikeGenerationGPU::entrypoint);
}
