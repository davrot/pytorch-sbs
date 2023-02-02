
#include <pybind11/pybind11.h>

#include "MultiplicationApproximationCPU.h"

namespace py = pybind11;

PYBIND11_MODULE(PyMultiplicationApproximationCPU, m) {
  m.doc() = "MultiplicationApproximationCPU Module";
  py::class_<MultiplicationApproximationCPU>(m, "MultiplicationApproximationCPU")
    .def(py::init<>())
    .def("update_entrypoint",
      &MultiplicationApproximationCPU::entrypoint);
}