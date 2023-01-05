
#include <pybind11/pybind11.h>

#include "MultiApp.h"

namespace py = pybind11;

PYBIND11_MODULE(PyMultiApp, m) {
  m.doc() = "MultiApp Module";
  py::class_<MultiApp>(m, "MultiApp")
      .def(py::init<>())
      .def("update_with_init_vector_multi_pattern",
           &MultiApp::update_with_init_vector_multi_pattern);
}