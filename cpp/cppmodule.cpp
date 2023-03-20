#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

cv::Mat pointwiseUndistort(py::array_t<unsigned char>& imBuf){
    auto rows = imBuf.shape(0);
    auto cols = imBuf.shape(1);
    auto type = CV_8UC3; // 3 dim unsigned char
    auto ptr = (unsigned char*)imBuf.data();

    cv::Mat img(rows, cols, type, ptr);

    cv::flip(img, img, 0);
    return img;
}

PYBIND11_MODULE(cppmodule, m){
    m.def("pointwiseUndistort", &pointwiseUndistort, py::return_value_policy::automatic);
    m.doc() = "Docstring for pointwiseUndistort function";
    py::class_<cv::Mat>(m, "Mat", py::buffer_protocol())
        .def_buffer([](cv::Mat &im) -> py::buffer_info {
            return py::buffer_info(
                im.data,                                            // pointer to data
                sizeof(unsigned char),                              // item size
                py::format_descriptor<unsigned char>::format(),     // item descriptor
                3,                                                  // matrix dimensionality
                {                                                   // buffer dimensions
                    im.rows, 
                    im.cols, 
                    im.channels()
                },          
                {                                                   // strides in bytes
                    sizeof(unsigned char) * im.channels() * im.cols, // (issue with padding)
                    sizeof(unsigned char) * im.channels(),
                    sizeof(unsigned char)
                }
            );
        });
}


