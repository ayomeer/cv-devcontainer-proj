#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

cv::Mat readImage(){
    cv::Mat image;
    image = cv::imread("/app/_img/dog.jpg");

    return image;
}

PYBIND11_MODULE(cppmodule, m){
    m.def("readImage", &readImage);
    m.doc() = "Docstring for readImage function";

    py::class_<cv::Mat>(m, "Mat", py::buffer_protocol())
        .def_buffer([](cv::Mat &im) -> py::buffer_info {
            return py::buffer_info(
                im.data,                                            // pointer to data
                sizeof(unsigned char),                              // item size
                py::format_descriptor<unsigned char>::format(),     // item descriptor
                3,                                            // matrix dimensionality
                {                                                   // buffer dimensions
                    im.rows, 
                    im.cols, 
                    im.channels()
                },          
                {                                                   // strides in bytes
                    sizeof(unsigned char) * im.channels() * im.cols, 
                    sizeof(unsigned char) * im.channels(),
                    sizeof(unsigned char)
                }
            );
        });
}


