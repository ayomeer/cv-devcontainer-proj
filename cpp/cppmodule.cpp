#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/*
void pointwiseUndistort(py::array& arr){
    auto rows = arr.shape(0);
    auto cols = arr.shape(1);
    auto type = CV_8UC3; // 3 dim unsigned char
    auto ptr = (unsigned char*)arr.data();

    cv::Mat img(rows, cols, type, ptr);
    cv::flip(img, img, 1);
}
*/

/*
void pointwiseUndistort(cv::Mat& arr){

    cv::flip(arr, arr, 1);

}
*/


cv::Mat pointwiseUndistort(cv::Mat& arr){

    cv::flip(arr, arr, 1);
    return arr;
}



PYBIND11_MODULE(cppmodule, m){
    m.def("pointwiseUndistort", &pointwiseUndistort, py::return_value_policy::automatic);
    m.doc() = "Docstring for pointwiseUndistort function";
    
    py::class_<cv::Mat>(m, "Mat", py::buffer_protocol())      
        // custom contructor for obj creation from numpy buffer object
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            
            auto dims = info.ndim;
            auto rows = info.shape[0];
            auto cols = info.shape[1];
            
            // determine cv::mat type 
            int type;
            if(dims == 2) {// transform matrix
                type = CV_64FC2;
                printf("\n Creating 2dim cv::Mat \n");
            }
            else if (dims == 3){ // image
                type = CV_8UC3;     
                printf("\n Creating 3dim cv::Mat \n");
            }

            return cv::Mat(rows, cols, type, info.ptr);
        }))

        // expose some public attributes
        .def_readwrite("dims", &cv::Mat::dims)
        .def_readwrite("data", &cv::Mat::data)

        // define buffer object to hand over to Python C-API instead of passing by value
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
        })
    ;
}

