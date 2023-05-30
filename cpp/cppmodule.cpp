#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


typedef std::uint8_t imgScalar;
typedef double matScalar;

using namespace std;
using namespace cv;
namespace py = pybind11;

// Sample function
void sayHello(){
	printf("Hello from C++! \n");
}


// Implement functions here



// Python bindings
PYBIND11_MODULE(cppmodule, m){
	m.doc() = "Cpp Module for fast algorithm implementations callable from Python";

	// Sample function binding
	m.def("sayHello", &sayHello);

	// Binding for cv::Mat class so these object can be passed back as return value
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
									{                                                    // strides in bytes
											sizeof(unsigned char) * im.channels() * im.cols, // (issue with padding)
											sizeof(unsigned char) * im.channels(),
											sizeof(unsigned char)
									}
							);
					})
			;
}
