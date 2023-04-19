#include <stdio.h>
#include <opencv2/opencv.hpp>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
#include <chrono>

// namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;

using namespace std;
using namespace cv;
 
 /*
__global__ void undistortKernel( Mat& img_u, Mat& img_d, Mat& H){
}*/

__global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
/*
Mat pointwiseUndistort( py::array_t<imgScalar>& pyImg_d, 
                        py::array_t<matScalar>& pyH, 
                        py::tuple img_u_shape ){
*/

int main(){
    // -- Data type management
    // link pyImg_d data to cv::Mat object img
    /*
    Mat img_d(
        pyImg_d.shape(0),               // rows
        pyImg_d.shape(1),               // cols
        CV_8UC3,                        // data type
        (imgScalar*)pyImg_d.data());    // data pointer

    // link H data to cv::Mat object
    Mat H(
        pyH.shape(0),                   // rows
        pyH.shape(1),                   // cols
        CV_64FC1,                       // data type
        (matScalar*)pyH.data());        // data pointer

    int M = img_u_shape[0].cast<int>();
    int N = img_u_shape[1].cast<int>();

    // -- Algorithm

    Mat img_u(M, N, CV_8UC3); // prepare return image
    */
    
    // --- unified memory test ---

    // return pointer
    int *ret;
    // allocate space of array to pointer
    cudaMallocManaged(&ret, 1000 * sizeof(int));
    
    // run kernels
    AplusB<<< 1, 1000 >>>(ret, 10, 100);
    cudaDeviceSynchronize();
    
    // output
    for(int i = 0; i < 1000; i++)
        printf("%d: A+B = %d\n", i, ret[i]);
    
    // free space
    cudaFree(ret);


    // Free memory



    return 0;//img_d;//img_u;
}       
/*
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
                        {                                                    // strides in bytes
                            sizeof(unsigned char) * im.channels() * im.cols, // (issue with padding)
                            sizeof(unsigned char) * im.channels(),
                            sizeof(unsigned char)
                        }
                    );
                })
            ;
    }
*/