#include <stdio.h>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>

namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;

using namespace std;
using namespace cv;
 
 /*
 __global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}*/


__global__ void undistortKernel
(
    cv::cuda::PtrStepSz<uchar3> img_u, 
    cv::cuda::PtrStepSz<uchar3> img_d
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    img_u.ptr(i)[j] = img_d.ptr(i)[j];
 


}

Mat pointwiseUndistort( py::array_t<imgScalar>& pyImg_d, 
                        py::array_t<matScalar>& pyH, 
                        py::tuple img_u_shape ){

    // --- Input data preparation --------------------------------------
     
    // link pyImg_d data to cv::Mat object img
    Mat img_d(
        pyImg_d.shape(0),               // rows
        pyImg_d.shape(1),               // cols
        CV_8UC3,                        // data type
        (imgScalar*)pyImg_d.data());    // data pointer
    
    cuda::GpuMat img_d_gpu(img_d); // create GpuMat from regular Mat

    // link H data to cv::Mat object
    Mat H(
        pyH.shape(0),                   // rows
        pyH.shape(1),                   // cols
        CV_64FC1,                       // data type
        (matScalar*)pyH.data());        // data pointer

    int M = img_u_shape[0].cast<int>();
    int N = img_u_shape[1].cast<int>();

    Mat img_u(M, N, CV_8UC3); // prepare return image
    cuda::GpuMat img_u_gpu(img_u); // create GpuMat from regular Mat

    // ---  Algorithm --------------------------------------------------

    // pointer to gpu data
    //cuda::GpuMat* u;
    //cuda::GpuMat* d;

    // allocate space of target image u
    //cudaMallocManaged(&u, sizeof(img_u));
    //cudaMallocManaged(&d, sizeof(img_d));
    
    // run kernels
    auto numBlocks = 2500;
    dim3 blockSize(16,16);
    dim3 gridSize(ceil(numBlocks/16), ceil(numBlocks/16)); // ceil: maybe not all threads used -> handle in kernel function
    
    undistortKernel<<<gridSize, blockSize>>>(img_u_gpu, img_d_gpu);
    cudaDeviceSynchronize();
    

    
    
    // free space
    //cudaFree(u);


    Mat ret = img_u_gpu;
    return ret;
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
                    {                                                    // strides in bytes
                        sizeof(unsigned char) * im.channels() * im.cols, // (issue with padding)
                        sizeof(unsigned char) * im.channels(),
                        sizeof(unsigned char)
                    }
                );
            })
        ;
    }
