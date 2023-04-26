#include <stdio.h>
#include <cmath>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudev.hpp>
//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
#include <chrono>

// namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;

using namespace std;
using namespace cv;

// Cuda Kernel
__global__ void undistortKernel
(
    const cv::cuda::PtrStepSz<uchar3> src,
    cv::cuda::PtrStepSz<uchar3> dst,
    float* H
)
{
    // Get dst pixel indexes for this thread from CUDA framework
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // H*xu_hom
    float xd_hom_0 = H[0]*i + H[1]*j + H[2];
    float xd_hom_1 = H[3]*i + H[4]*j + H[5];
    float xd_hom_2 = H[6]*i + H[7]*j + H[8];

    // Convert to inhom and round to int for use as indexes
    int xd_0 = (int)(xd_hom_0 / xd_hom_2); // x
    int xd_1 = (int)(xd_hom_1 / xd_hom_2); // y

    // Get rgb value from src image 
    dst.ptr(i)[j] = src.ptr(xd_0)[xd_1];
}

/*
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
*/

int main(){

   // Loading H-coefs into array for passing to CUDA Kernel
    float H[] = {3.55082e-1, 1.51274e-1, 4.8e+1, 
                -4.27999e-1, 5.60277e-1, 3.85e+2,
                -2.72420e-4, -1.27368e-4, 1e+0};
    
    float* dPtr_H = 0;
    cudaMalloc(&dPtr_H, sizeof(H));
    cudaMemcpy(dPtr_H, H, sizeof(H), cudaMemcpyHostToDevice);


    // prep input image and return image  
    Mat img = imread("/app/_img/chessboard_perspective.jpg", IMREAD_COLOR );
    cv::cuda::GpuMat src;
    
    Mat ret;
    cv::cuda::GpuMat dst(800, 800, CV_8UC3); // allocate space 
    
    // Prep Kernel Launch
    src.upload(img);
    
    const dim3 blockSize(16,16);
    const dim3 gridSize(cv::cudev::divUp(dst.cols, blockSize.x), 
                        cv::cudev::divUp(dst.rows, blockSize.y)); // ceil: maybe not all threads used -> handle in kernel function


    // -- Kernel Launch 1 (slow) ------------------------------------------------------- 

    src.upload(img);
    
    undistortKernel<<<gridSize, blockSize>>>(src, dst, dPtr_H);
    cudaDeviceSynchronize();
    
    dst.download(ret);

    // -- Kernel Launch 2 (fast) ------------------------------------------------------- 
    auto start = chrono::steady_clock::now();
    src.upload(img);
    
    undistortKernel<<<gridSize, blockSize>>>(src, dst, dPtr_H);
    cudaDeviceSynchronize();
    
    dst.download(ret);
    auto end = chrono::steady_clock::now();

    // --------------------------------------------------------------------------

    // show results
    cout << "Elapsed time in microseconds: "
        << chrono::duration_cast<chrono::microseconds>(end - start).count()
        << " µs" << endl;

    imshow("gpu image", ret);
    waitKey(0);


    return 0;
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