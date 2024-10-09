#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>


namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;


// Cuda Kernel
__global__ void transformKernel
(
    const cv::cuda::PtrStepSz<uchar3> src,
    cv::cuda::PtrStepSz<uchar3> dst,
    double* H
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


cv::Mat pointwiseUndistort( 
    py::array_t<imgScalar>& py_queryImage, 
    py::array_t<matScalar>& py_H, 
    py::tuple py_retImage_shape 
)
{
    // --- Input data preparation ----------------------------------------------------------
    cv::Mat queryImage(
        py_queryImage.shape(0),               // rows
        py_queryImage.shape(1),               // cols
        CV_8UC3,                              // data type
        (imgScalar*)py_queryImage.data());    // data pointer

    
    // Link py_H data to C-array
    double* arrH = (matScalar*)py_H.data(); // or: const double* arrH = py_H.data();
    double* d_ptr_H;                       // Device pointer for H-array on device                                        
    
    // Allocate space on device and copy H-array there
    cudaMalloc(&d_ptr_H, (3*3*3)*sizeof(double));           // allocate space on device
    cudaMemcpy( d_ptr_H, arrH,                              // destination, source
                py_H.shape(0)*py_H.shape(1)*sizeof(double),   // size
                cudaMemcpyHostToDevice);                    // direction

    // Unpack python tuple into integers
    auto M = py_retImage_shape[0].cast<uint32_t>();
    auto N = py_retImage_shape[1].cast<uint32_t>();

    // --- CUDA Host Code ------------------------------------------------------------------
   
    // Query (input) image
    cv::cuda::GpuMat d_queryImage; 
    
    // Output image
    cv::cuda::GpuMat d_outputImage(M, N, CV_8UC3, cv::Scalar(0,0,0)); // device memory
    cv::Mat outputImage;                                              // host memory

    // Kernel launch params
    const dim3 blockSize(16,16);
    const dim3 gridSize(cv::cudev::divUp(d_outputImage.cols, blockSize.x), 
                        cv::cudev::divUp(d_outputImage.rows, blockSize.y)); 

    
    // -- Kernel launch 1 (initializer run) --
    auto start_1 = std::chrono::steady_clock::now();
    d_queryImage.upload(queryImage);
    transformKernel<<<gridSize, blockSize>>>(d_queryImage, 
                                             d_outputImage, 
                                             d_ptr_H);
    
    cudaDeviceSynchronize(); // Wait for all kernels to finsh
    d_outputImage.download(outputImage); // download download output back to host
    auto end_1 = std::chrono::steady_clock::now();

    // -- Kernel launch 2 --
    auto start_2 = std::chrono::steady_clock::now();
    d_queryImage.upload(queryImage);
    transformKernel<<<gridSize, blockSize>>>(d_queryImage, 
                                             d_outputImage, 
                                             d_ptr_H);
    
    cudaDeviceSynchronize(); // Wait for all kernels to finsh
    d_outputImage.download(outputImage); // download download output back to host
    auto end_2 = std::chrono::steady_clock::now();

    // Free up device resources allocated using malloc (others handled automatically)
    cudaFree(d_ptr_H);
    

    // Print runtime results
    std::cout << "Runtime 1st kernel launch in microseconds: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count()
    << " µs" << std::endl;

    std::cout << "Runtime 2nd kernel launch in microseconds: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count()
    << " µs" << std::endl;
    
    return outputImage;
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