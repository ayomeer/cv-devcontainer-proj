#include <stdio.h>
#include <cmath>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudev.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>

namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;


using namespace std;
using namespace cv;


// === Interface Class ======================================================================

class CppHomography {
public:
    CppHomography() {}
    void setMember(int newVal) { memberVar = newVal; }
    int getMember() { return memberVar; }

    int publicVar;

    void pointwiseUndistort( py::array_t<imgScalar>& pyImg_d, 
                        py::array_t<matScalar>& pyH, 
                        py::tuple img_u_shape,
                        py::array_t<float> pyContour);

private:
    int memberVar;

};




// === Cuda device code (Kernel) ============================================================
__device__ void deviceFcn() {

}

__global__ void undistortKernel
(
    const cv::cuda::PtrStepSz<uchar3> src,
    cv::cuda::PtrStepSz<uchar3> dst,
    double* H,
    cv::InputArray c
)
{
    // Get dst pixel indexes for this thread from CUDA framework
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    /* // Homography Matprod
    // H*xu_hom 
    float xd_hom_0 = H[0]*i + H[1]*j + H[2];
    float xd_hom_1 = H[3]*i + H[4]*j + H[5];
    float xd_hom_2 = H[6]*i + H[7]*j + H[8];

    // Convert to inhom and round to int for use as indexes
    int xd_0 = (int)(xd_hom_0 / xd_hom_2); // x
    int xd_1 = (int)(xd_hom_1 / xd_hom_2); // y

    // Get rgb value from src image 
    dst.ptr(i)[j] = src.ptr(xd_0)[xd_1];
    */
    deviceFcn();

}


void CppHomography::pointwiseUndistort( py::array_t<imgScalar>& pyImg_d, 
                        py::array_t<matScalar>& pyH, 
                        py::tuple img_u_shape,
                        py::array_t<float> pyContour
)
{
    // === Input data preparation ===========================================================  
    // Link pyImg_d data to cv::Mat object img
    Mat img_d(
        pyImg_d.shape(0),               // rows
        pyImg_d.shape(1),               // cols
        CV_8UC3,                        // data type
        (imgScalar*)pyImg_d.data());    // data pointer
    
    // Link pyH data to C-array
    double* arrH = (matScalar*)pyH.data(); // or: const double* arrH = pyH.data();

    // Cast py::tuple into int
    int M = img_u_shape[0].cast<int>();
    int N = img_u_shape[1].cast<int>();

    // Link pyContour to C-array
    //const int* arrC = pyContour.data();
    Mat contour(
        pyContour.shape(0),
        pyContour.shape(1),
        CV_32FC1,
        (float*)pyContour.data()
    );
    cv::InputArray inpContour(contour);

    // ===  CUDA host code ==================================================================

    // Load H-array onto device, so it doesn't have to be passed to each kernel individually
    double* dPtr_H = 0; // device pointer to copy of H on GPU
    cudaMalloc(&dPtr_H, pyH.shape(0)*pyH.shape(1)*sizeof(double));
    cudaMemcpy(dPtr_H, arrH, pyH.shape(0)*pyH.shape(1)*sizeof(double), cudaMemcpyHostToDevice);


    // Prep input and return images
    Mat img;
    cvtColor(img_d, img, COLOR_RGB2BGR);
    
    cv::cuda::GpuMat src;
    
    Mat ret;
    cv::cuda::GpuMat dst(M, N, CV_8UC3); // allocate space for dst image
    
    // Prep kernel launch
    src.upload(img);
    
    const dim3 blockSize(16,16);
    const dim3 gridSize(cv::cudev::divUp(dst.cols, blockSize.x), 
                        cv::cudev::divUp(dst.rows, blockSize.y)); 


    // -- Kernel Launch 1 (slow) ------------------------------------------------------------ 

    src.upload(img);
    
    undistortKernel<<<gridSize, blockSize>>>(src, dst, dPtr_H, inpContour);
    cudaDeviceSynchronize();
    
    dst.download(ret);

    // -- Kernel Launch 2 (fast) ----------------------------------------------------------- 
    auto start = chrono::steady_clock::now();
    src.upload(img);
    
    undistortKernel<<<gridSize, blockSize>>>(src, dst, dPtr_H, inpContour);
    cudaDeviceSynchronize();
    
    dst.download(ret);
    auto end = chrono::steady_clock::now();

    // -------------------------------------------------------------------------------------

    // show results
    cout << "tKernel: "
        << chrono::duration_cast<chrono::microseconds>(end - start).count()
        << " µs" << endl;

    imshow("ret image", ret);
    waitKey(0);


    return;
}       

PYBIND11_MODULE(cppmodule, m){
    m.doc() = "Docstring for cpp homography module";
    // m.def("pointwiseUndistort", &pointwiseUndistort, py::return_value_policy::automatic);

    py::class_<CppHomography>(m, "CppHomography")
        .def(py::init()) // Wrap class constructor
        .def("setMember", &CppHomography::setMember)
        .def("getMember", &CppHomography::getMember)
        .def("pointwiseUndistort", &CppHomography::pointwiseUndistort)
        .def_readwrite("publicVar", &CppHomography::publicVar)
        ;

    




    // Returning Mat to Python as Numpy array
    py::class_<cv::Mat>(m, "Mat", py::buffer_protocol()) 
        .def_buffer([](cv::Mat &im) -> py::buffer_info { // for returning cvMat as pyBuffer
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
