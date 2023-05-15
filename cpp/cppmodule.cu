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
// Desc: Manages and runs homographies from queryImage to outputImage
class HomographyReconstruction {
public:
    HomographyReconstruction(py::array_t<imgScalar>& py_queryImage);
    ~HomographyReconstruction();

    cv::Mat getQueryImage(); // return gets changed to py::buffer_protocol

    // kernel launch function
    cv::Mat pointwiseTransform(
        py::array_t<matScalar>& pyH,
        const py::array_t<int>& py_polyPts,
        const py::array_t<int>& py_polyNrm);

    __device__ __host__ int pointInPoly(
        const int* pt,
        const int* polyPts
    );
    int py_pointInPoly( // Python Wrapper for pointInPoly for testing (unit test)
        const py::array_t<int>& py_polyPts,
        const py::array_t<int>& py_polyNrm);

private:
    cv::Mat queryImage;
    double* d_ptrH;
};

HomographyReconstruction::HomographyReconstruction(py::array_t<imgScalar>& py_queryImage){
    // Link py_queryImage data to cv::Mat object queryImage, member of HomographyReconstruction class
    queryImage = Mat(
        py_queryImage.shape(0),               // rows
        py_queryImage.shape(1),               // cols
        CV_8UC3,                              // data type
        (imgScalar*)py_queryImage.data());    // data pointer

    cvtColor(queryImage, queryImage, COLOR_RGB2BGR); // change color interpretation to openCV's
    
    // Allocate space on device for H-matrices and create pointer to them
    double* d_ptrH = 0; // device pointer to copy of H on GPU
    cudaMalloc(&d_ptrH, (3*3)*sizeof(double));

}

HomographyReconstruction::~HomographyReconstruction(){
    cudaFree(d_ptrH);
}

cv::Mat HomographyReconstruction::getQueryImage(){
    return queryImage; // for conversion code, see end of file (PYBIND11_MODULE)
}


// === Cuda device code (Kernel) ============================================================


int HomographyReconstruction::py_pointInPoly(
    const py::array_t<int>& py_pt, 
    const py::array_t<int>& py_polyPts
){
    // py types to C-types
    const int* pt = py_pt.data();
    const int* polyPts = py_polyPts.data();


    return pointInPoly(pt, polyPts);
}

int HomographyReconstruction::pointInPoly(
    const int* pt,
    const int* polyPts
){
    int pVect[4][2] = {0};
    for (std::size_t i = 0; i < 4; ++i){
        // x coord
        int x_idx = i*2; 
        int y_idx = i*2+1; 
        pVect[i][0] = pt[0] - polyPts[x_idx];
        pVect[i][1] = pt[1] - polyPts[y_idx];
    }
 
    return pVect[1][0];
}

__global__ void transformKernel
(
    const cv::cuda::PtrStepSz<uchar3> d_queryImage,
    cv::cuda::PtrStepSz<uchar3> d_outputImage,
    double* H,
    const int* polyPts
)
{
    
    // Get d_outputImage pixel indexes for this thread from CUDA framework
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check which polygon the point (i, j) is in



    /* // HomographyReconstruction Matprod
    // H*xu_hom 
    float xd_hom_0 = H[0]*i + H[1]*j + H[2];
    float xd_hom_1 = H[3]*i + H[4]*j + H[5];
    float xd_hom_2 = H[6]*i + H[7]*j + H[8];

    // Convert to inhom and round to int for use as indexes
    int xd_0 = (int)(xd_hom_0 / xd_hom_2); // x
    int xd_1 = (int)(xd_hom_1 / xd_hom_2); // y

    // Get rgb value from d_queryImage image 
    d_outputImage.ptr(i)[j] = d_queryImage.ptr(xd_0)[xd_1];
    */

   d_outputImage.ptr(i)[j] = d_queryImage.ptr(j)[i];

}


cv::Mat HomographyReconstruction::pointwiseTransform(
    py::array_t<matScalar>& pyH,
    const py::array_t<int>& py_polyPts,
    const py::array_t<int>& py_polyNrm
)
{
    // --- Input data preparation ----------------------------------------------------------
   
    // Link pyH data to C-array
    double* arrH = (matScalar*)pyH.data(); // or: const double* arrH = pyH.data();

    // Link py_polyPts to C-array
    const int* polyPts = py_polyPts.data();

    // Link py_polyNrm to C-array
    const int* polyNrm = py_polyNrm.data();

    // --- CUDA Host Code ------------------------------------------------------------------
    // Prepare images
    cv::Mat outputImage;
    cv::cuda::GpuMat d_queryImage;
    
    auto M = queryImage.rows;
    auto N = queryImage.cols;

    cv::cuda::GpuMat d_outputImage(M, N, CV_8UC3); // allocate space for d_outputImage image


    // Copy H-matrices onto device (the same for each kernel)
    cudaMemcpy(d_ptrH, arrH, pyH.shape(0)*pyH.shape(1)*sizeof(double), cudaMemcpyHostToDevice);
    
    // Prep kernel launch
    d_queryImage.upload(queryImage);
    
    const dim3 blockSize(16,16);
    const dim3 gridSize(cv::cudev::divUp(d_outputImage.cols, blockSize.x), 
                        cv::cudev::divUp(d_outputImage.rows, blockSize.y)); 


    // Kernel Launch 1 (slow) 
    d_queryImage.upload(queryImage);
    
    transformKernel<<<gridSize, blockSize>>>(d_queryImage, d_outputImage, d_ptrH, polyPts);
    cudaDeviceSynchronize();
    
    d_outputImage.download(outputImage);

    // -------------------------------------------------------------------------------------

    // show results
    // imshow("outputImage image", outputImage);
    // waitKey(0);

    return outputImage;
}       

PYBIND11_MODULE(cppmodule, m){
    m.doc() = "Docstring for cpp homography module";
    // m.def("pointwiseTransform", &pointwiseTransform, py::return_value_policy::automatic);

    py::class_<HomographyReconstruction>(m, "HomographyReconstruction")
        .def(py::init<py::array_t<imgScalar>&>()) // Wrap class constructor
        .def("pointwiseTransform", &HomographyReconstruction::pointwiseTransform)
        .def("getQueryImage", &HomographyReconstruction::getQueryImage)
        // examples for other class element types
        // .def_readwrite("publicVar", &HomographyReconstruction::publicVar)
        .def("py_pointInPoly", &HomographyReconstruction::py_pointInPoly)
    ;

    




    // Returning Mat to Python as Numpy array (not currently needed)
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
