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

// === Cuda device code (Kernel) ============================================================

__host__ __device__ int pointInPoly(
    const int* pt,
    const int* polyPts,
    const int* polyNrm
){

    // Create vectors from each polygon vertex to the point to check
    int pVect[4][2] = {0};
    for (int i = 0; i < 4; ++i){
        int x_idx = i*2; 
        int y_idx = i*2+1; 
        pVect[i][0] = pt[0] - polyPts[x_idx];
        pVect[i][1] = pt[1] - polyPts[y_idx];
    }
 
    int inside = 1;
    
    for (int i = 0; i < 4; ++i){
        int x_idx = i*2; 
        int y_idx = i*2+1;
        int dotP_i = pVect[i][0] * polyNrm[x_idx] + pVect[i][1] * polyNrm[y_idx];
        if (dotP_i < 0){inside = 0;}
    }
    return inside; 
}

enum H_idx_offsets {H_A_offset=0*3*3, H_B_offset=1*3*3, H_C_offset=2*3*3};
enum pt_offsets {pts_A_offset=0*4*2, pts_B_offset=1*4*2, pts_C_offset=2*4*2};
enum nrm_offsets {nrm_A_offset=0*4*2, nrm_B_offset=1*4*2, nrm_C_offset=2*4*2};

__global__ void transformKernel
(
    const cv::cuda::PtrStepSz<uchar3> queryImage,
    cv::cuda::PtrStepSz<uchar3> outputImage,
    double* H,
    const int* polyPts,
    const int* polyNrm
)
{
    // Get d_outputImage pixel indexes for this thread from CUDA framework
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // right
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // down
    int pt[2] = {j, i}; //coord switch: cv --> x-down

    int H_offset = 0;
    if (pointInPoly(pt, &polyPts[pts_A_offset], &polyNrm[nrm_A_offset])){
        H_offset = H_A_offset;
    }
    else if (pointInPoly(pt, &polyPts[pts_B_offset], &polyNrm[nrm_B_offset])){
        H_offset = H_B_offset;
    }
    else if (pointInPoly(pt, &polyPts[pts_C_offset], &polyNrm[nrm_C_offset])){
        H_offset = H_C_offset;
    }
    else {
        // Point in none of the polygons -> nothing to do
        return;
    }

    // Do transform depending on poly (H_X*xu_hom)
    float xd_hom_0 = H[0+H_offset]*j + H[1+H_offset]*i + H[2+H_offset];
    float xd_hom_1 = H[3+H_offset]*j + H[4+H_offset]*i + H[5+H_offset];
    float xd_hom_2 = H[6+H_offset]*j + H[7+H_offset]*i + H[8+H_offset];

    // Convert to inhom and round to int for use as indexes
    int xd_0 = (int)(xd_hom_0 / xd_hom_2); // x
    int xd_1 = (int)(xd_hom_1 / xd_hom_2); // y

    // Get rgb value from d_queryImage image 
    outputImage.ptr(j)[i] = queryImage.ptr(xd_0)[xd_1];

    return;
}

// === Interface Class ======================================================================
// Desc: Manages and runs homographies from queryImage to outputImage
class HomographyReconstruction {
public:
    HomographyReconstruction(py::array_t<imgScalar>& py_queryImage);
    ~HomographyReconstruction();

    cv::Mat getQueryImage(); // return gets changed to py::buffer_protocol
    void showQueryImage();

    // kernel launch function
    cv::Mat pointwiseTransform(
        py::array_t<matScalar>& pyH,
        const py::array_t<int>& py_polyPts,
        const py::array_t<int>& py_polyNrm);

private:
    cv::Mat queryImage;
    double* d_ptr_H;
    int* d_ptr_polyPts;
    int* d_ptr_polyNrm;
};

// --- Method Definitions --------------------------------------------------------------------
HomographyReconstruction::HomographyReconstruction(py::array_t<imgScalar>& py_queryImage)
    : d_ptr_H(NULL), d_ptr_polyPts(NULL), d_ptr_polyNrm(NULL)
{
    // Link py_queryImage data to cv::Mat object queryImage, member of HomographyReconstruction class
    queryImage = Mat(
        py_queryImage.shape(0),               // rows
        py_queryImage.shape(1),               // cols
        CV_8UC3,                              // data type
        (imgScalar*)py_queryImage.data());    // data pointer

    cvtColor(queryImage, queryImage, COLOR_RGB2BGR); // change color interpretation to openCV's

    // Allocate space on device for H-matrices and poly data and link device pointers
    cudaMalloc(&d_ptr_H, (3*3*3)*sizeof(double));
    cudaMalloc(&d_ptr_polyPts, (3*4*2)*sizeof(int));
    cudaMalloc(&d_ptr_polyNrm, (3*4*2)*sizeof(int));
}

HomographyReconstruction::~HomographyReconstruction(){
    // Free device memory
    cudaFree(d_ptr_H);
    cudaFree(d_ptr_polyPts);
    cudaFree(d_ptr_polyNrm);
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
    const int* polyPts = (const int*)py_polyPts.data();

    // Link py_polyNrm to C-array
    const int* polyNrm = (const int*)py_polyNrm.data();

    // --- CUDA Host Code ------------------------------------------------------------------
    // Prepare images
    cv::Mat outputImage;
    cv::cuda::GpuMat d_queryImage;
    
    auto M = queryImage.rows;
    auto N = queryImage.cols;

    cv::cuda::GpuMat d_outputImage(M, N, CV_8UC3, cv::Scalar(0,0,0)); // allocate space for d_outputImage image

    
    // --- Start of repeated code ---------------------------------------------------------
    // Copy H-matrices onto device (the same for each kernel)
    cudaMemcpy(d_ptr_H, arrH, pyH.shape(0)*pyH.shape(1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_polyPts, polyPts, 3*4*2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_polyNrm, polyNrm, 3*4*2*sizeof(int), cudaMemcpyHostToDevice);
    
    // Prep kernel launch
    d_queryImage.upload(queryImage);
       
    const dim3 blockSize(16,16);
    const dim3 gridSize(cv::cudev::divUp(d_outputImage.cols, blockSize.x), 
                        cv::cudev::divUp(d_outputImage.rows, blockSize.y)); 

    // Kernel Launch 
    transformKernel<<<gridSize, blockSize>>>(d_queryImage, 
                                             d_outputImage, 
                                             d_ptr_H, 
                                             d_ptr_polyPts, 
                                             d_ptr_polyNrm);
    cudaDeviceSynchronize();
    
    d_outputImage.download(outputImage);

    // show results
    // imshow("Image", outputImage);
    // waitKey(15);

    return outputImage;
}       

cv::Mat HomographyReconstruction::getQueryImage(){
    return queryImage; // for conversion code, see end of file (PYBIND11_MODULE)
}

void HomographyReconstruction::showQueryImage(){
    cv::imshow("queryImage", queryImage);
    cv::waitKey(1);
    return;
}

// === Python Interfacing =========================================================================
PYBIND11_MODULE(cppmodule, m){
    m.doc() = "Docstring for cpp homography module";
    m.def("pointInPoly", []( // Python interface wrapper     
                                    const py::array_t<int>& py_pt, 
                                    const py::array_t<int>& py_polyPts,
                                    const py::array_t<int>& py_polyNrm
                               ){
                                    const int* pt = py_pt.data();
                                    const int* polyPts = py_polyPts.data();
                                    const int* polyNrm = py_polyNrm.data();

                                    return pointInPoly(pt, polyPts, polyNrm);
                               });
    // m.def("renderTest", &renderTest);

    py::class_<HomographyReconstruction>(m, "HomographyReconstruction")
        .def(py::init<py::array_t<imgScalar>&>()) // Wrap class constructor
        .def("pointwiseTransform", &HomographyReconstruction::pointwiseTransform)
        .def("getQueryImage", &HomographyReconstruction::getQueryImage)
        .def("showQueryImage", &HomographyReconstruction::showQueryImage)
        
        // examples for other class element types
        // .def_readwrite("publicVar", &HomographyReconstruction::publicVar)
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
