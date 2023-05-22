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

        //printf("pVect[%d][0]: %d \n", i, pVect[i][0]);
        //printf("pVect[%d][1]: %d \n", i, pVect[i][1]);
    }
 
    int inside = 1;
    
    for (int i = 0; i < 4; ++i){
        int x_idx = i*2; 
        int y_idx = i*2+1;
        int dotP_i = pVect[i][0] * polyNrm[x_idx] + pVect[i][1] * polyNrm[y_idx];
        // printf("%d * %d + %d * %d = %d \n",  pVect[i][0], polyNrm[x_idx], pVect[i][1], polyNrm[y_idx], dotP_i);
        if (dotP_i < 0){inside = 0;}
    }
    return inside; 
}

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

    if (pointInPoly(pt, polyPts, polyNrm)){
        // H*xu_hom 
        float xd_hom_0 = H[0]*j + H[1]*i + H[2];
        float xd_hom_1 = H[3]*j + H[4]*i + H[5];
        float xd_hom_2 = H[6]*j + H[7]*i + H[8];

        // Convert to inhom and round to int for use as indexes
        int xd_0 = (int)(xd_hom_0 / xd_hom_2); // x
        int xd_1 = (int)(xd_hom_1 / xd_hom_2); // y

        // Get rgb value from d_queryImage image 
        outputImage.ptr(j)[i] = queryImage.ptr(xd_0)[xd_1];
    }
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
    cudaMalloc(&d_ptr_H, (3*3)*sizeof(double));
    cudaMalloc(&d_ptr_polyPts, (4*2)*sizeof(int));
    cudaMalloc(&d_ptr_polyNrm, (4*2)*sizeof(int));
}

HomographyReconstruction::~HomographyReconstruction(){
    // Free device memory
    cudaFree(d_ptr_H);
    cudaFree(d_ptr_polyPts);
    cudaFree(d_ptr_polyNrm);
}

cv::Mat HomographyReconstruction::getQueryImage(){
    return queryImage; // for conversion code, see end of file (PYBIND11_MODULE)
}

void HomographyReconstruction::showQueryImage(){
    cv::imshow("queryImage", queryImage);
    cv::waitKey(1);
    return;
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

    cv::cuda::GpuMat d_outputImage(M, N, CV_8UC3, cv::Scalar(0,0,0)); // allocate space for d_outputImage image

    
    // --- Start of repeated code ---------------------------------------------------------
    // Copy H-matrices onto device (the same for each kernel)
    cudaMemcpy(d_ptr_H, arrH, pyH.shape(0)*pyH.shape(1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_polyPts, polyPts, 4*2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_polyNrm, polyNrm, 4*2*sizeof(int), cudaMemcpyHostToDevice);
    
    // Prep kernel launch
    d_queryImage.upload(queryImage);
    
    const dim3 blockSize(16,16);
    const dim3 gridSize(cv::cudev::divUp(d_outputImage.cols, blockSize.x), 
                        cv::cudev::divUp(d_outputImage.rows, blockSize.y)); 

    

    
    // Kernel Launch 2
    d_queryImage.upload(queryImage);
    
    transformKernel<<<gridSize, blockSize>>>(d_queryImage, 
                                             d_outputImage, 
                                             d_ptr_H, 
                                             d_ptr_polyPts, 
                                             d_ptr_polyNrm);
    cudaDeviceSynchronize();
    
    d_outputImage.download(outputImage);
    // -------------------------------------------------------------------------------------

    // show results
    // imshow("Image", outputImage);
    // waitKey(15);

    return outputImage;
}       

// auto start = chrono::steady_clock::now();
// auto end = chrono::steady_clock::now();
// cout << "Kernel Run Time: "
// << chrono::duration_cast<chrono::microseconds>(end - start).count()
// << " µs" << endl;

// === OTHER ====================================================================================

void renderTest(){

    Mat img(100, 100, CV_8UC1, Scalar(0));
    auto M = img.cols;
    auto N = img.rows;

    imshow("imshow", img);
    waitKey(1);

    for (std::size_t i = 0; i < M; ++i){
        for (std::size_t j = 0; j < N; ++j){
            img.at<char>(i,j) = 255;
            imshow("imshow", img);
            waitKey(1);
        }
    }
    imshow("imshow", img);
    waitKey(0);
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
    m.def("renderTest", &renderTest);

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
