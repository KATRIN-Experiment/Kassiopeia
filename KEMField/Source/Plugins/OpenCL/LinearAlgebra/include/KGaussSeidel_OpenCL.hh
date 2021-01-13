#ifndef KGAUSSSEIDEL_OPENCL_DEF
#define KGAUSSSEIDEL_OPENCL_DEF

#include "KGaussSeidel.hh"
#include "KOpenCLAction.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"

#include <limits.h>

namespace KEMField
{
template<typename ValueType> class KGaussSeidel_OpenCL : public KOpenCLAction
{
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KGaussSeidel_OpenCL(const Matrix& A, Vector& x, const Vector& b);
    ~KGaussSeidel_OpenCL();

    void ConstructOpenCLKernels() const;
    void AssignBuffers() const;

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void CompleteResidualNormalization(double& residualNorm);
    void IncrementIndex();
    void ComputeCorrection();
    void UpdateSolutionApproximation();
    void UpdateVectorApproximation();
    void CoalesceData();
    void Finalize();

    unsigned int Dimension() const
    {
        return fB.Dimension();
    }

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    mutable cl::Kernel* fInitializeVectorApproximationKernel;
    mutable cl::Kernel* fFindResidualKernel;
    mutable cl::Kernel* fFindResidualNormKernel;
    mutable cl::Kernel* fCompleteResidualNormalizationKernel;
    mutable cl::Kernel* fIncrementIndexKernel;
    mutable cl::Kernel* fComputeCorrectionKernel;
    mutable cl::Kernel* fUpdateSolutionApproximationKernel;
    mutable cl::Kernel* fUpdateVectorApproximationKernel;

    mutable cl::Buffer* fBufferResidual;
    mutable cl::Buffer* fBufferB_iterative;
    mutable cl::Buffer* fBufferCorrection;
    mutable cl::Buffer* fIndex;
    mutable cl::Buffer* fBufferPartialResidualNorm;
    mutable cl::Buffer* fBufferResidualNorm;
    mutable cl::Buffer* fBufferNWarps;
    mutable cl::Buffer* fBufferCounter;

    mutable cl::NDRange* fGlobalRange;
    mutable cl::NDRange* fLocalRange;
    mutable cl::NDRange* fGlobalRangeOffset;
    mutable cl::NDRange* fGlobalSize;
    mutable cl::NDRange* fGlobalMin;
    mutable cl::NDRange* fRangeOne;

    mutable CL_TYPE* fCLResidual;
    mutable CL_TYPE* fCLB_iterative;
    mutable CL_TYPE* fCLCorrection;
    mutable cl_int* fCLIndex;
    mutable CL_TYPE* fCLPartialResidualNorm;
    mutable CL_TYPE* fCLResidualNorm;
    mutable cl_int* fCLNWarps;
    mutable cl_int* fCLCounter;

    mutable bool fReadResidual;
};

template<typename ValueType>
KGaussSeidel_OpenCL<ValueType>::KGaussSeidel_OpenCL(const Matrix& A, Vector& x, const Vector& b) :
    KOpenCLAction((dynamic_cast<const KOpenCLAction&>(A)).GetData()),
    fA(A),
    fX(x),
    fB(b),
    fInitializeVectorApproximationKernel(NULL),
    fFindResidualKernel(NULL),
    fFindResidualNormKernel(NULL),
    fCompleteResidualNormalizationKernel(NULL),
    fIncrementIndexKernel(NULL),
    fComputeCorrectionKernel(NULL),
    fUpdateSolutionApproximationKernel(NULL),
    fUpdateVectorApproximationKernel(NULL),
    fBufferResidual(NULL),
    fBufferB_iterative(NULL),
    fBufferCorrection(NULL),
    fIndex(NULL),
    fBufferPartialResidualNorm(NULL),
    fBufferResidualNorm(NULL),
    fBufferNWarps(NULL),
    fBufferCounter(NULL),
    fCLResidual(NULL),
    fCLB_iterative(NULL),
    fCLCorrection(NULL),
    fCLIndex(NULL),
    fCLPartialResidualNorm(NULL),
    fCLResidualNorm(NULL),
    fCLNWarps(NULL),
    fCLCounter(NULL),
    fReadResidual(false)
{
    KOpenCLAction::Initialize();
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::ConstructOpenCLKernels() const
{
    // Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_GaussSeidel_kernel.cl";

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"Reading source file "
    // 			     <<"\""<<clFile.str()<<"\"...";
    //   KOpenCLInterface::GetInstance()->
    // 	Message("KTGaussSeidel_OpenCL",
    // 		"InitializeOpenCLPrimitives",
    // 		s.str(),
    // 		0,
    // 		1);
    // }

    // Read kernel source from file
    std::string sourceCode;
    std::ifstream sourceFile(clFile.str().c_str());

    sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source = {{sourceCode.c_str(), sourceCode.length() + 1}};

    // Make program of the source code in the context
    cl::Program program(KOpenCLInterface::GetInstance()->GetContext(), source, 0);

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"@Building the OpenCL Kernel (if the active GPU is running a display, communication may be temporarily interrupted)...@";
    //   KOpenCLInterface::GetInstance()->
    // 	Message("KTGaussSeidel_OpenCL",
    // 		"InitializeOpenCLPrimitives",
    // 		s.str(),
    // 		0,
    // 		2);
    // }

    // Build program for these specific devices
    try {
        // use only target device!
        CL_VECTOR_TYPE<cl::Device> devices;
        devices.push_back(KOpenCLInterface::GetInstance()->GetDevice());
        program.build(devices, (dynamic_cast<const KOpenCLAction&>(fA)).GetOpenCLFlags().c_str());
    }
    catch (cl::Error& error) {
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:"
                  << std::endl;
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        std::cout << "Build Status: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice()) << ""
                  << std::endl;
        std::cout << "Build Options:\t"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice()) << ""
                  << std::endl;
        std::cout << "Build Log:\t "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice());

        // int msgPart = 0;
        // if (fVerbose>1 && fRank == 0)
        // 	msgPart = 3;

        // std::stringstream s;
        // s<<"There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:@@";
        // // s<<error.what()<<"("<<error.err()<<")@@";
        // s<<"Build Status: "<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice())<<"@@";
        // s<<"Build Options:\t"<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice())<<"@@";
        // s<<"Build Log:\t "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice());
        // KOpenCLInterface::GetInstance()->
        // 	Message("KTGaussSeidel_OpenCL",
        // 		"InitializeOpenCLPrimitives",
        // 		s.str(),
        // 		2,
        // 		msgPart);
    }

#ifdef DEBUG_OPENCL_COMPILER_OUTPUT
    std::stringstream s;
    s << "Build Log for OpenCL " << clFile.str() << " :\t ";
    std::stringstream build_log_stream;
    build_log_stream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice())
                     << std::endl;
    std::string build_log;
    build_log = build_log_stream.str();
    if (build_log.size() != 0) {
        s << build_log;
        std::cout << s.str() << std::endl;
    }
#endif


    // Make kernels
    fInitializeVectorApproximationKernel = new cl::Kernel(program, "InitializeVectorApproximation");
    fFindResidualKernel = new cl::Kernel(program, "FindResidual");
    fFindResidualNormKernel = new cl::Kernel(program, "FindResidualNorm");
    fCompleteResidualNormalizationKernel = new cl::Kernel(program, "CompleteResidualNormalization");
    fIncrementIndexKernel = new cl::Kernel(program, "IncrementIndex");
    fComputeCorrectionKernel = new cl::Kernel(program, "ComputeCorrection");
    fUpdateSolutionApproximationKernel = new cl::Kernel(program, "UpdateSolutionApproximation");
    fUpdateVectorApproximationKernel = new cl::Kernel(program, "UpdateVectorApproximation");

    std::vector<cl::Kernel*> kernelArray;
    kernelArray.push_back(fInitializeVectorApproximationKernel);
    kernelArray.push_back(fFindResidualKernel);
    kernelArray.push_back(fFindResidualNormKernel);
    kernelArray.push_back(fCompleteResidualNormalizationKernel);
    kernelArray.push_back(fIncrementIndexKernel);
    kernelArray.push_back(fComputeCorrectionKernel);
    kernelArray.push_back(fUpdateSolutionApproximationKernel);
    kernelArray.push_back(fUpdateVectorApproximationKernel);

    fNLocal = UINT_MAX;

    for (std::vector<cl::Kernel*>::iterator it = kernelArray.begin(); it != kernelArray.end(); ++it) {
        unsigned int workgroupSize =
            (*it)->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice()));
        if (workgroupSize < fNLocal)
            fNLocal = workgroupSize;
    }

    fData.SetMinimumWorkgroupSizeForKernels(fNLocal);
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::AssignBuffers() const
{
    fNLocal = fData.GetMinimumWorkgroupSizeForKernels();
    fNWorkgroups = fData.GetNBufferedElements() / fNLocal;

    KOpenCLSurfaceContainer& container = dynamic_cast<KOpenCLSurfaceContainer&>(fData);

    fCLB_iterative = new CL_TYPE[fData.GetNBufferedElements()];

    int maxNumWarps = fData.GetNBufferedElements() / fNLocal;

    fCLNWarps = new cl_int[1];
    fCLNWarps[0] = fData.GetNBufferedElements() / fNLocal;
    fCLCounter = new cl_int[1];
    fCLCounter[0] = fData.GetNBufferedElements();

    fCLResidual = new CL_TYPE[fData.GetNBufferedElements()];
    fCLCorrection = new CL_TYPE[1];
    fCLIndex = new cl_int[1];
    fCLPartialResidualNorm = new CL_TYPE[maxNumWarps];
    fCLResidualNorm = new CL_TYPE[1];

    for (unsigned int i = 0; i < fData.GetNBufferedElements(); i++) {
        if (i < container.size())
            fCLB_iterative[i] = 0.;
        else
            fCLB_iterative[i] = 1.e30;
        fCLResidual[i] = 0.;
    }

    fCLCorrection[0] = 0.;
    fCLIndex[0] = -1;
    fCLResidualNorm[0] = 0.;

    // Create memory buffers
    fBufferResidual = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                     CL_MEM_READ_WRITE,
                                     container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE));
    fBufferB_iterative = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                        CL_MEM_READ_WRITE,
                                        container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE));
    fBufferCorrection =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, sizeof(CL_TYPE));
    fIndex = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, sizeof(cl_int));
    fBufferPartialResidualNorm =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, maxNumWarps * sizeof(CL_TYPE));
    fBufferResidualNorm =
        new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, sizeof(CL_TYPE));
    fBufferNWarps = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sizeof(cl_int));
    fBufferCounter = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, sizeof(cl_int));

    // Copy lists to the memory buffers
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferResidual,
                                                                   CL_TRUE,
                                                                   0,
                                                                   container.GetBoundarySize() *
                                                                       fData.GetNBufferedElements() * sizeof(CL_TYPE),
                                                                   fCLResidual);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferB_iterative,
                                                                   CL_TRUE,
                                                                   0,
                                                                   container.GetBoundarySize() *
                                                                       fData.GetNBufferedElements() * sizeof(CL_TYPE),
                                                                   fCLB_iterative);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferCorrection,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sizeof(CL_TYPE),
                                                                   fCLCorrection);
    KOpenCLInterface::GetInstance()->KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fIndex,
                                                                                                    CL_TRUE,
                                                                                                    0,
                                                                                                    sizeof(cl_int),
                                                                                                    fCLIndex);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferPartialResidualNorm,
                                                                   CL_TRUE,
                                                                   0,
                                                                   maxNumWarps * sizeof(CL_TYPE),
                                                                   fCLPartialResidualNorm);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferResidualNorm,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sizeof(CL_TYPE),
                                                                   fCLResidualNorm);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferNWarps,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sizeof(cl_int),
                                                                   fCLNWarps);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferCounter,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sizeof(cl_int),
                                                                   fCLCounter);

    fGlobalRange = new cl::NDRange(fData.GetNBufferedElements());
    fLocalRange = new cl::NDRange(fNLocal);
    fGlobalRangeOffset = new cl::NDRange(0);
    fGlobalSize = new cl::NDRange(fNLocal * fNWorkgroups);
    fGlobalMin = new cl::NDRange(fNLocal);
    fRangeOne = new cl::NDRange(1);

    fInitializeVectorApproximationKernel->setArg(0, *container.GetBoundaryInfo());
    fInitializeVectorApproximationKernel->setArg(1, *container.GetBoundaryData());
    fInitializeVectorApproximationKernel->setArg(2, *container.GetShapeInfo());
    fInitializeVectorApproximationKernel->setArg(3, *container.GetShapeData());
    fInitializeVectorApproximationKernel->setArg(4, *container.GetBasisData());
    fInitializeVectorApproximationKernel->setArg(5, *fBufferResidual);

    fFindResidualKernel->setArg(0, *fBufferResidual);
    fFindResidualKernel->setArg(1, *container.GetBoundaryInfo());
    fFindResidualKernel->setArg(2, *container.GetBoundaryData());
    fFindResidualKernel->setArg(3, *fBufferB_iterative);
    fFindResidualKernel->setArg(4, *fBufferCounter);

    fFindResidualNormKernel->setArg(0, *container.GetBoundaryInfo());
    fFindResidualNormKernel->setArg(1, *fBufferResidual);
    fFindResidualNormKernel->setArg(2, *fBufferResidualNorm);

    fCompleteResidualNormalizationKernel->setArg(0, *container.GetBoundaryInfo());
    fCompleteResidualNormalizationKernel->setArg(1, *container.GetBoundaryData());
    fCompleteResidualNormalizationKernel->setArg(2, *fBufferResidualNorm);

    fIncrementIndexKernel->setArg(0, *container.GetBoundaryInfo());
    fIncrementIndexKernel->setArg(1, *fIndex);

    fComputeCorrectionKernel->setArg(0, *container.GetShapeInfo());
    fComputeCorrectionKernel->setArg(1, *container.GetShapeData());
    fComputeCorrectionKernel->setArg(2, *container.GetBoundaryInfo());
    fComputeCorrectionKernel->setArg(3, *container.GetBoundaryData());
    fComputeCorrectionKernel->setArg(4, *container.GetBasisData());
    fComputeCorrectionKernel->setArg(5, *fBufferB_iterative);
    fComputeCorrectionKernel->setArg(6, *fBufferCorrection);
    fComputeCorrectionKernel->setArg(7, *fIndex);
    fComputeCorrectionKernel->setArg(8, *fBufferCounter);

    fUpdateSolutionApproximationKernel->setArg(0, *container.GetBasisData());
    fUpdateSolutionApproximationKernel->setArg(1, *fBufferCorrection);
    fUpdateSolutionApproximationKernel->setArg(2, *fIndex);

    fUpdateVectorApproximationKernel->setArg(0, *container.GetShapeInfo());
    fUpdateVectorApproximationKernel->setArg(1, *container.GetShapeData());
    fUpdateVectorApproximationKernel->setArg(2, *container.GetBoundaryInfo());
    fUpdateVectorApproximationKernel->setArg(3, *container.GetBoundaryData());
    fUpdateVectorApproximationKernel->setArg(4, *fBufferB_iterative);
    fUpdateVectorApproximationKernel->setArg(5, *fBufferCorrection);
    fUpdateVectorApproximationKernel->setArg(6, *fIndex);

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"@Done.";
    //   KOpenCLInterface::GetInstance()->
    // 	Message("KTGaussSeidel_OpenCL",
    // 		"InitializeOpenCLPrimitives",
    // 		s.str(),
    // 		0,
    // 		3);
    // }
}

template<typename ValueType> KGaussSeidel_OpenCL<ValueType>::~KGaussSeidel_OpenCL()
{
    if (fInitializeVectorApproximationKernel)
        delete fInitializeVectorApproximationKernel;
    if (fFindResidualKernel)
        delete fFindResidualKernel;
    if (fFindResidualNormKernel)
        delete fFindResidualNormKernel;
    if (fCompleteResidualNormalizationKernel)
        delete fCompleteResidualNormalizationKernel;
    if (fIncrementIndexKernel)
        delete fIncrementIndexKernel;
    if (fComputeCorrectionKernel)
        delete fComputeCorrectionKernel;
    if (fUpdateSolutionApproximationKernel)
        delete fUpdateSolutionApproximationKernel;
    if (fUpdateVectorApproximationKernel)
        delete fUpdateVectorApproximationKernel;

    if (fBufferResidual)
        delete fBufferResidual;
    if (fBufferB_iterative)
        delete fBufferB_iterative;
    if (fBufferCorrection)
        delete fBufferCorrection;
    if (fIndex)
        delete fIndex;
    if (fBufferPartialResidualNorm)
        delete fBufferPartialResidualNorm;
    if (fBufferResidualNorm)
        delete fBufferResidualNorm;
    if (fBufferNWarps)
        delete fBufferNWarps;
    if (fBufferCounter)
        delete fBufferCounter;

    if (fGlobalRange)
        delete fGlobalRange;
    if (fLocalRange)
        delete fLocalRange;
    if (fGlobalRangeOffset)
        delete fGlobalRangeOffset;
    if (fGlobalSize)
        delete fGlobalSize;
    if (fGlobalMin)
        delete fGlobalMin;
    if (fRangeOne)
        delete fRangeOne;

    if (fCLResidual)
        delete fCLResidual;
    if (fCLB_iterative)
        delete fCLB_iterative;
    if (fCLCorrection)
        delete fCLCorrection;
    if (fCLIndex)
        delete fCLIndex;
    if (fCLPartialResidualNorm)
        delete fCLPartialResidualNorm;
    if (fCLResidualNorm)
        delete fCLResidualNorm;
    if (fCLNWarps)
        delete fCLNWarps;
    if (fCLCounter)
        delete fCLCounter;
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::Initialize()
{
    if (!fReadResidual) {
        if (fX.InfinityNorm() > 1.e-16) {
            cl::Event event;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fInitializeVectorApproximationKernel,
                                                                             cl::NullRange,
                                                                             *fGlobalSize,
                                                                             *fLocalRange,
                                                                             NULL,
                                                                             &event);
            event.wait();
        }
    }
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::FindResidual()
{
    try {
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fFindResidualKernel,
                                                                         cl::NullRange,
                                                                         *fGlobalSize,
                                                                         // *fLocalRange);
                                                                         cl::NullRange);
    }
    catch (cl::Error& error) {
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::FindResidualNorm(double& residualNorm)
{
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fFindResidualNormKernel,
                                                                     cl::NullRange,
                                                                     *fRangeOne,
                                                                     *fRangeOne);
    CompleteResidualNormalization(residualNorm);
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::CompleteResidualNormalization(double& residualNorm)
{
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fCompleteResidualNormalizationKernel,
                                                                     cl::NullRange,
                                                                     *fRangeOne,
                                                                     *fRangeOne);
    cl::Event event;
    KOpenCLInterface::GetInstance()
        ->GetQueue()
        .enqueueReadBuffer(*fBufferResidualNorm, CL_TRUE, 0, sizeof(CL_TYPE), fCLResidualNorm, NULL, &event);
    event.wait();
    residualNorm = fCLResidualNorm[0];
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::IncrementIndex()
{
    try {
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fIncrementIndexKernel,
                                                                         cl::NullRange,
                                                                         *fRangeOne,
                                                                         *fRangeOne);
    }
    catch (cl::Error& error) {
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::ComputeCorrection()
{
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fComputeCorrectionKernel,
                                                                     cl::NullRange,
                                                                     *fRangeOne,
                                                                     *fRangeOne);
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::UpdateSolutionApproximation()
{
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fUpdateSolutionApproximationKernel,
                                                                     cl::NullRange,
                                                                     *fRangeOne,
                                                                     *fRangeOne);
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::UpdateVectorApproximation()
{
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fUpdateVectorApproximationKernel,
                                                                     cl::NullRange,
                                                                     *fGlobalSize,
                                                                     *fLocalRange);
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::CoalesceData()
{
    dynamic_cast<KOpenCLSurfaceContainer&>(fData).ReadBasisData();
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::Finalize()
{
    dynamic_cast<KOpenCLSurfaceContainer&>(fData).ReadBasisData();
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::SetResidualVector(const Vector& v)
{
    fReadResidual = true;

    for (unsigned int i = 0; i < v.Dimension(); i++) {
        fCLResidual[i] = v(i);
        fCLB_iterative[i] = fB(i) - fCLResidual[i];
    }

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(
        *fBufferResidual,
        CL_TRUE,
        0,
        dynamic_cast<KOpenCLSurfaceContainer&>(fData).GetBoundarySize() * fData.GetNBufferedElements() *
            sizeof(CL_TYPE),
        fCLResidual);

    cl::Event event;
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(
        *fBufferB_iterative,
        CL_TRUE,
        0,
        dynamic_cast<KOpenCLSurfaceContainer&>(fData).GetBoundarySize() * fData.GetNBufferedElements() *
            sizeof(CL_TYPE),
        fCLB_iterative,
        NULL,
        &event);
    event.wait();
}

template<typename ValueType> void KGaussSeidel_OpenCL<ValueType>::GetResidualVector(Vector& v) const
{
    cl::Event event;
    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferResidual,
                                                                  CL_TRUE,
                                                                  0,
                                                                  fData.GetNBufferedElements() * sizeof(CL_TYPE),
                                                                  fCLResidual,
                                                                  NULL,
                                                                  &event);
    event.wait();

    for (unsigned int i = 0; i < v.Dimension(); i++)
        v[i] = fCLResidual[i];
}
}  // namespace KEMField

#endif /* KGAUSSSEIDEL_OPENCL_DEF */
