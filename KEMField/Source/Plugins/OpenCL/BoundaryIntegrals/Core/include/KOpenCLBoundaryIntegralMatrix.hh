#ifndef KOPENCLBOUNDARYINTEGRALMATRIX_DEF
#define KOPENCLBOUNDARYINTEGRALMATRIX_DEF

#include "KBoundaryIntegralMatrix.hh"
#include "KOpenCLAction.hh"
#include "KOpenCLBoundaryIntegrator.hh"
#include "KOpenCLSurfaceContainer.hh"

#include <fstream>
#include <sstream>

namespace KEMField
{
template<class BasisPolicy>
class KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>> :
    public KSquareMatrix<typename BasisPolicy::ValueType>,
    public KOpenCLAction
{
  public:
    typedef typename BasisPolicy::ValueType ValueType;
    friend class KOpenCLSurfaceContainer;

    KBoundaryIntegralMatrix(KOpenCLSurfaceContainer& c, KOpenCLBoundaryIntegrator<BasisPolicy>& integrator);

    ~KBoundaryIntegralMatrix() override;

    unsigned int Dimension() const
    {
        return fDimension;
    }

    const ValueType& operator()(unsigned int i, unsigned int j) const;

    void SetNLocal(int nLocal) const
    {
        fNLocal = nLocal;
    }
    int GetNLocal() const
    {
        return fNLocal;
    }

    KOpenCLBoundaryIntegrator<BasisPolicy>& GetIntegrator() const
    {
        return fIntegrator;
    }

    std::string GetOpenCLFlags() const override;

  private:
    KOpenCLSurfaceContainer& fContainer;
    KOpenCLBoundaryIntegrator<BasisPolicy>& fIntegrator;

    const unsigned int fDimension;

    void ConstructOpenCLKernels() const override;
    void AssignBuffers() const override;

    mutable int fNLocal;

    mutable cl::Kernel* fGetMatrixElementKernel;
    mutable cl::Kernel* fGetVectorElementKernel;

    mutable cl::Buffer* fBufferIJ;
    mutable cl::Buffer* fBufferValue;

    mutable ValueType fValue;
};

template<class BasisPolicy>
KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>>::KBoundaryIntegralMatrix(
    KOpenCLSurfaceContainer& c, KOpenCLBoundaryIntegrator<BasisPolicy>& integrator) :
    KSquareMatrix<ValueType>(),
    KOpenCLAction(c),
    fContainer(c),
    fIntegrator(integrator),
    fDimension(c.size() * BasisPolicy::Dimension),
    fNLocal(-1),
    fGetMatrixElementKernel(nullptr),
    fBufferIJ(nullptr),
    fBufferValue(nullptr)
{}

template<class BasisPolicy> KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>>::~KBoundaryIntegralMatrix()
{
    if (fGetMatrixElementKernel)
        delete fGetMatrixElementKernel;

    if (fBufferIJ)
        delete fBufferIJ;
    if (fBufferValue)
        delete fBufferValue;
}

template<class BasisPolicy>
const typename BasisPolicy::ValueType&
KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>>::operator()(unsigned int i, unsigned int j) const
{
    cl_int ij[2];
    ij[0] = static_cast<int>(i);
    ij[1] = static_cast<int>(j);


    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferIJ, CL_TRUE, 0, 2 * sizeof(cl_int), ij);

    cl::NDRange global(1);
    cl::NDRange local(1);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fGetMatrixElementKernel,
                                                                     cl::NullRange,
                                                                     global,
                                                                     local);

    CL_TYPE value;
    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferValue, CL_TRUE, 0, sizeof(CL_TYPE), &value);
    fValue = value;
    return fValue;
}

template<class BasisPolicy>
std::string KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>>::GetOpenCLFlags() const
{
    return (fIntegrator.GetOpenCLFlags() + fData.GetOpenCLFlags());
}


template<class BasisPolicy>
void KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>>::ConstructOpenCLKernels() const
{
    // Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_LinearAlgebra_kernel.cl";

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"Reading source file "
    // 			     <<"\""<<clFile.str()<<"\"...";
    //   KIOManager::GetInstance()->
    // 	Message("KTRobinHood_OpenCL",
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
    cl::Program program(KOpenCLInterface::GetInstance()->GetContext(), source, nullptr);

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"@Building the OpenCL Kernel (if the active GPU is running a display, communication may be temporarily interrupted)...@";
    //   KIOManager::GetInstance()->
    // 	Message("KTRobinHood_OpenCL",
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
        program.build(devices, GetOpenCLFlags().c_str());
    }
    catch (cl::Error& error) {
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout
            << "There was an error compiling Boundary Integral kernels.  Here is the information from the OpenCL C++ API:"
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
        // s<<"Build Status: "<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KIOManager::GetInstance()->GetDevice())<<"@@";
        // s<<"Build Options:\t"<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KIOManager::GetInstance()->GetDevice())<<"@@";
        // s<<"Build Log:\t "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KIOManager::GetInstance()->GetDevice());
        // KIOManager::GetInstance()->
        // 	Message("KTRobinHood_OpenCL",
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

    // Make kernel
    fGetMatrixElementKernel = new cl::Kernel(program, "GetMatrixElement");

    // define fNLocal
    if (fNLocal == -1)
        fNLocal = fGetMatrixElementKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
            (KOpenCLInterface::GetInstance()->GetDevice()));

    fData.SetMinimumWorkgroupSizeForKernels(fNLocal);

    // Create memory buffers
    fBufferIJ = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, 2 * sizeof(cl_int));
    fBufferValue = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_WRITE_ONLY, sizeof(CL_TYPE));
}

template<class BasisPolicy> void KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<BasisPolicy>>::AssignBuffers() const
{
    // Copy lists to the memory buffers
    fGetMatrixElementKernel->setArg(0, *fBufferIJ);
    fGetMatrixElementKernel->setArg(1, *fContainer.GetBoundaryInfo());
    fGetMatrixElementKernel->setArg(2, *fContainer.GetBoundaryData());
    fGetMatrixElementKernel->setArg(3, *fContainer.GetShapeInfo());
    fGetMatrixElementKernel->setArg(4, *fContainer.GetShapeData());
    fGetMatrixElementKernel->setArg(5, *fBufferValue);
}
}  // namespace KEMField

#endif /* KOPENCLBOUNDARYINTEGRALMATRIX_DEF */
