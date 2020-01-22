#ifndef KOPENCLBOUNDARYINTEGRALSOLUTIONVECTOR_DEF
#define KOPENCLBOUNDARYINTEGRALSOLUTIONVECTOR_DEF

#include "KOpenCLAction.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLBoundaryIntegrator.hh"

namespace KEMField
{
  template <class BasisPolicy>
  class KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> > : public KVector<typename BasisPolicy::ValueType>, public KOpenCLAction
  {
  public:
    typedef typename BasisPolicy::ValueType ValueType;
    friend class KOpenCLSurfaceContainer;

    KBoundaryIntegralSolutionVector(KOpenCLSurfaceContainer& c,
				    KOpenCLBoundaryIntegrator<BasisPolicy>& integrator);

    ~KBoundaryIntegralSolutionVector();

    unsigned int Dimension() const { return fDimension; }

    const ValueType& operator()(unsigned int i) const;
    // Currently, this method does not return a modifiable value
    ValueType& operator[](unsigned int i);

    const ValueType& InfinityNorm() const;

    void SetNLocal(int nLocal) const { fNLocal = nLocal; }

    int GetNLocal() const { return fNLocal; }

  private:
    KOpenCLSurfaceContainer& fContainer;
    KOpenCLBoundaryIntegrator<BasisPolicy>& fIntegrator;

    const unsigned int fDimension;

    void ConstructOpenCLKernels() const;
    void AssignBuffers() const;

    mutable int fNLocal;

    mutable cl::Kernel *fGetSolutionVectorElementKernel;
    mutable cl::Kernel *fGetMaximumSolutionVectorElementKernel;

    mutable cl::Buffer *fBufferI;
    mutable cl::Buffer *fBufferValue;

    mutable ValueType fValue;
  };

  template <class BasisPolicy>
  KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::KBoundaryIntegralSolutionVector(KOpenCLSurfaceContainer& c,KOpenCLBoundaryIntegrator<BasisPolicy>& integrator) :
    KVector<ValueType>(),
    KOpenCLAction(c),
    fContainer(c),
    fIntegrator(integrator),
    // TO DO: add mult. factors
    fDimension(c.size()),
    fNLocal(-1),
    fGetSolutionVectorElementKernel(NULL),
    fGetMaximumSolutionVectorElementKernel(NULL),
    fBufferI(NULL),
    fBufferValue(NULL)
  {

  }

  template <class BasisPolicy>
  KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::~KBoundaryIntegralSolutionVector()
  {
    if (fGetSolutionVectorElementKernel) delete fGetSolutionVectorElementKernel;
    if (fGetMaximumSolutionVectorElementKernel)
      delete fGetMaximumSolutionVectorElementKernel;

    if (fBufferI) delete fBufferI;
    if (fBufferValue) delete fBufferValue;
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::operator()(unsigned int i) const
  {
    cl_int i_[1];
    i_[0] = static_cast<int>(i);

    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueWriteBuffer(*fBufferI,
			 CL_TRUE,
			 0,
			 sizeof(cl_int),
			 i_);

    cl::NDRange global(1);
    cl::NDRange local(1);

    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueNDRangeKernel(*fGetSolutionVectorElementKernel,
			   cl::NullRange,
			   global,
			   local);

    CL_TYPE value;
    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueReadBuffer(*fBufferValue,
			CL_TRUE,
			0,
			sizeof(CL_TYPE),
			&value);
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  typename BasisPolicy::ValueType& KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::operator[](unsigned int i)
  {
    cl_int i_[1];
    i_[0] = static_cast<int>(i);

    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueWriteBuffer(*fBufferI,
			 CL_TRUE,
			 0,
			 sizeof(cl_int),
			 i_);

    cl::NDRange global(1);
    cl::NDRange local(1);

    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueNDRangeKernel(*fGetSolutionVectorElementKernel,
			   cl::NullRange,
			   global,
			   local);

    CL_TYPE value;
    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueReadBuffer(*fBufferValue,
			CL_TRUE,
			0,
			sizeof(CL_TYPE),
			&value);
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  const typename BasisPolicy::ValueType& KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::InfinityNorm() const
  {
    cl::NDRange global(1);
    cl::NDRange local(1);

    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueNDRangeKernel(*fGetMaximumSolutionVectorElementKernel,
			   cl::NullRange,
			   global,
			   local);

    CL_TYPE value;
    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueReadBuffer(*fBufferValue,
			CL_TRUE,
			0,
			sizeof(CL_TYPE),
			&value);
    fValue = value;
    return fValue;
  }

  template <class BasisPolicy>
  void KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::ConstructOpenCLKernels() const
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

    sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile),
			     (std::istreambuf_iterator<char>()));

    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(),
						  sourceCode.length()+1));

    // Make program of the source code in the context
    cl::Program program(KOpenCLInterface::GetInstance()->GetContext(),source,0);

    // Define some options to for building
    std::stringstream options;
    options << fIntegrator.GetOpenCLFlags();
    options << fContainer.GetOpenCLFlags();

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
    try
    {
      // use only target device!
      CL_VECTOR_TYPE<cl::Device> devices;
      devices.push_back( KOpenCLInterface::GetInstance()->GetDevice() );
      program.build(devices,
		    options.str().c_str());
    }
    catch (cl::Error &error)
    {
      std::cout<<__FILE__<<":"<<__LINE__<<std::endl;
      std::cout<<"There was an error compiling the solution vector kernel.  Here is the information from the OpenCL C++ API:"<<std::endl;
      std::cout<<error.what()<<"("<<error.err()<<")"<<std::endl;
      std::cout<<"Build Status: "<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice())<<""<<std::endl;
      std::cout<<"Build Options:\t"<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice())<<""<<std::endl;
      std::cout<<"Build Log:\t "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice());

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
    s << "Build Log for OpenCL "<<clFile.str()<<" :\t ";
    std::stringstream build_log_stream;
    build_log_stream<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
    std::string build_log;
    build_log = build_log_stream.str();
    if(build_log.size() != 0)
    {
        s << build_log;
        std::cout<<s.str()<<std::endl;
    }
    #endif

    // Make kernel
    fGetSolutionVectorElementKernel =
      new cl::Kernel(program, "GetSolutionVectorElement");
    fGetMaximumSolutionVectorElementKernel =
      new cl::Kernel(program, "GetMaximumSolutionVectorElement");

    // define fNLocal
    if (fNLocal == -1)
      fNLocal = fGetSolutionVectorElementKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice()));

    // Create memory buffers
    fBufferI =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_ONLY,
		     sizeof(cl_int));
    fBufferValue =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_WRITE_ONLY,
		     sizeof(CL_TYPE));
  }

  template <class BasisPolicy>
  void KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<BasisPolicy> >::AssignBuffers() const
  {
    // Copy lists to the memory buffers
    fGetSolutionVectorElementKernel->setArg(0,*fBufferI);
    fGetSolutionVectorElementKernel->setArg(1,*fContainer.GetBasisData());
    fGetSolutionVectorElementKernel->setArg(2,*fBufferValue);

    fGetMaximumSolutionVectorElementKernel->setArg(0,*fContainer.GetBoundaryInfo());
    fGetMaximumSolutionVectorElementKernel->setArg(1,*fContainer.GetBasisData());
    fGetMaximumSolutionVectorElementKernel->setArg(2,*fBufferValue);
  }
}

#endif /* KOPENCLBOUNDARYINTEGRALSOLUTIONVECTOR_DEF */
