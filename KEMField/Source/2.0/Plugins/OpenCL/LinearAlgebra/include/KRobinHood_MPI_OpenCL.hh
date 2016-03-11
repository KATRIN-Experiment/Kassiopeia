#ifndef KROBINHOOD_MPI_OPENCL_DEF
#define KROBINHOOD_MPI_OPENCL_DEF

#include "KRobinHood.hh"

#include "KOpenCLAction.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"

#include "KMPIInterface.hh"

namespace KEMField
{
  template <typename ValueType>
  class KRobinHood_MPI_OpenCL : public KOpenCLAction
  {
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KRobinHood_MPI_OpenCL(const Matrix& A,Vector& x,const Vector& b);
    ~KRobinHood_MPI_OpenCL();

    void ConstructOpenCLKernels() const;
    void AssignBuffers() const;

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void CompleteResidualNormalization(double& residualNorm);
    void IdentifyLargestResidualElement();
    void ComputeCorrection();
    void UpdateSolutionApproximation();
    void UpdateVectorApproximation();
    void CoalesceData();
    void Finalize();

    unsigned int Dimension() const { return fB.Dimension(); }

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    double fBInfinityNorm;

    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    mutable cl::Kernel *fInitializeVectorApproximationKernel;
    mutable cl::Kernel *fFindResidualKernel;
    mutable cl::Kernel *fFindResidualNormKernel;
    mutable cl::Kernel *fCompleteResidualNormalizationKernel;
    mutable cl::Kernel *fIdentifyLargestResidualElementKernel;
    mutable cl::Kernel *fCompleteLargestResidualIdentificationKernel;
    mutable cl::Kernel *fComputeCorrectionKernel;
    mutable cl::Kernel *fUpdateSolutionApproximationKernel;
    mutable cl::Kernel *fUpdateVectorApproximationKernel;

    mutable cl::Buffer *fBufferResidual;
    mutable cl::Buffer *fBufferB_iterative;
    mutable cl::Buffer *fBufferCorrection;
    mutable cl::Buffer *fBufferPartialMaxResidualIndex;
    mutable cl::Buffer *fBufferMaxResidualIndex;
    mutable cl::Buffer fBufferMaxResidual;
    mutable cl::Buffer *fBufferPartialResidualNorm;
    mutable cl::Buffer *fBufferResidualNorm;
    mutable cl::Buffer *fBufferNWarps;
    mutable cl::Buffer *fBufferCounter;

    mutable cl::NDRange *fGlobalRange;
    mutable cl::NDRange *fLocalRange;
    mutable cl::NDRange *fGlobalRangeOffset;
    mutable cl::NDRange *fGlobalSize;
    mutable cl::NDRange *fGlobalMin;
    mutable cl::NDRange *fRangeOne;

    mutable CL_TYPE *fCLResidual;
    mutable CL_TYPE *fCLB_iterative;
    mutable CL_TYPE *fCLCorrection;
    mutable cl_int  *fCLPartialMaxResidualIndex;
    mutable cl_int  *fCLMaxResidualIndex;
    mutable CL_TYPE *fCLMaxResidual;
    mutable CL_TYPE *fCLPartialResidualNorm;
    mutable CL_TYPE *fCLResidualNorm;
    mutable cl_int  *fCLNWarps;
    mutable cl_int  *fCLCounter;

    mutable bool fReadResidual;

    MPI_Status   fStatus;
    MPI_Datatype fMPI_Res_type;
    MPI_Op       fMPI_Max;
    MPI_Op       fMPI_Min;

    typedef struct Res_Real_t {
      int    fIndex;
      double fRes;
      double fCorrection;
    } Res_Real;

    typedef struct Res_Complex_t {
      int    fIndex;
      double fRes;
      double fCorrection_real;
      double fCorrection_imag;
    } Res_Complex;

    Res_Real fRes_real;
    Res_Complex fRes_complex;

    static void MPIRealMax(Res_Real* in,
			   Res_Real* inout,
			   int* len,
			   MPI_Datatype*);
    static void MPIRealMin(Res_Real* in,
			   Res_Real* inout,
			   int* len,
			   MPI_Datatype*);
    static void MPIComplexMax(Res_Complex* in,
			      Res_Complex* inout,
			      int* len,
			      MPI_Datatype*);
    static void MPIComplexMin(Res_Complex* in,
			      Res_Complex* inout,
			      int* len,
			      MPI_Datatype*);

    void InitializeMPIStructs(Type2Type<double>);
    void InitializeMPIStructs(Type2Type<std::complex<double> >);
  };

  template <typename ValueType>
  KRobinHood_MPI_OpenCL<ValueType>::KRobinHood_MPI_OpenCL(const Matrix& A, Vector& X, const Vector& B) : KOpenCLAction(dynamic_cast<const KOpenCLAction&>(A).GetData()), fA(A), fX(X), fB(B), fBInfinityNorm(1.e30), fInitializeVectorApproximationKernel(NULL), fFindResidualKernel(NULL), fFindResidualNormKernel(NULL), fCompleteResidualNormalizationKernel(NULL), fIdentifyLargestResidualElementKernel(NULL), fCompleteLargestResidualIdentificationKernel(NULL), fComputeCorrectionKernel(NULL), fUpdateSolutionApproximationKernel(NULL), fUpdateVectorApproximationKernel(NULL), fBufferResidual(NULL), fBufferB_iterative(NULL), fBufferCorrection(NULL), fBufferPartialMaxResidualIndex(NULL), fBufferMaxResidualIndex(NULL), fBufferPartialResidualNorm(NULL), fBufferResidualNorm(NULL), fBufferNWarps(NULL), fBufferCounter(NULL), fCLResidual(NULL), fCLB_iterative(NULL), fCLCorrection(NULL), fCLPartialMaxResidualIndex(NULL), fCLMaxResidualIndex(NULL), fCLMaxResidual(NULL), fCLPartialResidualNorm(NULL), fCLResidualNorm(NULL), fCLNWarps(NULL), fCLCounter(NULL), fReadResidual(false)
  {
    InitializeMPIStructs(Type2Type<ValueType>());
    KOpenCLAction::Initialize();
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::ConstructOpenCLKernels() const
  {
    // Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_RobinHood_kernel.cl";

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"Reading source file "
    // 			     <<"\""<<clFile.str()<<"\"...";
    //   KOpenCLInterface::GetInstance()->
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

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"@Building the OpenCL Kernel (if the active GPU is running a display, communication may be temporarily interrupted)...@";
    //   KOpenCLInterface::GetInstance()->
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
      program.build(devices,dynamic_cast<const KOpenCLAction&>(fA).GetOpenCLFlags().c_str());
    }
    catch (cl::Error error)
    {
      std::cout<<"There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:"<<std::endl;
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
      // s<<"Build Status: "<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice())<<"@@";
      // s<<"Build Options:\t"<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice())<<"@@";
      // s<<"Build Log:\t "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice());
      // KOpenCLInterface::GetInstance()->
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

    // Make kernels
    fInitializeVectorApproximationKernel = new cl::Kernel(program, "InitializeVectorApproximation");
    fFindResidualKernel = new cl::Kernel(program, "FindResidual");
    fFindResidualNormKernel = new cl::Kernel(program, "FindResidualNorm");
    fCompleteResidualNormalizationKernel = new cl::Kernel(program, "CompleteResidualNormalization");
    fIdentifyLargestResidualElementKernel = new cl::Kernel(program, "IdentifyLargestResidualElement");
    fCompleteLargestResidualIdentificationKernel = new cl::Kernel(program, "CompleteLargestResidualIdentification");
    fComputeCorrectionKernel = new cl::Kernel(program, "ComputeCorrection");
    fUpdateSolutionApproximationKernel = new cl::Kernel(program, "UpdateSolutionApproximation");
    fUpdateVectorApproximationKernel = new cl::Kernel(program, "UpdateVectorApproximation");

    std::vector<cl::Kernel*> kernelArray;
    kernelArray.push_back(fInitializeVectorApproximationKernel);
    kernelArray.push_back(fFindResidualKernel);
    kernelArray.push_back(fFindResidualNormKernel);
    kernelArray.push_back(fCompleteResidualNormalizationKernel);
    kernelArray.push_back(fIdentifyLargestResidualElementKernel);
    kernelArray.push_back(fCompleteLargestResidualIdentificationKernel);
    kernelArray.push_back(fComputeCorrectionKernel);
    kernelArray.push_back(fUpdateSolutionApproximationKernel);
    kernelArray.push_back(fUpdateVectorApproximationKernel);

    fNLocal = UINT_MAX;

    for (std::vector<cl::Kernel*>::iterator it=kernelArray.begin();it!=kernelArray.end();++it)
    {
      unsigned int workgroupSize = (*it)->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>((KOpenCLInterface::GetInstance()->GetDevice()));
      if (workgroupSize < fNLocal) fNLocal = workgroupSize;
    }

    MPI_Allreduce(MPI_IN_PLACE, &fNLocal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    fNLocal *= KMPIInterface::GetInstance()->GetNProcesses();

    fData.SetMinimumWorkgroupSizeForKernels(fNLocal);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::AssignBuffers() const
  {
    if (fData.GetMinimumWorkgroupSizeForKernels()%KMPIInterface::GetInstance()->GetNProcesses()!=0)
      KEMField::cout<<"Number of streams does not evenly divide across processes!"<<KEMField::endl;

    fNLocal = fData.GetMinimumWorkgroupSizeForKernels();
    fNWorkgroups = fData.GetNBufferedElements()/fNLocal;
    fNLocal /= KMPIInterface::GetInstance()->GetNProcesses();

    KOpenCLSurfaceContainer& container = dynamic_cast<KOpenCLSurfaceContainer&>(fData);

    // Construct the ranges over which the OpenCL kernels will run
    fGlobalRange = new cl::NDRange(fData.GetNBufferedElements());
    fLocalRange = new cl::NDRange(fNLocal);
    fGlobalRangeOffset = new cl::NDRange(fNLocal*fNWorkgroups*KMPIInterface::GetInstance()->GetProcess());
    fGlobalSize = new cl::NDRange(fData.GetNBufferedElements()/KMPIInterface::GetInstance()->GetNProcesses());
    fGlobalMin = new cl::NDRange(fNLocal);
    fRangeOne = new cl::NDRange(1);

    fCLB_iterative = new CL_TYPE[fData.GetNBufferedElements()];

    fCLNWarps = new cl_int[1];
    fCLNWarps[0] = fNWorkgroups;
    fCLCounter = new cl_int[1];
    fCLCounter[0] = fData.GetNBufferedElements();

    fCLResidual = new CL_TYPE[fData.GetNBufferedElements()];
    fCLCorrection = new CL_TYPE[1];
    fCLPartialMaxResidualIndex = new cl_int[fNWorkgroups];
    fCLMaxResidualIndex = new cl_int[1];
    fCLMaxResidual = new CL_TYPE[1];
    fCLPartialResidualNorm = new CL_TYPE[fNWorkgroups];
    fCLResidualNorm = new CL_TYPE[1];

    for (unsigned int i=0;i<fData.GetNBufferedElements();i++)
      fCLResidual[i] = 0.;

    // set inactive elemnts to fCLB_iterative[i] > 1.e10;
    {
      unsigned int counter = 0;
      for (unsigned int i=0;i<container.NUniqueBoundaries();i++)
      {
	for (unsigned int j=0;j<container.size(i);j++)
	{
	  if (counter<fNLocal*fNWorkgroups*KMPIInterface::GetInstance()->GetProcess() || counter >=fNLocal*fNWorkgroups*(KMPIInterface::GetInstance()->GetProcess()+1))
	    fCLB_iterative[counter] = 1.e30;
	  else
	    fCLB_iterative[counter] = 0.;
	  counter++;
	}
      }

      for (;counter<fData.GetNBufferedElements();counter++)
	fCLB_iterative[counter] = 1.e30;
    }

    fCLCorrection[0] = 0.;
    fCLMaxResidualIndex[0] = -1;
    fCLResidualNorm[0] = 0.;

    // Create memory buffers
    fBufferResidual =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_WRITE,
		     container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE));
    fBufferB_iterative =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_WRITE,
		     container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE));
    fBufferCorrection =new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
				 CL_MEM_READ_WRITE,
				 sizeof(CL_TYPE));
    fBufferPartialMaxResidualIndex =new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
					CL_MEM_READ_WRITE,
					fNWorkgroups * sizeof(cl_int));
    fBufferMaxResidualIndex =new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
				 CL_MEM_READ_WRITE,
				 sizeof(cl_int));
  cl_buffer_region region;
  region.size = sizeof(CL_TYPE);
  region.origin = 0;
  fBufferMaxResidual =
    fBufferResidual->createSubBuffer(CL_MEM_READ_ONLY,
				     CL_BUFFER_CREATE_TYPE_REGION,
				     &region);
    fBufferPartialResidualNorm =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_WRITE,
		     fNWorkgroups * sizeof(CL_TYPE));
    fBufferResidualNorm =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_WRITE,
		     sizeof(CL_TYPE));
    fBufferNWarps =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_ONLY,
		     sizeof(cl_int));
    fBufferCounter =
      new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
		     CL_MEM_READ_WRITE,
		     sizeof(cl_int));

    // Copy lists to the memory buffers
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferResidual,
				    CL_TRUE,
				    0,
				    container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE),
				    fCLResidual);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferB_iterative,
				    CL_TRUE,
				    0,
				    container.GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE),
				    fCLB_iterative);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferCorrection,
				    CL_TRUE,
				    0,
				    sizeof(CL_TYPE),
				    fCLCorrection);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferPartialMaxResidualIndex,
				    CL_TRUE,
				    0,
				    fNWorkgroups * sizeof(cl_int),
				    fCLPartialMaxResidualIndex);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferMaxResidualIndex,
				    CL_TRUE,
				    0,
				    sizeof(cl_int),
				    fCLMaxResidualIndex);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferPartialResidualNorm,
				    CL_TRUE,
				    0,
				    fNWorkgroups * sizeof(CL_TYPE),
				    fCLPartialResidualNorm);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferResidualNorm,
				    CL_TRUE,
				    0,
				    sizeof(CL_TYPE),
				    fCLResidualNorm);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferNWarps,
				    CL_TRUE,
				    0,
				    sizeof(cl_int),
				    fCLNWarps);
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueWriteBuffer(*fBufferCounter,
				    CL_TRUE,
				    0,
				    sizeof(cl_int),
				    fCLCounter);

    fInitializeVectorApproximationKernel->setArg(0,*container.GetBoundaryInfo());
    fInitializeVectorApproximationKernel->setArg(1,*container.GetBoundaryData());
    fInitializeVectorApproximationKernel->setArg(2,*container.GetShapeInfo());
    fInitializeVectorApproximationKernel->setArg(3,*container.GetShapeData());
    fInitializeVectorApproximationKernel->setArg(4,*container.GetBasisData());
    fInitializeVectorApproximationKernel->setArg(5,*fBufferResidual);

    fFindResidualKernel->setArg(0,*fBufferResidual);
    fFindResidualKernel->setArg(1,*container.GetBoundaryInfo());
    fFindResidualKernel->setArg(2,*container.GetBoundaryData());
    fFindResidualKernel->setArg(3,*fBufferB_iterative);
    fFindResidualKernel->setArg(4,*fBufferCounter);

    fFindResidualNormKernel->setArg(0,*container.GetBoundaryInfo());
    fFindResidualNormKernel->setArg(1,*fBufferResidual);
    fFindResidualNormKernel->setArg(2,*fBufferResidualNorm);

    fCompleteResidualNormalizationKernel->setArg(0,*container.GetBoundaryInfo());
    fCompleteResidualNormalizationKernel->setArg(1,*container.GetBoundaryData());
    fCompleteResidualNormalizationKernel->setArg(2,*fBufferResidualNorm);

    fIdentifyLargestResidualElementKernel->setArg(0,*fBufferResidual);
    fIdentifyLargestResidualElementKernel->setArg(1,fNLocal * sizeof(cl_int), NULL);
    fIdentifyLargestResidualElementKernel->setArg(2,*fBufferPartialMaxResidualIndex);

    fCompleteLargestResidualIdentificationKernel->setArg(0,*fBufferResidual);
    fCompleteLargestResidualIdentificationKernel->setArg(1,*container.GetBoundaryInfo());
    fCompleteLargestResidualIdentificationKernel->setArg(2,*fBufferPartialMaxResidualIndex);
    fCompleteLargestResidualIdentificationKernel->setArg(3,*fBufferMaxResidualIndex);
    fCompleteLargestResidualIdentificationKernel->setArg(4,*fBufferNWarps);

    fComputeCorrectionKernel->setArg(0,*container.GetShapeInfo());
    fComputeCorrectionKernel->setArg(1,*container.GetShapeData());
    fComputeCorrectionKernel->setArg(2,*container.GetBoundaryInfo());
    fComputeCorrectionKernel->setArg(3,*container.GetBoundaryData());
    fComputeCorrectionKernel->setArg(4,*container.GetBasisData());
    fComputeCorrectionKernel->setArg(5,*fBufferB_iterative);
    fComputeCorrectionKernel->setArg(6,*fBufferCorrection);
    fComputeCorrectionKernel->setArg(7,*fBufferMaxResidualIndex);
    fComputeCorrectionKernel->setArg(8,*fBufferCounter);

    fUpdateSolutionApproximationKernel->setArg(0,*container.GetBasisData());
    fUpdateSolutionApproximationKernel->setArg(1,*fBufferCorrection);
    fUpdateSolutionApproximationKernel->setArg(2,*fBufferMaxResidualIndex);

    fUpdateVectorApproximationKernel->setArg(0,*container.GetShapeInfo());
    fUpdateVectorApproximationKernel->setArg(1,*container.GetShapeData());
    fUpdateVectorApproximationKernel->setArg(2,*container.GetBoundaryInfo());
    fUpdateVectorApproximationKernel->setArg(3,*container.GetBoundaryData());
    fUpdateVectorApproximationKernel->setArg(4,*fBufferB_iterative);
    fUpdateVectorApproximationKernel->setArg(5,*fBufferCorrection);
    fUpdateVectorApproximationKernel->setArg(6,*fBufferMaxResidualIndex);

    // if (fVerbose>1 && fRank == 0)
    // {
    //   std::stringstream s;  s<<"@Done.";
    //   KOpenCLInterface::GetInstance()->
    // 	Message("KTRobinHood_OpenCL",
    // 		"InitializeOpenCLPrimitives",
    // 		s.str(),
    // 		0,
    // 		3);
    // }
  }

  template <typename ValueType>
  KRobinHood_MPI_OpenCL<ValueType>::~KRobinHood_MPI_OpenCL()
  {
    if (fMPI_Max != MPI_OP_NULL)
      MPI_Op_free(&fMPI_Max);
    if (fMPI_Min != MPI_OP_NULL)
      MPI_Op_free(&fMPI_Min);

    if (fInitializeVectorApproximationKernel) delete fInitializeVectorApproximationKernel;
    if (fFindResidualKernel) delete fFindResidualKernel;
    if (fFindResidualNormKernel) delete fFindResidualNormKernel;
    if (fCompleteResidualNormalizationKernel) delete fCompleteResidualNormalizationKernel;
    if (fIdentifyLargestResidualElementKernel) delete fIdentifyLargestResidualElementKernel;
    if (fCompleteLargestResidualIdentificationKernel) delete fCompleteLargestResidualIdentificationKernel;
    if (fComputeCorrectionKernel) delete fComputeCorrectionKernel;
    if (fUpdateSolutionApproximationKernel) delete fUpdateSolutionApproximationKernel;
    if (fUpdateVectorApproximationKernel) delete fUpdateVectorApproximationKernel;

    if (fBufferResidual) delete fBufferResidual;
    if (fBufferB_iterative) delete fBufferB_iterative;
    if (fBufferCorrection) delete fBufferCorrection;
    if (fBufferPartialMaxResidualIndex) delete fBufferPartialMaxResidualIndex;
    if (fBufferMaxResidualIndex) delete fBufferMaxResidualIndex;
    if (fBufferPartialResidualNorm) delete fBufferPartialResidualNorm;
    if (fBufferResidualNorm) delete fBufferResidualNorm;
    if (fBufferNWarps) delete fBufferNWarps;
    if (fBufferCounter) delete fBufferCounter;

    if (fGlobalRange) delete fGlobalRange;
    if (fLocalRange) delete fLocalRange;
    if (fGlobalRangeOffset) delete fGlobalRangeOffset;
    if (fGlobalSize) delete fGlobalSize;
    if (fGlobalMin) delete fGlobalMin;
    if (fRangeOne) delete fRangeOne;

    if (fCLResidual) delete fCLResidual;
    if (fCLB_iterative) delete fCLB_iterative;
    if (fCLCorrection) delete fCLCorrection;
    if (fCLPartialMaxResidualIndex) delete fCLPartialMaxResidualIndex;
    if (fCLMaxResidualIndex) delete fCLMaxResidualIndex;
    if (fCLMaxResidual) delete fCLMaxResidual;
    if (fCLPartialResidualNorm) delete fCLPartialResidualNorm;
    if (fCLResidualNorm) delete fCLResidualNorm;
    if (fCLNWarps) delete fCLNWarps;
    if (fCLCounter) delete fCLCounter;
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::Initialize()
  {
    if (!fReadResidual)
    {
      if (fX.InfinityNorm()>1.e-16)
      {
	cl::Event event;
	KOpenCLInterface::GetInstance()->
	  GetQueue().enqueueNDRangeKernel(*fInitializeVectorApproximationKernel,
					  *fGlobalRangeOffset,
					  *fGlobalSize,
					  *fLocalRange,
					  NULL,
					  &event);
	event.wait();
      }
    }
    KMPIInterface::GetInstance()->GlobalBarrier();
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::FindResidual()
  {
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueNDRangeKernel(*fFindResidualKernel,
				      *fGlobalRangeOffset,
				      *fGlobalSize,
				      *fLocalRange);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::FindResidualNorm(double& residualNorm)
  {
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueNDRangeKernel(*fFindResidualNormKernel,
				      cl::NullRange,
				      *fRangeOne,
				      *fRangeOne);
    CompleteResidualNormalization(residualNorm);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::CompleteResidualNormalization(double& residualNorm)
  {
    cl::Event event;
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueReadBuffer(*fBufferResidualNorm,
				   CL_TRUE,
				   0,
				   sizeof(CL_TYPE),
				   fCLResidualNorm,
				   NULL,
				   &event);
    event.wait();

    residualNorm = fCLResidualNorm[0];

    MPI_Allreduce(MPI_IN_PLACE,&residualNorm,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    if (fBInfinityNorm > 1.e10)
      fBInfinityNorm = fB.InfinityNorm();

    residualNorm /= fBInfinityNorm;
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::IdentifyLargestResidualElement()
  {
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueNDRangeKernel(*fIdentifyLargestResidualElementKernel,
    				      *fGlobalRangeOffset,
    				      *fGlobalSize,
    				      *fLocalRange);

    {
      cl::Event event;
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueNDRangeKernel(*fCompleteLargestResidualIdentificationKernel,
				      cl::NullRange,
				      *fRangeOne,
				      *fRangeOne,
				      NULL,
				      &event);
      event.wait();
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::ComputeCorrection()
  {
    {
      cl::Event event;
      KOpenCLInterface::GetInstance()->
	GetQueue().enqueueNDRangeKernel(*fComputeCorrectionKernel,
					cl::NullRange,
					*fRangeOne,
					*fRangeOne,
					NULL,
					&event);
      event.wait();
    }

    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueReadBuffer(*fBufferMaxResidualIndex,
				   CL_TRUE,
				   0,
				   sizeof(cl_int),
				   fCLMaxResidualIndex);

    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueReadBuffer(*fBufferCorrection,
				   CL_TRUE,
				   0,
				   sizeof(CL_TYPE),
				   fCLCorrection);

      {
	cl::Event event;
	KOpenCLInterface::GetInstance()->
	  GetQueue().enqueueReadBuffer(fBufferMaxResidual,
				       CL_TRUE,
				       0,
				       sizeof(CL_TYPE),
				       fCLMaxResidual,
				       NULL,
				       &event);
	event.wait();
      }

      fRes_real.fIndex = fCLMaxResidualIndex[0];
      fRes_real.fRes = fabs(fCLMaxResidual[0]);
      fRes_real.fCorrection = fCLCorrection[0];

      MPI_Allreduce(MPI_IN_PLACE,&fRes_real,1,fMPI_Res_type,fMPI_Max,MPI_COMM_WORLD);

      fCLMaxResidualIndex[0] = fRes_real.fIndex;
      fCLCorrection[0] = fRes_real.fCorrection;

      KOpenCLInterface::GetInstance()->
	GetQueue().enqueueWriteBuffer(*fBufferMaxResidualIndex,
				      CL_TRUE,
				      0,
				      sizeof(cl_int),
				      fCLMaxResidualIndex);

      {
	cl::Event event;
	KOpenCLInterface::GetInstance()->
	  GetQueue().enqueueWriteBuffer(*fBufferCorrection,
					CL_TRUE,
					0,
					sizeof(CL_TYPE),
					fCLCorrection,
					NULL,
					&event);
	event.wait();
      }
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::UpdateSolutionApproximation()
  {
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueNDRangeKernel(*fUpdateSolutionApproximationKernel,
				      cl::NullRange,
				      *fRangeOne,
				      *fRangeOne);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::UpdateVectorApproximation()
  {
    KOpenCLInterface::GetInstance()->
      GetQueue().enqueueNDRangeKernel(*fUpdateVectorApproximationKernel,
				      *fGlobalRangeOffset,
				      *fGlobalSize,
				      *fLocalRange);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::CoalesceData()
  {
    dynamic_cast<KOpenCLSurfaceContainer&>(fData).ReadBasisData();
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::Finalize()
  {
    dynamic_cast<KOpenCLSurfaceContainer&>(fData).ReadBasisData();
    KMPIInterface::GetInstance()->GlobalBarrier();
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::SetResidualVector(const Vector& v)
  {
    fReadResidual = true;

    for (unsigned int i = 0;i<v.Dimension();i++)
    {
      fCLResidual[i] = v(i);
      fCLB_iterative[i] = fB(i) - fCLResidual[i];
    }

    // set inactive elemnts to fCLB_iterative[i] > 1.e10;
    {
      unsigned int counter = 0;
      for (unsigned int i=0;i<dynamic_cast<KOpenCLSurfaceContainer&>(fData).NUniqueBoundaries();i++)
      {
	for (unsigned int j=0;j<dynamic_cast<KOpenCLSurfaceContainer&>(fData).size(i);j++)
	{
	  if (counter<fNLocal*fNWorkgroups*KMPIInterface::GetInstance()->GetProcess() || counter >=fNLocal*fNWorkgroups*(KMPIInterface::GetInstance()->GetProcess()+1))
	  {
	    fCLResidual[counter] = 0.;
	    fCLB_iterative[counter] = 1.e30;
	  }
	  counter++;
	}
      }

      for (;counter<fData.GetNBufferedElements();counter++)
      {
	fCLResidual[counter] = 0.;
	fCLB_iterative[counter] = 1.e30;
      }
    }

    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueWriteBuffer(*fBufferResidual,
			 CL_TRUE,
			 0,
			 dynamic_cast<KOpenCLSurfaceContainer&>(fData).GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE),
			 fCLResidual);

    cl::Event event;
    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueWriteBuffer(*fBufferB_iterative,
			 CL_TRUE,
			 0,
			 dynamic_cast<KOpenCLSurfaceContainer&>(fData).GetBoundarySize() * fData.GetNBufferedElements() * sizeof(CL_TYPE),
			 fCLB_iterative,
			 NULL,
			 &event);
    event.wait();
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::GetResidualVector(Vector& v) const
  {
    cl::Event event;
    KOpenCLInterface::GetInstance()->GetQueue().
      enqueueReadBuffer(*fBufferResidual,
			CL_TRUE,
			0,
			fData.GetNBufferedElements()*sizeof(CL_TYPE),
			fCLResidual,
			NULL,
			&event);
    event.wait();

    MPI_Reduce(&fCLResidual[0], &v[0], Dimension(), MPI_DOUBLE, MPI_MAX, 0,MPI_COMM_WORLD);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::MPIRealMax(Res_Real* in,
						    Res_Real* inout,
						    int* len,
						    MPI_Datatype*)
  {
    int i;

    for (i=0; i< *len; ++i)
    {
      if (in->fRes > inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex = in->fIndex;
	inout->fRes  = in->fRes;
	inout->fCorrection  = in->fCorrection;
      }
      in++; inout++;
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::MPIRealMin(Res_Real* in,
						    Res_Real* inout,
						    int* len,
						    MPI_Datatype*)
  {
    int i;

    for (i=0; i< *len; ++i)
    {
      if (in->fRes < inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex = in->fIndex;
	inout->fRes  = in->fRes;
	inout->fCorrection  = in->fCorrection;
      }
      in++; inout++;
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::MPIComplexMax(Res_Complex* in,
						       Res_Complex* inout,
						       int* len,
						       MPI_Datatype*)
  {
    int i;

    for (i=0; i< *len; ++i)
    {
      if (in->fRes > inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex  = in->fIndex;
	inout->fRes = in->fRes;
	inout->fCorrection_real = in->fCorrection_real;
	inout->fCorrection_imag = in->fCorrection_imag;
      }
      in++; inout++;
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::MPIComplexMin(Res_Complex* in,
						       Res_Complex* inout,
						       int* len,
						       MPI_Datatype*)
  {
    int i;

    for (i=0; i< *len; ++i)
    {
      if (in->fRes < inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex  = in->fIndex;
	inout->fRes = in->fRes;
	inout->fCorrection_real = in->fCorrection_real;
	inout->fCorrection_imag = in->fCorrection_imag;
      }
      in++; inout++;
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::InitializeMPIStructs(Type2Type<double>)
  {
    int block_lengths[3] = {1,1,1};
    MPI_Aint displacements[3];
    MPI_Aint addresses[4];
    MPI_Datatype typelist[3] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE};

    MPI_Address(&fRes_real,&addresses[0]);
    MPI_Address(&(fRes_real.fIndex),&addresses[1]);
    MPI_Address(&(fRes_real.fRes),&addresses[2]);
    MPI_Address(&(fRes_real.fCorrection),&addresses[3]);

    displacements[0] = addresses[1] - addresses[0];
    displacements[1] = addresses[2] - addresses[0];
    displacements[2] = addresses[3] - addresses[0];

    MPI_Type_create_struct(3,
  			   block_lengths,
  			   displacements,
  			   typelist,
  			   &fMPI_Res_type);

    MPI_Type_commit(&fMPI_Res_type);

    MPI_Op_create((MPI_User_function *)MPIRealMax,true,&fMPI_Max);
    MPI_Op_create((MPI_User_function *)MPIRealMin,true,&fMPI_Min);
  }

  template <typename ValueType>
  void KRobinHood_MPI_OpenCL<ValueType>::InitializeMPIStructs(Type2Type<std::complex<double> >)
  {
    int block_lengths[4] = {1,1,1,1};
    MPI_Aint displacements[4];
    MPI_Aint addresses[5];
    MPI_Datatype typelist[4] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};

    MPI_Address(&fRes_complex,&addresses[0]);
    MPI_Address(&(fRes_complex.fIndex),&addresses[1]);
    MPI_Address(&(fRes_complex.fRes),&addresses[2]);
    MPI_Address(&(fRes_complex.fCorrection_real),&addresses[3]);
    MPI_Address(&(fRes_complex.fCorrection_imag),&addresses[4]);

    displacements[0] = addresses[1] - addresses[0];
    displacements[1] = addresses[2] - addresses[0];
    displacements[2] = addresses[3] - addresses[0];
    displacements[3] = addresses[4] - addresses[0];

    MPI_Type_create_struct(4,
  			   block_lengths,
  			   displacements,
  			   typelist,
  			   &fMPI_Res_type);

    MPI_Type_commit(&fMPI_Res_type);

    MPI_Op_create((MPI_User_function *)MPIComplexMax,true,&fMPI_Max);
    MPI_Op_create((MPI_User_function *)MPIComplexMin,true,&fMPI_Min);
  }
}

#endif /* KROBINHOOD_MPI_OPENCL_DEF */
