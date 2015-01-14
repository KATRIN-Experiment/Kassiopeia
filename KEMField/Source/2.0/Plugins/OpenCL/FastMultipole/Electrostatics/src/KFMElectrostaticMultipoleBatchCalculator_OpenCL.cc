#include "KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh"
#include "KFMMessaging.hh"

#include "KFMPinchonJMatrixCalculator.hh"

#include "KFMPointCloud.hh"
#include "KFMBasisData.hh"

#include <fstream>


namespace KEMField
{


KFMElectrostaticMultipoleBatchCalculator_OpenCL::KFMElectrostaticMultipoleBatchCalculator_OpenCL():
KFMElectrostaticMultipoleBatchCalculatorBase(),
fJSize(0),
fOpenCLFlags(""),
fJMatrix(NULL),
fAxialPlm(NULL),
fEquatorialPlm(NULL),
fBasisData(NULL),
fIntermediateOriginData(NULL),
fVertexData(NULL),
fIntermediateMomentData(NULL),
fOriginBufferCL(NULL),
fVertexDataBufferCL(NULL),
fBasisDataBufferCL(NULL),
fMomentBufferCL(NULL),
fACoefficientBufferCL(NULL),
fEquatorialPlmBufferCL(NULL),
fAxialPlmBufferCL(NULL),
fJMatrixBufferCL(NULL)
{
    fAnalyticCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();
};


KFMElectrostaticMultipoleBatchCalculator_OpenCL::~KFMElectrostaticMultipoleBatchCalculator_OpenCL()
{
    delete[] fJMatrix;
    delete[] fAxialPlm;
    delete[] fEquatorialPlm;
    delete[] fBasisData;
    delete[] fIntermediateOriginData;
    delete[] fVertexData;
    delete[] fIntermediateMomentData;

    delete fOriginBufferCL;
    delete fVertexDataBufferCL;
    delete fBasisDataBufferCL;
    delete fMomentBufferCL;
    delete fACoefficientBufferCL;
    delete fEquatorialPlmBufferCL;
    delete fAxialPlmBufferCL;
    delete fJMatrixBufferCL;

    delete fAnalyticCalc;
}

void KFMElectrostaticMultipoleBatchCalculator_OpenCL::SetDegree(int degree)
{
    if(!fInitialized)
    {
        fDegree = std::fabs(degree);
        fStride = (fDegree+1)*(fDegree+2)/2;
        fComplexStride = (fDegree+1)*(fDegree + 1); //scratch space stride

        //create the build flags
        std::stringstream ss;
        ss << " -D KFM_DEGREE=" << fDegree;
        ss << " -D KFM_COMPLEX_STRIDE=" << fComplexStride;
        ss << " -D KFM_REAL_STRIDE=" << fStride;
        ss << " -I " <<KOpenCLInterface::GetInstance()->GetKernelPath();
        fOpenCLFlags = ss.str();

        fAnalyticCalc->SetDegree(fDegree);
    }
}

void KFMElectrostaticMultipoleBatchCalculator_OpenCL::Initialize()
{

    if(!fInitialized)
    {
        //construct the kernel and determine the number of work group size
        ConstructOpenCLKernels();

        //now lets figure out how many elements we can process at a time
        unsigned int bytes_per_element = fStride*sizeof(CL_TYPE2);
        unsigned int buff_size = fMaxBufferSizeInBytes;
        unsigned int max_items = buff_size/bytes_per_element;

        if(max_items > fNLocal)
        {
            unsigned int mod = max_items%fNLocal;
            max_items -= mod;
        }
        else
        {
            max_items = fNLocal;
        }

        //compute new buffer size, and number of items we can process
        buff_size = bytes_per_element*max_items;
        fMaxBufferSizeInBytes = buff_size;
        fNMaxItems = max_items;
        fNMaxWorkgroups = fNMaxItems/fNLocal;
        fValidSize = fNMaxItems;

        if(fNMaxItems != 0)
        {
            fIDBuffer = new int[fNMaxItems];
            fMomentBuffer = new double[2*fStride*fNMaxItems];
            fOriginBuffer = new double[3*fNMaxItems];
        }
        else
        {
            //warning & abort
            std::stringstream ss;
            ss << "Buffer size of ";
            ss << fMaxBufferSizeInBytes;
            ss <<" bytes is not large enough for a single element. ";
            ss <<"Required bytes per element = "<<bytes_per_element<<". Aborting.";
            kfmout<<ss.str()<<std::endl;
            kfmexit(1);
        }

        BuildBuffers();

        AssignBuffers();

        fInitialized = true;
    }
}


void
KFMElectrostaticMultipoleBatchCalculator_OpenCL::ConstructOpenCLKernels()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticMultipole_kernel.cl";

    //read in the source code
    std::string sourceCode;
    std::ifstream sourceFile(clFile.str().c_str());
    sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    //Make program of the source code in the context
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
    cl::Program program(KOpenCLInterface::GetInstance()->GetContext(), source);

    //set the build options
    std::stringstream options;
    options << GetOpenCLFlags();

    // Build program for these specific devices
    try
    {
        // use only target device!
        cl::vector<cl::Device> devices;
        devices.push_back( KOpenCLInterface::GetInstance()->GetDevice() );
        program.build(devices,options.str().c_str());
    }
    catch (cl::Error error)
    {
        std::cout<<__FILE__<<":"<<__LINE__<<std::endl;
        std::stringstream s;
        s<<"There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:"<<std::endl;
        s<<error.what()<<"("<<error.err()<<")"<<std::endl;
        s<<"Build Status: "<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
        s<<"Build Options:\t"<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
        s<<"Build Log:\t "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
        std::cout<<s.str()<<std::endl;
        std::exit(1);
    }

    // Make kernel
	cl_int err_code = -1;
    try
    {
        fMultipoleKernel = new cl::Kernel(program, "ElectrostaticMultipole", &err_code);
    }
    catch(cl::Error error)
    {
        std::cout<<"Kernel construction failed with error code: "<<err_code<<std::endl;
        std::exit(1);
    }

    fNLocal = fMultipoleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
}

void
KFMElectrostaticMultipoleBatchCalculator_OpenCL::BuildBuffers()
{
    //origin data
    fIntermediateOriginData = new CL_TYPE4[fNMaxItems];

    //vertex data for elements to process
    fVertexData = new CL_TYPE16[fNMaxItems];

    //basis data for elements to process
    fBasisData = new CL_TYPE[fNMaxItems];

    //moment data
    fIntermediateMomentData = new CL_TYPE2[fStride*fNMaxItems];

    //compute the equatorial associated legendre polynomial array
    fEquatorialPlm = new CL_TYPE[fStride];
    KFMMath::ALP_nm_array(fDegree, 0.0, (double*)(fEquatorialPlm));

    //compute the axial associated legendre polynomial array
    fAxialPlm = new CL_TYPE[fStride];
    KFMMath::ALP_nm_array(fDegree, 1.0, (double*)(fAxialPlm));

    fACoefficient = new CL_TYPE[fStride];
    //compute the A coefficients
    int si;
    for(int n=0; n <=fDegree; n++)
    {
        for(int m=0; m <=n; m++)
        {
            si =  KFMScalarMultipoleExpansion::RealBasisIndex(n,m);
            fACoefficient[si] = KFMMath::A_Coefficient(m, n);
        }
    }

    //compute the pinchon j-matrices
    KFMPinchonJMatrixCalculator j_matrix_calc;
    std::vector< kfm_matrix* > j_matrix_vector;

    j_matrix_calc.SetDegree(fDegree);
    j_matrix_calc.AllocateMatrices(&j_matrix_vector);
    j_matrix_calc.ComputeMatrices(&j_matrix_vector);

    //figure out the size of the array we need:
    fJSize = 0;
    for(int i=0; i<=fDegree; i++)
    {
        fJSize += (2*i +1)*(2*i + 1);
    }
    fJMatrix = new CL_TYPE[fJSize];

    //loop over array of j matrices and push their data into the array
    int j_size = 0;
    int current_size;
    for(int l=0; l <= fDegree; l++)
    {
        current_size = 2*l+1;

        for(int row=0; row < current_size; row++)
        {
            for(int col=0; col < current_size; col++)
            {
                fJMatrix[j_size + row*current_size + col] = kfm_matrix_get(j_matrix_vector.at(l), row, col);
            }

        }
        j_size += (2*l +1)*(2*l + 1);
    }

    j_matrix_calc.DeallocateMatrices(&j_matrix_vector);


    //create buffers for the constant objects
    fACoefficientBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fStride*sizeof(CL_TYPE));

    fEquatorialPlmBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fStride*sizeof(CL_TYPE));

    fAxialPlmBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fStride*sizeof(CL_TYPE));

    fJMatrixBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fJSize*sizeof(CL_TYPE));


    //create buffers for the non-constant objects (must be enqueued writen/read on each call)
    fOriginBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(CL_TYPE4));

    fVertexDataBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(CL_TYPE16));

    fBasisDataBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(CL_TYPE));

    fMomentBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_WRITE_ONLY, fStride*fNMaxItems*sizeof(CL_TYPE2));
}


void
KFMElectrostaticMultipoleBatchCalculator_OpenCL::AssignBuffers()
{
    // Set arguments to kernel
    fMultipoleKernel->setArg(0, fNMaxItems); //must be set to (fValidSize) on each call

    //set constant arguments/buffers
    fMultipoleKernel->setArg(1, *fACoefficientBufferCL);
    fMultipoleKernel->setArg(2, *fEquatorialPlmBufferCL);
    fMultipoleKernel->setArg(3, *fAxialPlmBufferCL);
    fMultipoleKernel->setArg(4, *fJMatrixBufferCL);

    //set non-const arguments/buffers
    fMultipoleKernel->setArg(5, *fOriginBufferCL);
    fMultipoleKernel->setArg(6, *fVertexDataBufferCL);
    fMultipoleKernel->setArg(7, *fBasisDataBufferCL);
    fMultipoleKernel->setArg(8, *fMomentBufferCL);

    //copy const data to the GPU
    cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();
    Q.enqueueWriteBuffer(*fACoefficientBufferCL, CL_TRUE, 0, fStride*sizeof(CL_TYPE), fACoefficient);
    Q.enqueueWriteBuffer(*fEquatorialPlmBufferCL, CL_TRUE, 0, fStride*sizeof(CL_TYPE), fEquatorialPlm);
    Q.enqueueWriteBuffer(*fAxialPlmBufferCL, CL_TRUE, 0, fStride*sizeof(CL_TYPE), fAxialPlm);
    Q.enqueueWriteBuffer(*fJMatrixBufferCL, CL_TRUE, 0, fJSize*sizeof(CL_TYPE), fJMatrix);
}


void
KFMElectrostaticMultipoleBatchCalculator_OpenCL::FillTemporaryBuffers()
{
    for(size_t i=0; i<fValidSize; i++)
    {
        fIntermediateOriginData[i].s0 = fOriginBuffer[3*i];
        fIntermediateOriginData[i].s1 = fOriginBuffer[3*i + 1];
        fIntermediateOriginData[i].s2 = fOriginBuffer[3*i + 2];

        const KFMBasisData<1>* basis = fContainer->GetBasisData(fIDBuffer[i]);
        fBasisData[i] = (*basis)[0];

        const KFMPointCloud<3>* vertex_cloud = fContainer->GetPointCloud(fIDBuffer[i]);
        unsigned int cloud_size = vertex_cloud->GetNPoints();
        KFMPoint<3> vertex;

        //pattern indicates the number of valid vertices
        int msb, lsb;
        if(cloud_size == 1){msb = -1, lsb = -1;}; //single point
        if(cloud_size == 2){msb = -1, lsb = 1;}; //line element
        if(cloud_size == 3){msb = 1, lsb = -1;}; //triangle element
        if(cloud_size == 4){msb = 1, lsb = 1;}; //rectangle element

        size_t zero = 0;
        size_t one = 1;
        size_t two = 2;


        if(cloud_size > 0)
        {
            vertex = vertex_cloud->GetPoint(0);
            fVertexData[i].s0 = vertex[zero];
            fVertexData[i].s1 = vertex[one];
            fVertexData[i].s2 = vertex[two];
            fVertexData[i].s3 = msb;
        }

        if(cloud_size > 1)
        {
            vertex = vertex_cloud->GetPoint(1);
            fVertexData[i].s4 = vertex[zero];
            fVertexData[i].s5 = vertex[one];
            fVertexData[i].s6 = vertex[two];
            fVertexData[i].s7 = lsb;
        }

        if(cloud_size > 2)
        {
            vertex = vertex_cloud->GetPoint(2);
            fVertexData[i].s8 = vertex[zero];
            fVertexData[i].s9 = vertex[one];
            fVertexData[i].sA = vertex[two];
            //fVertexData[i].sB is irrelevant
        }

        if(cloud_size > 3)
        {
            vertex = vertex_cloud->GetPoint(3);
            fVertexData[i].sC = vertex[zero];
            fVertexData[i].sD = vertex[one];
            fVertexData[i].sE = vertex[two];
            //fVertexData[i].sF is irrelevant
        }
    }

}

void KFMElectrostaticMultipoleBatchCalculator_OpenCL::ComputeMoments()
{
    fMultipoleKernel->setArg(0, fValidSize); //must set to (fValidSize) on each call

    //access the element container and fill the vertex buffer and basis data buffer
    FillTemporaryBuffers();

    //copy data to GPU
    cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fOriginBufferCL, CL_TRUE, 0, fValidSize*sizeof(CL_TYPE4), fIntermediateOriginData);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fVertexDataBufferCL, CL_TRUE, 0, fValidSize*sizeof(CL_TYPE16), fVertexData);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBasisDataBufferCL, CL_TRUE, 0, fValidSize*sizeof(CL_TYPE), fBasisData);

    unsigned int nDummy = fNLocal - (fValidSize%fNLocal);
    if(nDummy == fNLocal)
    {
         nDummy = 0;
    }
    cl::NDRange global(fValidSize + nDummy);
    cl::NDRange local(fNLocal);

    Q.enqueueNDRangeKernel(*fMultipoleKernel, cl::NullRange, global, local);

    Q.enqueueReadBuffer(*fMomentBufferCL, CL_TRUE, 0, fStride*fValidSize*sizeof(CL_TYPE2), fIntermediateMomentData);


    //fill the moment buffer from the computed moments
    for(unsigned int i=0; i<fValidSize; i++)
    {
        int si;
        for(int l=0; l<=fDegree; l++)
        {
            for(int k=0; k<=l; k++)
            {
                si = KFMScalarMultipoleExpansion::RealBasisIndex(l,k);
                fMomentBuffer[i*2*fStride + 2*si] = fIntermediateMomentData[i*fStride + si].s0;
                fMomentBuffer[i*2*fStride + 2*si + 1] = fIntermediateMomentData[i*fStride + si].s1;
            }
        }
    }
}


}//end of KEMField
