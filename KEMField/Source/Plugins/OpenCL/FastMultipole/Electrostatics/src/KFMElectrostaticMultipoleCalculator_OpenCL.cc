#include "KFMElectrostaticMultipoleCalculator_OpenCL.hh"

#include "KFMBasisData.hh"
#include "KFMMessaging.hh"
#include "KFMPinchonJMatrixCalculator.hh"
#include "KFMPointCloud.hh"

#include <fstream>


namespace KEMField
{


KFMElectrostaticMultipoleCalculator_OpenCL::KFMElectrostaticMultipoleCalculator_OpenCL(bool standalone) :
    fStandAlone(standalone),
    fPrimaryIntegrationMode(-1),
    fSecondaryIntegrationMode(1),
    fJSize(0),
    fAbscissa(NULL),
    fWeights(NULL),
    fAbscissaBufferCL(NULL),
    fWeightsBufferCL(NULL),
    fOpenCLFlags(""),
    fJMatrix(NULL),
    fAxialPlm(NULL),
    fEquatorialPlm(NULL),
    fACoefficient(NULL),
    fBasisData(NULL),
    fIntermediateOriginData(NULL),
    fVertexData(NULL),
    fNodeIDData(NULL),
    fOriginBufferCL(NULL),
    fVertexDataBufferCL(NULL),
    fBasisDataBufferCL(NULL),
    fMomentBufferCL(NULL),
    fACoefficientBufferCL(NULL),
    fEquatorialPlmBufferCL(NULL),
    fAxialPlmBufferCL(NULL),
    fJMatrixBufferCL(NULL),
    fNodeIDBufferCL(NULL),
    fNodeIndexBufferCL(NULL),
    fStartIndexBufferCL(NULL),
    fSizeBufferCL(NULL),
    fNodeMomentBufferCL(NULL)
{
    fMultipoleNodes = NULL;
    fAnalyticCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();

    unsigned int buffer_mb = KEMFIELD_MULTIPOLE_BUFFER_SIZE_MB;
    fMaxBufferSizeInBytes = buffer_mb * 1024 * 1024;
    fInitialized = false;

    fMultipoleKernel = NULL;
    fMultipoleDistributionKernel = NULL;
    fZeroKernel = NULL;

    //warning about use of standalone mode
    if (fStandAlone) {
        kfmout
            << "KFMElectrostaticMultipoleCalculator_OpenCL: Warning! Stand-alone mode has been turned on. This is only meant to be used for debugging/testing purposes."
            << kfmendl;
    }
};


KFMElectrostaticMultipoleCalculator_OpenCL::~KFMElectrostaticMultipoleCalculator_OpenCL()
{
    delete[] fJMatrix;
    delete[] fAxialPlm;
    delete[] fEquatorialPlm;
    delete[] fACoefficient;
    delete[] fBasisData;
    delete[] fIntermediateOriginData;
    delete[] fVertexData;
    delete[] fNodeIDData;
    delete[] fNodeIndexData;
    delete[] fStartIndexData;
    delete[] fSizeData;

    delete[] fAbscissa;
    delete[] fWeights;

    delete fOriginBufferCL;
    delete fVertexDataBufferCL;
    delete fBasisDataBufferCL;
    delete fMomentBufferCL;
    delete fACoefficientBufferCL;
    delete fEquatorialPlmBufferCL;
    delete fAxialPlmBufferCL;
    delete fJMatrixBufferCL;
    delete fNodeIDBufferCL;
    delete fNodeIndexBufferCL;
    delete fStartIndexBufferCL;
    delete fSizeBufferCL;
    delete fAbscissaBufferCL;
    delete fWeightsBufferCL;

    delete fAnalyticCalc;

    delete fMultipoleDistributionKernel;
    delete fMultipoleKernel;
    delete fZeroKernel;
}

void KFMElectrostaticMultipoleCalculator_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    fDegree = params.degree;
    fVerbosity = params.verbosity;
    fStride = (fDegree + 1) * (fDegree + 2) / 2;
    fScratchStride = (fDegree + 1) * (fDegree + 1);  //scratch space stride
}

void KFMElectrostaticMultipoleCalculator_OpenCL::Initialize()
{
    //create the build flags
    std::stringstream ss;
    ss << " -D KFM_DEGREE=" << fDegree;
    ss << " -D KFM_COMPLEX_STRIDE=" << fScratchStride;
    ss << " -D KFM_REAL_STRIDE=" << fStride;
    ss << " -I " << KOpenCLInterface::GetInstance()->GetKernelPath();
    fOpenCLFlags = ss.str();

    fAnalyticCalc->SetDegree(fDegree);

    //now we need to build the index between the elements and their nodes
    if (!fStandAlone) {
        BuildElementNodeIndex();
    }

    if (!fInitialized) {
        //compute gaussian quadrature rules
        fQuadratureTableCalc.SetNTerms(fDegree + 1);
        fQuadratureTableCalc.Initialize();
        fQuadratureTableCalc.GetAbscissa(&fAbscissaVector);
        fQuadratureTableCalc.GetWeights(&fWeightsVector);

        //construct the kernel and determine the number of work group size
        ConstructOpenCLKernels();

        //now lets figure out how many elements we can process at a time
        //due to size constraint of multipole buffer
        unsigned int bytes_per_element = fStride * sizeof(CL_TYPE2);
        unsigned int buff_size = fMaxBufferSizeInBytes;
        unsigned int max_items = buff_size / bytes_per_element;

        //now lets figure out how many elements we can fit in memory
        //due to the native device buffer size (and size of geometry data per element)
        unsigned int max_geometric_items = buff_size / (sizeof(CL_TYPE16));

        if (max_items > max_geometric_items) {
            max_items = max_geometric_items;
        }

        //ensure that max items is a multiple of the workgroup size
        //though this may result in a buffer that is slightly larger than the requested size
        unsigned int nDummy = fNLocal - (max_items % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        max_items += nDummy;

        //compute new buffer size, and number of items we can process
        buff_size = bytes_per_element * max_items;
        fMaxBufferSizeInBytes = buff_size;
        fNMaxItems = max_items;
        fNMaxWorkgroups = fNMaxItems / fNLocal;
        fValidSize = fNMaxItems;

        if (fNMaxItems == 0) {
            //warning & abort
            std::stringstream ss;
            ss << "Buffer size of ";
            ss << fMaxBufferSizeInBytes;
            ss << " bytes is not large enough for a single element. ";
            ss << "Required bytes per element = " << bytes_per_element << ". Aborting.";
            kfmout << ss.str() << std::endl;
            kfmexit(1);
        }

        BuildBuffers();

        AssignBuffers();

        fInitialized = true;
    }
}


void KFMElectrostaticMultipoleCalculator_OpenCL::ConstructOpenCLKernels()
{

    unsigned int preferredWorkgroupMultiple;

    ////////////////////////////////////////////////////////////////////////////
    //build the multipole calculation kernel

    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticMultipole_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fMultipoleKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticMultipole"), fOpenCLFlags);

    //get n-local
    fNLocal =
        fMultipoleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    preferredWorkgroupMultiple = fMultipoleKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
        KOpenCLInterface::GetInstance()->GetDevice());

    if (preferredWorkgroupMultiple < fNLocal) {
        fNLocal = preferredWorkgroupMultiple;
    }

    ////////////////////////////////////////////////////////////////////////////
    //build the multipole distribution kernel

    //Get name of kernel source file
    std::stringstream clFile2;
    clFile2 << KOpenCLInterface::GetInstance()->GetKernelPath()
            << "/kEMField_KFMElectrostaticMultipoleDistribution_kernel.cl";

    KOpenCLKernelBuilder k_builder2;
    fMultipoleDistributionKernel =
        k_builder2.BuildKernel(clFile2.str(), std::string("DistributeElectrostaticMultipole"), fOpenCLFlags);

    //get n-local
    fNDistributeLocal = fMultipoleDistributionKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
        KOpenCLInterface::GetInstance()->GetDevice());

    preferredWorkgroupMultiple =
        fMultipoleDistributionKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            KOpenCLInterface::GetInstance()->GetDevice());

    if (preferredWorkgroupMultiple < fNDistributeLocal) {
        fNDistributeLocal = preferredWorkgroupMultiple;
    }


    ////////////////////////////////////////////////////////////////////////////
    //build the multipole zeroing kernel

    //Get name of kernel source file
    std::stringstream clFile3;
    clFile3 << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMZeroComplexArray_kernel.cl";

    KOpenCLKernelBuilder k_builder3;
    fZeroKernel = k_builder3.BuildKernel(clFile3.str(), std::string("ZeroComplexArray"));

    //get n-local
    fNZeroLocal =
        fZeroKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    preferredWorkgroupMultiple = fZeroKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
        KOpenCLInterface::GetInstance()->GetDevice());

    if (preferredWorkgroupMultiple < fNZeroLocal) {
        fNZeroLocal = preferredWorkgroupMultiple;
    }
}


void KFMElectrostaticMultipoleCalculator_OpenCL::BuildBuffers()
{
    //origin data
    fIntermediateOriginData = new CL_TYPE4[fNMaxItems];

    //vertex data for elements to process
    fVertexData = new CL_TYPE16[fNMaxItems];

    //basis data for elements to process
    fBasisData = new CL_TYPE[fNMaxItems];

    fNodeIDData = new unsigned int[fNMaxItems];

    fNodeIndexData = new unsigned int[fNMaxItems];
    ;
    fStartIndexData = new unsigned int[fNMaxItems];
    ;
    fSizeData = new unsigned int[fNMaxItems];
    ;

    //moment data
    fMultipoleBufferSize = fNMultipoleNodes * fStride;

    //numerical integrator data
    fAbscissa = new CL_TYPE[fDegree + 1];
    fWeights = new CL_TYPE[fDegree + 1];

    for (int i = 0; i <= fDegree; i++) {
        fAbscissa[i] = fAbscissaVector[i];
        fWeights[i] = fWeightsVector[i];
    }

    //compute the equatorial associated legendre polynomial array
    fEquatorialPlm = new CL_TYPE[fStride];
    KFMMath::ALP_nm_array(fDegree, 0.0, (double*) (fEquatorialPlm));

    //compute the axial associated legendre polynomial array
    fAxialPlm = new CL_TYPE[fStride];
    KFMMath::ALP_nm_array(fDegree, 1.0, (double*) (fAxialPlm));

    fACoefficient = new CL_TYPE[fStride];
    //compute the A coefficients
    int si;
    for (int n = 0; n <= fDegree; n++) {
        for (int m = 0; m <= n; m++) {
            si = KFMScalarMultipoleExpansion::RealBasisIndex(n, m);
            fACoefficient[si] = KFMMath::A_Coefficient(m, n);
        }
    }

    //compute the pinchon j-matrices
    KFMPinchonJMatrixCalculator j_matrix_calc;
    std::vector<kfm_matrix*> j_matrix_vector;

    j_matrix_calc.SetDegree(fDegree);
    j_matrix_calc.AllocateMatrices(&j_matrix_vector);
    j_matrix_calc.ComputeMatrices(&j_matrix_vector);

    //figure out the size of the array we need:
    fJSize = 0;
    for (int i = 0; i <= fDegree; i++) {
        fJSize += (2 * i + 1) * (2 * i + 1);
    }
    fJMatrix = new CL_TYPE[fJSize];

    //loop over array of j matrices and push their data into the array
    int j_size = 0;
    int current_size;
    for (int l = 0; l <= fDegree; l++) {
        current_size = 2 * l + 1;

        for (int row = 0; row < current_size; row++) {
            for (int col = 0; col < current_size; col++) {
                fJMatrix[j_size + row * current_size + col] = kfm_matrix_get(j_matrix_vector.at(l), row, col);
            }
        }
        j_size += (2 * l + 1) * (2 * l + 1);
    }

    j_matrix_calc.DeallocateMatrices(&j_matrix_vector);


    //create buffers for the constant objects
    CL_ERROR_TRY
    {
        fACoefficientBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fStride * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fEquatorialPlmBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fStride * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fAxialPlmBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fStride * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fJMatrixBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fJSize * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    //create buffers for the non-constant objects (must be enqueued writen/read on each call)
    CL_ERROR_TRY
    {
        fNodeIDBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                         CL_MEM_READ_ONLY,
                                         fNMaxItems * sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fNodeIndexBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                            CL_MEM_READ_ONLY,
                                            fNMaxItems * sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fStartIndexBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                             CL_MEM_READ_ONLY,
                                             fNMaxItems * sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fSizeBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                       CL_MEM_READ_ONLY,
                                       fNMaxItems * sizeof(unsigned int));
    }
    CL_ERROR_CATCH


    CL_ERROR_TRY
    {
        fOriginBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                         CL_MEM_READ_ONLY,
                                         fNMaxItems * sizeof(CL_TYPE4));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fVertexDataBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                             CL_MEM_READ_ONLY,
                                             fNMaxItems * sizeof(CL_TYPE16));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fBasisDataBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                            CL_MEM_READ_ONLY,
                                            fNMaxItems * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fMomentBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                         CL_MEM_WRITE_ONLY,
                                         fStride * fNMaxItems * sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fAbscissaBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                           CL_MEM_READ_ONLY,
                                           (fDegree + 1) * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fWeightsBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                          CL_MEM_READ_ONLY,
                                          (fDegree + 1) * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH
}


void KFMElectrostaticMultipoleCalculator_OpenCL::AssignBuffers()
{
    // Set arguments to kernel
    fMultipoleKernel->setArg(0, fNMaxItems);  //must be set to (fValidSize) on each call

    //set constant arguments/buffers
    fMultipoleKernel->setArg(1, *fACoefficientBufferCL);
    fMultipoleKernel->setArg(2, *fEquatorialPlmBufferCL);
    fMultipoleKernel->setArg(3, *fAxialPlmBufferCL);
    fMultipoleKernel->setArg(4, *fJMatrixBufferCL);
    fMultipoleKernel->setArg(5, *fAbscissaBufferCL);
    fMultipoleKernel->setArg(6, *fWeightsBufferCL);

    //set non-const arguments/buffers
    fMultipoleKernel->setArg(7, *fOriginBufferCL);
    fMultipoleKernel->setArg(8, *fVertexDataBufferCL);
    fMultipoleKernel->setArg(9, *fBasisDataBufferCL);
    fMultipoleKernel->setArg(10, *fMomentBufferCL);

    //copy const data to the GPU
    cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();
    Q.enqueueWriteBuffer(*fACoefficientBufferCL, CL_TRUE, 0, fStride * sizeof(CL_TYPE), fACoefficient);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
    Q.enqueueWriteBuffer(*fEquatorialPlmBufferCL, CL_TRUE, 0, fStride * sizeof(CL_TYPE), fEquatorialPlm);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
    Q.enqueueWriteBuffer(*fAxialPlmBufferCL, CL_TRUE, 0, fStride * sizeof(CL_TYPE), fAxialPlm);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
    Q.enqueueWriteBuffer(*fJMatrixBufferCL, CL_TRUE, 0, fJSize * sizeof(CL_TYPE), fJMatrix);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
    Q.enqueueWriteBuffer(*fAbscissaBufferCL, CL_TRUE, 0, (fDegree + 1) * sizeof(CL_TYPE), fAbscissa);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
    Q.enqueueWriteBuffer(*fWeightsBufferCL, CL_TRUE, 0, (fDegree + 1) * sizeof(CL_TYPE), fWeights);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

/*
    double mem_size = 0;
    mem_size += fStride * sizeof(CL_TYPE);
    mem_size += fStride * sizeof(CL_TYPE);
    mem_size += fStride * sizeof(CL_TYPE);
    mem_size += fJSize * sizeof(CL_TYPE);
    mem_size += (fDegree + 1) * sizeof(CL_TYPE);
    mem_size += (fDegree + 1) * sizeof(CL_TYPE);
*/

    fMultipoleDistributionKernel->setArg(0, fNMaxItems);  //must be set to (fNGroupUniqueNodes) on each call
    fMultipoleDistributionKernel->setArg(1, *fNodeIndexBufferCL);
    fMultipoleDistributionKernel->setArg(2, *fStartIndexBufferCL);
    fMultipoleDistributionKernel->setArg(3, *fSizeBufferCL);
    fMultipoleDistributionKernel->setArg(4, *fMomentBufferCL);

    if (!fStandAlone) {
        fMultipoleDistributionKernel->setArg(5, *fNodeMomentBufferCL);
    }

    fZeroKernel->setArg(0, fMultipoleBufferSize);

    if (!fStandAlone) {
        fZeroKernel->setArg(1, *fNodeMomentBufferCL);
    }
}


void KFMElectrostaticMultipoleCalculator_OpenCL::FillTemporaryBuffers()
{
    for (size_t i = 0; i < fValidSize; i++) {
        unsigned int j = fCurrentElementIndex + i;

        fIntermediateOriginData[i].s[0] = (*fOriginList)[j][0];
        fIntermediateOriginData[i].s[1] = (*fOriginList)[j][1];
        fIntermediateOriginData[i].s[2] = (*fOriginList)[j][2];

        fNodeIDData[i] = fMultipoleNodeIDList[j];


        unsigned int element_index = (*fElementIDList)[j];

        const KFMBasisData<1>* basis = fContainer->GetBasisData(element_index);
        fBasisData[i] = (*basis)[0];

        const KFMPointCloud<3>* vertex_cloud = fContainer->GetPointCloud(element_index);
        unsigned int cloud_size = vertex_cloud->GetNPoints();
        KFMPoint<3> vertex;

        //pattern indicates the number of valid vertices
        int msb = -1;
        int lsb = -1;
        if (cloud_size == 1) {
            msb = -1;
            lsb = -1;
        };  //single point
        if (cloud_size == 2) {
            msb = -1;
            lsb = 1;
        };  //line element
        if (cloud_size == 3) {
            msb = 1;
            lsb = -1;
        };  //triangle element
        if (cloud_size == 4) {
            msb = 1;
            lsb = 1;
        };  //rectangle element

        //this value if the element has an acceptable aspect ratio
        //only relevant for triangles and rectangles

        int aspect_indicator = fPrimaryIntegrationMode;
        if (cloud_size == 3 || cloud_size == 4) {
            double ar = fContainer->GetAspectRatio(element_index);
            if (ar > 30.0) {
                //a large aspect ratio triggers a slower but
                //more accurate method of computing the multipole moments
                aspect_indicator = fSecondaryIntegrationMode;
            };
        }


        if (cloud_size > 0) {
            vertex = vertex_cloud->GetPoint(0);
            fVertexData[i].s[0] = vertex[0];
            fVertexData[i].s[1] = vertex[1];
            fVertexData[i].s[2] = vertex[2];
            fVertexData[i].s[3] = msb;
        }

        if (cloud_size > 1) {
            vertex = vertex_cloud->GetPoint(1);
            fVertexData[i].s[4] = vertex[0];
            fVertexData[i].s[5] = vertex[1];
            fVertexData[i].s[6] = vertex[2];
            fVertexData[i].s[7] = lsb;
        }

        if (cloud_size > 2) {
            vertex = vertex_cloud->GetPoint(2);
            fVertexData[i].s[8] = vertex[0];
            fVertexData[i].s[9] = vertex[1];
            fVertexData[i].s[10] = vertex[2];
            fVertexData[i].s[11] = aspect_indicator;  //validity of aspect ratio indicator
        }

        if (cloud_size > 3) {
            vertex = vertex_cloud->GetPoint(3);
            fVertexData[i].s[12] = vertex[0];
            fVertexData[i].s[13] = vertex[1];
            fVertexData[i].s[14] = vertex[2];
            //fVertexData[i].sF is irrelevant
        }
    }

    fNGroupUniqueNodes = 0;
    fStartIndexData[0] = 0;
    fNodeIndexData[0] = fNodeIDData[0];

    for (unsigned int i = 0; i < fValidSize; i++) {
        if (fNodeIndexData[fNGroupUniqueNodes] != fNodeIDData[i]) {
            fSizeData[fNGroupUniqueNodes] = i - fStartIndexData[fNGroupUniqueNodes];
            fNGroupUniqueNodes++;
            fStartIndexData[fNGroupUniqueNodes] = i;
            fNodeIndexData[fNGroupUniqueNodes] = fNodeIDData[i];
        }
    }

    fSizeData[fNGroupUniqueNodes] = fValidSize - fStartIndexData[fNGroupUniqueNodes];
    fNGroupUniqueNodes++;
}

void KFMElectrostaticMultipoleCalculator_OpenCL::ComputeMoments()
{
    fTotalElementsToProcess = fElementIDList->size();
    fRemainingElementsToProcess = fTotalElementsToProcess;
    fCurrentElementIndex = 0;


    //start up the kernel which zero's out the multipole moment buffer
    //we need to use a kernel to zero out the buffer because the OpenCL 1.1 spec
    //lacks the function clEnqueueFillBuffer

    unsigned int nDummy = fNZeroLocal - (fMultipoleBufferSize % fNZeroLocal);
    if (nDummy == fNZeroLocal) {
        nDummy = 0;
    }
    cl::NDRange global(fMultipoleBufferSize + nDummy);
    cl::NDRange local(fNZeroLocal);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fZeroKernel, cl::NullRange, global, local);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

    do {
        if (fRemainingElementsToProcess < fNMaxItems) {
            fNumberOfElementsToProcessOnThisPass = fRemainingElementsToProcess;
        }
        else {
            fNumberOfElementsToProcessOnThisPass = fNMaxItems;
        }

        fValidSize = fNumberOfElementsToProcessOnThisPass;

        FillTemporaryBuffers();

        ComputeCurrentMoments();

        DistributeCurrentMoments();

        fCurrentElementIndex += fNumberOfElementsToProcessOnThisPass;

        fRemainingElementsToProcess = fRemainingElementsToProcess - fNumberOfElementsToProcessOnThisPass;
    } while (fRemainingElementsToProcess > 0);
}

void KFMElectrostaticMultipoleCalculator_OpenCL::ComputeCurrentMoments()
{
    fMultipoleKernel->setArg(0, fValidSize);  //must set to (fValidSize) on each call

    //copy data to GPU
    cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fOriginBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fValidSize * sizeof(CL_TYPE4),
                                                                   fIntermediateOriginData);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fVertexDataBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fValidSize * sizeof(CL_TYPE16),
                                                                   fVertexData);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBasisDataBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fValidSize * sizeof(CL_TYPE),
                                                                   fBasisData);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

    unsigned int nDummy = fNLocal - (fValidSize % fNLocal);
    if (nDummy == fNLocal) {
        nDummy = 0;
    }
    cl::NDRange global(fValidSize + nDummy);
    cl::NDRange local(fNLocal);

    Q.enqueueNDRangeKernel(*fMultipoleKernel, cl::NullRange, global, local);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
}

void KFMElectrostaticMultipoleCalculator_OpenCL::DistributeCurrentMoments()
{
    fMultipoleDistributionKernel->setArg(0, fNGroupUniqueNodes);  //must be set to fNGroupUniqueNodes on each call

    //copy node id's to GPU
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIndexBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNGroupUniqueNodes * sizeof(unsigned int),
                                                                   fNodeIndexData);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fStartIndexBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNGroupUniqueNodes * sizeof(unsigned int),
                                                                   fStartIndexData);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSizeBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNGroupUniqueNodes * sizeof(unsigned int),
                                                                   fSizeData);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif


    //run the distribution kernel
    unsigned int nGlobal = fNGroupUniqueNodes * fStride;
    unsigned int nDummy = fNDistributeLocal - (nGlobal % fNDistributeLocal);
    if (nDummy == fNDistributeLocal) {
        nDummy = 0;
    }
    cl::NDRange global(nGlobal + nDummy);
    cl::NDRange local(fNDistributeLocal);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fMultipoleDistributionKernel,
                                                                     cl::NullRange,
                                                                     global,
                                                                     local);
#ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
}


void KFMElectrostaticMultipoleCalculator_OpenCL::BuildElementNodeIndex()
{
    //find the association between elements and nodes
    fElementNodeAssociator.Clear();
    fTree->ApplyRecursiveAction(&fElementNodeAssociator);

    fElementIDList = fElementNodeAssociator.GetElementIDList();
    fNElements = fElementIDList->size();
    fNodePtrList = fElementNodeAssociator.GetNodeList();
    fNodeIDList = fElementNodeAssociator.GetNodeIDList();
    fOriginList = fElementNodeAssociator.GetOriginList();

    //now we have to fill the list of the multipole-set ids
    fNMultipoleNodes = fMultipoleNodes->GetSize();
    fMultipoleNodeIDList.clear();

    for (unsigned int i = 0; i < fNodeIDList->size(); i++) {

        int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID((*fNodeIDList)[i]);

        if (special_id != -1) {
            unsigned int index = static_cast<unsigned int>(special_id);
            fMultipoleNodeIDList.push_back(index);
        }
        else {
            //error
            kfmout
                << "KFMElectrostaticMultipoleCalculator_OpenCL::BuildElementNodeIndex: Error, node missing from non-zero multipole set. Aborting. "
                << kfmendl;
            kfmexit(1);
        }
    }

    if (fVerbosity > 4) {
        kfmout
            << "KFMElectrostaticMultipoleCalculator_OpenCL::BuildElementNodeIndex: Element to node association done. "
            << kfmendl;
    }
}


bool KFMElectrostaticMultipoleCalculator_OpenCL::ConstructExpansion(double* target_origin,
                                                                    const KFMPointCloud<3>* vertex_cloud,
                                                                    KFMScalarMultipoleExpansion* moments) const
{
    if (fStandAlone) {
        //copy target origin to the origin buffer
        fIntermediateOriginData[0].s[0] = target_origin[0];
        fIntermediateOriginData[0].s[1] = target_origin[1];
        fIntermediateOriginData[0].s[2] = target_origin[2];

        //copy the vertices into the vertex buffer
        unsigned int cloud_size = vertex_cloud->GetNPoints();
        KFMPoint<3> vertex;

        //pattern indicates the number of valid vertices
        int msb, lsb;
        if (cloud_size == 1) {
            msb = -1;
            lsb = -1;
        };  //single point
        if (cloud_size == 2) {
            msb = -1;
            lsb = 1;
        };  //line element
        if (cloud_size == 3) {
            msb = 1;
            lsb = -1;
        };  //triangle element
        if (cloud_size == 4) {
            msb = 1;
            lsb = 1;
        };  //rectangle element

        //integration mode is selected externally (does not depend on aspect ratio in standalone mode)
        int aspect_indicator = fPrimaryIntegrationMode;

        if (cloud_size > 0) {
            vertex = vertex_cloud->GetPoint(0);
            fVertexData[0].s[0] = vertex[0];
            fVertexData[0].s[1] = vertex[1];
            fVertexData[0].s[2] = vertex[2];
            fVertexData[0].s[3] = msb;
        }

        if (cloud_size > 1) {
            vertex = vertex_cloud->GetPoint(1);
            fVertexData[0].s[4] = vertex[0];
            fVertexData[0].s[5] = vertex[1];
            fVertexData[0].s[6] = vertex[2];
            fVertexData[0].s[7] = lsb;
        }

        if (cloud_size > 2) {
            vertex = vertex_cloud->GetPoint(2);
            fVertexData[0].s[8] = vertex[0];
            fVertexData[0].s[9] = vertex[1];
            fVertexData[0].s[10] = vertex[2];
            fVertexData[0].s[11] = aspect_indicator;  //validity of aspect ratio indicator
        }

        if (cloud_size > 3) {
            vertex = vertex_cloud->GetPoint(3);
            fVertexData[0].s[12] = vertex[0];
            fVertexData[0].s[13] = vertex[1];
            fVertexData[0].s[14] = vertex[2];
            //fVertexData[i].sF is irrelevant
        }

        //set basis data to 1 (unit charge density)
        fBasisData[0] = 1.0;


        fMultipoleKernel->setArg(0, 1);  //valid size is 1

        //copy data to GPU
        cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fOriginBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       fValidSize * sizeof(CL_TYPE4),
                                                                       fIntermediateOriginData);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fVertexDataBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       fValidSize * sizeof(CL_TYPE16),
                                                                       fVertexData);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBasisDataBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       fValidSize * sizeof(CL_TYPE),
                                                                       fBasisData);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        unsigned int nDummy = fNLocal - (fValidSize % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        }
        cl::NDRange global(fValidSize + nDummy);
        cl::NDRange local(fNLocal);

        Q.enqueueNDRangeKernel(*fMultipoleKernel, cl::NullRange, global, local);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        //now read out the multipole moment buffer
        std::vector<CL_TYPE2> moment_data;
        moment_data.resize(fStride * fValidSize);
        std::vector<std::complex<double>> moment_data_converted;
        moment_data_converted.resize((fDegree + 1) * (fDegree + 1));

        Q.enqueueReadBuffer(*fMomentBufferCL, CL_TRUE, 0, fStride * fValidSize * sizeof(CL_TYPE2), &(moment_data[0]));
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        //fill the moment buffer from the computed moments
        int rsi, pcsi, ncsi;
        for (int l = 0; l <= fDegree; l++) {
            for (int k = 0; k <= l; k++) {
                rsi = KFMScalarMultipoleExpansion::RealBasisIndex(l, k);
                pcsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(l, k);
                ncsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(l, -k);
                double real = moment_data[rsi].s[0];
                double imag = moment_data[rsi].s[1];
                moment_data_converted[pcsi] = std::complex<double>(real, imag);
                moment_data_converted[ncsi] = std::complex<double>(real, -1.0 * imag);
            }
        }

        moments->SetMoments(&moment_data_converted);
        return true;
    }

    return false;
}


}  // namespace KEMField
