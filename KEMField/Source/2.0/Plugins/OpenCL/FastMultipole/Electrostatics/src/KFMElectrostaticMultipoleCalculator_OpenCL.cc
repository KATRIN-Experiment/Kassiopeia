#include "KFMElectrostaticMultipoleCalculator_OpenCL.hh"
#include "KFMMessaging.hh"

#include "KFMPinchonJMatrixCalculator.hh"

#include "KFMPointCloud.hh"
#include "KFMBasisData.hh"

#include <fstream>


namespace KEMField
{


KFMElectrostaticMultipoleCalculator_OpenCL::KFMElectrostaticMultipoleCalculator_OpenCL():
fJSize(0),
fOpenCLFlags(""),
fJMatrix(NULL),
fAxialPlm(NULL),
fEquatorialPlm(NULL),
fBasisData(NULL),
fIntermediateOriginData(NULL),
fVertexData(NULL),
fIntermediateMomentData(NULL),
fZeroData(NULL),
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
};


KFMElectrostaticMultipoleCalculator_OpenCL::~KFMElectrostaticMultipoleCalculator_OpenCL()
{
    delete[] fJMatrix;
    delete[] fAxialPlm;
    delete[] fEquatorialPlm;
    delete[] fBasisData;
    delete[] fIntermediateOriginData;
    delete[] fVertexData;
    delete[] fIntermediateMomentData;
    delete[] fZeroData;
    delete[] fNodeIDData;
    delete[] fNodeIndexData;
    delete[] fStartIndexData;
    delete[] fSizeData;

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

    delete fAnalyticCalc;
}

void KFMElectrostaticMultipoleCalculator_OpenCL::Initialize()
{
    KFMElectrostaticParameters params = fTree->GetParameters();
    fDegree = params.degree;
    fVerbosity = params.degree;
    fStride = (fDegree+1)*(fDegree+2)/2;
    fScratchStride = (fDegree+1)*(fDegree + 1); //scratch space stride

    //create the build flags
    std::stringstream ss;
    ss << " -D KFM_DEGREE=" << fDegree;
    ss << " -D KFM_COMPLEX_STRIDE=" << fScratchStride;
    ss << " -D KFM_REAL_STRIDE=" << fStride;
    ss << " -I " <<KOpenCLInterface::GetInstance()->GetKernelPath();
    fOpenCLFlags = ss.str();

    fAnalyticCalc->SetDegree(fDegree);

    //now we need to build the index between the elements and their nodes
    BuildElementNodeIndex();

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

        if(fNMaxItems == 0)
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
KFMElectrostaticMultipoleCalculator_OpenCL::ConstructOpenCLKernels()
{
    ////////////////////////////////////////////////////////////////////////////
    //build the multipole calculation kernel

    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticMultipole_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fMultipoleKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticMultipole"), fOpenCLFlags );

    //get n-local
    fNLocal = fMultipoleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    ////////////////////////////////////////////////////////////////////////////
    //build the multipole distribution kernel

    //Get name of kernel source file
    std::stringstream clFile2;
    clFile2 << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticMultipoleDistribution_kernel.cl";

    KOpenCLKernelBuilder k_builder2;
    fMultipoleDistributionKernel = k_builder2.BuildKernel(clFile2.str(), std::string("DistributeElectrostaticMultipole"), fOpenCLFlags );

    //get n-local
    fNDistributeLocal = fMultipoleDistributionKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
}

void
KFMElectrostaticMultipoleCalculator_OpenCL::BuildBuffers()
{
    //origin data
    fIntermediateOriginData = new CL_TYPE4[fNMaxItems];

    //vertex data for elements to process
    fVertexData = new CL_TYPE16[fNMaxItems];

    //basis data for elements to process
    fBasisData = new CL_TYPE[fNMaxItems];

    fNodeIDData = new unsigned int[fNMaxItems];

    fNodeIndexData = new unsigned int[fNMaxItems];;
    fStartIndexData = new unsigned int[fNMaxItems];;
    fSizeData = new unsigned int[fNMaxItems];;

    //moment data
    fIntermediateMomentData = new CL_TYPE2[fStride*fNMaxItems];

    fZeroData = new CL_TYPE2[fNMultipoleNodes*fStride];

    for(unsigned int i=0; i<fNMultipoleNodes*fStride; i++)
    {
        fZeroData[i].s0 = 0.0;
        fZeroData[i].s1 = 0.0;
    }

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
    fNodeIDBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(unsigned int));

    fNodeIndexBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(unsigned int));

    fStartIndexBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(unsigned int));

    fSizeBufferCL =
    new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxItems*sizeof(unsigned int));


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
KFMElectrostaticMultipoleCalculator_OpenCL::AssignBuffers()
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

    fMultipoleDistributionKernel->setArg(0, fNMaxItems);  //must be set to (fNGroupUniqueNodes) on each call
    fMultipoleDistributionKernel->setArg(1, *fNodeIndexBufferCL);
    fMultipoleDistributionKernel->setArg(2, *fStartIndexBufferCL);
    fMultipoleDistributionKernel->setArg(3, *fSizeBufferCL);
    fMultipoleDistributionKernel->setArg(4, *fMomentBufferCL);
    fMultipoleDistributionKernel->setArg(5, *fNodeMomentBufferCL);

}


void
KFMElectrostaticMultipoleCalculator_OpenCL::FillTemporaryBuffers()
{
    for(size_t i=0; i<fValidSize; i++)
    {
        unsigned int j = fCurrentElementIndex + i;

        fIntermediateOriginData[i].s0 = (*fOriginList)[j][0];
        fIntermediateOriginData[i].s1 = (*fOriginList)[j][1];
        fIntermediateOriginData[i].s2 = (*fOriginList)[j][2];

        fNodeIDData[i] = fMultipoleNodeIDList[j];

        //std::cout<<"fNode data @ "<<i<<" = "<<fNodeIDData[i]<<std::endl;

        const KFMBasisData<1>* basis = fContainer->GetBasisData( (*fElementIDList)[j] );
        fBasisData[i] = (*basis)[0];

        const KFMPointCloud<3>* vertex_cloud = fContainer->GetPointCloud( (*fElementIDList)[j] );
        unsigned int cloud_size = vertex_cloud->GetNPoints();
        KFMPoint<3> vertex;

        //pattern indicates the number of valid vertices
        int msb, lsb;
        if(cloud_size == 1){msb = -1, lsb = -1;}; //single point
        if(cloud_size == 2){msb = -1, lsb = 1;}; //line element
        if(cloud_size == 3){msb = 1, lsb = -1;}; //triangle element
        if(cloud_size == 4){msb = 1, lsb = 1;}; //rectangle element

        if(cloud_size > 0)
        {
            vertex = vertex_cloud->GetPoint(0);
            fVertexData[i].s0 = vertex[0];
            fVertexData[i].s1 = vertex[1];
            fVertexData[i].s2 = vertex[2];
            fVertexData[i].s3 = msb;
        }

        if(cloud_size > 1)
        {
            vertex = vertex_cloud->GetPoint(1);
            fVertexData[i].s4 = vertex[0];
            fVertexData[i].s5 = vertex[1];
            fVertexData[i].s6 = vertex[2];
            fVertexData[i].s7 = lsb;
        }

        if(cloud_size > 2)
        {
            vertex = vertex_cloud->GetPoint(2);
            fVertexData[i].s8 = vertex[0];
            fVertexData[i].s9 = vertex[1];
            fVertexData[i].sA = vertex[2];
            //fVertexData[i].sB is irrelevant
        }

        if(cloud_size > 3)
        {
            vertex = vertex_cloud->GetPoint(3);
            fVertexData[i].sC = vertex[0];
            fVertexData[i].sD = vertex[1];
            fVertexData[i].sE = vertex[2];
            //fVertexData[i].sF is irrelevant
        }
    }

    fNGroupUniqueNodes = 0;
    unsigned int previous = UINT_MAX; //no elements with this id
    unsigned int size = 0;

    fStartIndexData[0] = 0;
    fNodeIndexData[0] = fNodeIDData[0];

    for(size_t i=0; i<fValidSize; i++)
    {
        size++;
        if( ( previous != fNodeIDData[i] ) || ( i+1 == fValidSize ) )
        {
            fSizeData[fNGroupUniqueNodes] = size;
            previous = fNodeIDData[i];
            fNGroupUniqueNodes++;
            fStartIndexData[fNGroupUniqueNodes] = i;
            fNodeIndexData[fNGroupUniqueNodes] = fNodeIDData[i];
            size = 0;
        };
    }
}

void KFMElectrostaticMultipoleCalculator_OpenCL::ComputeMoments()
{
    fTotalElementsToProcess = fElementIDList->size();
    fRemainingElementsToProcess = fTotalElementsToProcess;
    fCurrentElementIndex = 0;

    //we need to zero out the multipole moment buffer
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeMomentBufferCL, CL_TRUE, 0, fNMultipoleNodes*fStride*sizeof(CL_TYPE2), fZeroData);

    do
    {
        if(fRemainingElementsToProcess < fNMaxItems)
        {
            fNumberOfElementsToProcessOnThisPass = fRemainingElementsToProcess;
        }
        else
        {
            fNumberOfElementsToProcessOnThisPass = fNMaxItems;
        }

        fValidSize = fNumberOfElementsToProcessOnThisPass;

        FillTemporaryBuffers();

        ComputeCurrentMoments();

        DistributeCurrentMoments();

        fCurrentElementIndex += fNumberOfElementsToProcessOnThisPass;

        fRemainingElementsToProcess = fRemainingElementsToProcess - fNumberOfElementsToProcessOnThisPass;
    }
    while(fRemainingElementsToProcess > 0);

}

void
KFMElectrostaticMultipoleCalculator_OpenCL::ComputeCurrentMoments()
{
    fMultipoleKernel->setArg(0, fValidSize); //must set to (fValidSize) on each call

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
}

void
KFMElectrostaticMultipoleCalculator_OpenCL::DistributeCurrentMoments()
{
    fMultipoleDistributionKernel->setArg(0, fNGroupUniqueNodes);  //must be set to fNGroupUniqueNodes on each call

    //copy node id's to GPU
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIndexBufferCL, CL_TRUE, 0, fNGroupUniqueNodes*sizeof(unsigned int), fNodeIndexData);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fStartIndexBufferCL, CL_TRUE, 0, fNGroupUniqueNodes*sizeof(unsigned int), fStartIndexData);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSizeBufferCL, CL_TRUE, 0, fNGroupUniqueNodes*sizeof(unsigned int), fSizeData);


    //run the distribution kernel
    unsigned int nDummy = fNDistributeLocal - (fNGroupUniqueNodes%fNDistributeLocal);
    if(nDummy == fNDistributeLocal)
    {
         nDummy = 0;
    }
    cl::NDRange global(fNGroupUniqueNodes + nDummy);
    cl::NDRange local(fNDistributeLocal);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fMultipoleDistributionKernel, cl::NullRange, global, local);
}


void
KFMElectrostaticMultipoleCalculator_OpenCL::BuildElementNodeIndex()
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

    for(unsigned int i=0; i<fNodeIDList->size(); i++)
    {

        int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID( (*fNodeIDList)[i]);

        if(special_id != -1)
        {
            unsigned int index = static_cast<unsigned int>(special_id);
            fMultipoleNodeIDList.push_back(index);
        }
        else
        {
            //error
            kfmout<<"KFMElectrostaticMultipoleCalculator_OpenCL::BuildElementNodeIndex: Error, node missing from non-zero multipole set. Aborting. "<<kfmendl;
            kfmexit(1);
        }
    }

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticMultipoleCalculator_OpenCL::BuildElementNodeIndex: Element to node association done. "<<kfmendl;
    }
}


}//end of KEMField
