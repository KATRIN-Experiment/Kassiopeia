#include "KFMElectrostaticBatchedLocalToLocalConverter_OpenCL.hh"


#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif

namespace KEMField
{

KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::KFMElectrostaticBatchedLocalToLocalConverter_OpenCL()
{
    fTree = NULL;

    fTransformKernel = NULL;
    fNTransformLocal = 0;

    fL2LCoeffBufferCL = NULL;
    fSourceScaleFactorBufferCL = NULL;
    fTargetScaleFactorBufferCL = NULL;
    fNodeMomentBufferCL = NULL;
    fNodeIDBufferCL = NULL;
    fBlockSetIDListBufferCL = NULL;
    fParentIDBufferCL = NULL;

    fNPrimaryNodes = 0;
    fPrimaryNodes = NULL;

    fCachedPrimaryNodeIDLists.clear();
    fCachedBlockSetIDLists.clear();

    fDegree = 0;
    fNTerms = 0;
    fStride = 0;
    fDivisions = 0;
    fTopLevelDivisions = 0;
    fLowerLevelDivisions = 0;
    fMaxTreeDepth = 0;
    fWorldLength = 0;
    fNMaxParentNodes = 100;
    fNMaxBufferedNodes = 0;  //set in initialization, fNMaxParentNodes*n_children
    fNBufferedNodes = 0;

    fKernelResponse = new KFMKernelResponseArray_3DLaplaceL2L();  //false -> origin is the target
    fScaleInvariantKernel = NULL;

    fL2LCoeff = NULL;
};


KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::~KFMElectrostaticBatchedLocalToLocalConverter_OpenCL()
{
    delete fTransformKernel;

    delete fL2LCoeffBufferCL;
    delete fSourceScaleFactorBufferCL;
    delete fTargetScaleFactorBufferCL;
    delete fNodeIDBufferCL;
    delete fBlockSetIDListBufferCL;
    delete fParentIDBufferCL;

    delete fKernelResponse;
    delete fL2LCoeff;
};

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    //set parameters
    fDegree = params.degree;
    fNTerms = (fDegree + 1) * (fDegree + 1);
    fStride = (fDegree + 1) * (fDegree + 2) / 2;
    fDivisions = params.divisions;
    fMaxTreeDepth = params.maximum_tree_depth;
    fZeroMaskSize = params.zeromask;
}


void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;

    //determine world region size to compute scale factors
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube =
        KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM>>::GetNodeObject(
            fTree->GetRootNode());
    fWorldLength = world_cube->GetLength();

    //now we want to retrieve the top level and lower level divisions
    //since we need both in order to compute the scale factor for the different tree levels correctly
    const unsigned int* dim_size;
    dim_size =
        KFMObjectRetriever<KFMElectrostaticNodeObjects,
                           KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>>::GetNodeObject(fTree->GetRootNode())
            ->GetTopLevelDimensions();
    fTopLevelDivisions = dim_size[0];

    dim_size =
        KFMObjectRetriever<KFMElectrostaticNodeObjects,
                           KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>>::GetNodeObject(fTree->GetRootNode())
            ->GetDimensions();
    fLowerLevelDivisions = dim_size[0];
};


void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::SetPrimaryNodeSet(
    KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* primary_node_set)
{
    fPrimaryNodes = primary_node_set;

    fNPrimaryNodes = fPrimaryNodes->GetSize();
    fCachedPrimaryNodeIDLists.resize(fNPrimaryNodes);
    fCachedBlockSetIDLists.resize(fNPrimaryNodes);

    //fill the node's list of child nodes and their positions (block set ids)
    for (unsigned int i = 0; i < fNPrimaryNodes; i++) {
        fCachedPrimaryNodeIDLists[i].clear();
        fCachedBlockSetIDLists[i].clear();

        KFMNode<KFMElectrostaticNodeObjects>* node = fPrimaryNodes->GetNodeFromSpecializedID(i);

        if (node->HasChildren()) {
            unsigned int n_children = node->GetNChildren();
            for (unsigned int j = 0; j < n_children; j++) {
                int special_id = fPrimaryNodes->GetSpecializedIDFromOrdinaryID(node->GetChild(j)->GetID());

                if (special_id != -1) {
                    unsigned int id = static_cast<unsigned int>(special_id);
                    fCachedPrimaryNodeIDLists[i].push_back(id);
                    fCachedBlockSetIDLists[i].push_back(j);
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::Prepare()
{
    ClearBuffers();
    fCachedNodeLevel = 0;
}


////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::Finalize()
{
    //finish any remaining L2L translation
    ExecuteBufferedAction();
    ClearBuffers();

    //read the primary node local coefficients back from the gpu;
    unsigned int primary_size = fNPrimaryNodes * fStride;

    std::vector<std::complex<double>> primary_local_coeff;
    primary_local_coeff.resize(primary_size, std::complex<double>(0., 0.));

    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fNodeMomentBufferCL,
                                                                  CL_TRUE,
                                                                  0,
                                                                  primary_size * sizeof(CL_TYPE2),
                                                                  &(primary_local_coeff[0]));

    //now distribute the primary node moments
    for (unsigned int i = 0; i < fNPrimaryNodes; i++) {
        KFMNode<KFMElectrostaticNodeObjects>* node = fPrimaryNodes->GetNodeFromSpecializedID(i);
        KFMElectrostaticLocalCoefficientSet* set =
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);

        if (set != NULL) {
            std::complex<double> temp;
            //we use raw ptr for speed
            double* rmoments = &((*(set->GetRealMoments()))[0]);
            double* imoments = &((*(set->GetImaginaryMoments()))[0]);
            for (unsigned int j = 0; j < fStride; ++j) {
                temp = primary_local_coeff[i * fStride + j];
                rmoments[j] = temp.real();
                imoments[j] = temp.imag();
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////
void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::Initialize()
{
    //set number of terms in series for response functions
    fKernelResponse->SetNumberOfTermsInSeries(fNTerms);

    fLowerLimits[0] = 0;
    fLowerLimits[1] = 0;
    fUpperLimits[0] = fNTerms;
    fUpperLimits[1] = fNTerms;
    fDimensionSize[0] = fNTerms;
    fDimensionSize[1] = fNTerms;

    for (unsigned int i = 0; i < KFMELECTROSTATICS_DIM; i++) {
        fLowerLimits[i + 2] = 0;
        fUpperLimits[i + 2] = fDivisions;
        fDimensionSize[i + 2] = fDivisions;
    }

    fTotalSpatialSize = KFMArrayMath::TotalArraySize<KFMELECTROSTATICS_DIM>(&(fDimensionSize[2]));

    fKernelResponse->SetLowerSpatialLimits(&(fLowerLimits[2]));
    fKernelResponse->SetUpperSpatialLimits(&(fUpperLimits[2]));

    //set the source origin here...the position of the source
    //origin should be measured with respect to the center of the child node that is
    //indexed by (0,0,0), spacing between child nodes should be equal to 1.0
    //scaling for various tree levels is handled elsewhere

    double source_origin[KFMELECTROSTATICS_DIM];

    for (unsigned int i = 0; i < KFMELECTROSTATICS_DIM; i++) {
        if (fDivisions % 2 == 0) {
            source_origin[i] = 0.5;
        }
        else {
            source_origin[i] = 0.0;
        }
    }

    int shift[KFMELECTROSTATICS_DIM];
    for (unsigned int i = 0; i < KFMELECTROSTATICS_DIM; i++) {
        shift[i] = -1 * (std::ceil(1.0 * (((double) fDivisions) / 2.0)) - 1);
    }

    fKernelResponse->SetOrigin(source_origin);
    fKernelResponse->SetShift(shift);


    //allocate space and wrapper for L2L coeff
    fRawL2LCoeff.resize(fNTerms * fNTerms * fTotalSpatialSize);
    fL2LCoeff =
        new KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM + 2>(&(fRawL2LCoeff[0]), fDimensionSize);

    //here we need to initialize the L2L calculator
    //and fill the array of L2L coefficients
    fKernelResponse->SetZeroMaskSize(0);
    fKernelResponse->SetDistance(1.0);
    fKernelResponse->SetOutput(fL2LCoeff);
    fKernelResponse->Initialize();
    fKernelResponse->ExecuteOperation();

    //now compute the source and target scale factors
    fSourceScaleFactorArray.resize((fMaxTreeDepth + 1) * fStride);
    fTargetScaleFactorArray.resize((fMaxTreeDepth + 1) * fStride);

    //create the scale factor arrays
    //fill them with the scale factors
    double level_side_length = fWorldLength;
    double div_power;

    fScaleInvariantKernel =
        dynamic_cast<KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>*>(fKernelResponse->GetKernel());

    for (size_t level = 0; level <= fMaxTreeDepth; level++) {
        div_power = (double) fLowerLevelDivisions;

        if (level == 0) {
            div_power = 1.0;
        };
        if (level == 1) {
            div_power = (double) fTopLevelDivisions;
        };

        level_side_length /= div_power;

        //recompute the scale factors
        std::complex<double> factor(level_side_length, 0.0);
        for (unsigned int n = 0; n <= fDegree; n++) {
            for (unsigned int m = 0; m <= n; m++) {
                unsigned int csi = KFMScalarMultipoleExpansion::ComplexBasisIndex(n, m);
                unsigned int rsi = KFMScalarMultipoleExpansion::RealBasisIndex(n, m);

                std::complex<double> s;
                //compute the needed re-scaling for this tree level
                s = fScaleInvariantKernel->GetSourceScaleFactor(csi, factor);
                fSourceScaleFactorArray[level * fStride + rsi] = std::real(s);

                s = fScaleInvariantKernel->GetTargetScaleFactor(csi, factor);
                fTargetScaleFactorArray[level * fStride + rsi] = std::real(s);
            }
        }
    }


    ConstructTransformKernel();

    //set the max number of buffered nodes
    unsigned int n_children = std::pow(fDivisions, KFMELECTROSTATICS_DIM);

    //must be at least this size
    fNMaxBufferedNodes = fNMaxParentNodes * n_children;
    //but should be the next largest multiple of the local workgroup size
    unsigned int nDummy = fNTransformLocal - (fNMaxBufferedNodes % fNTransformLocal);
    if (nDummy == fNTransformLocal) {
        nDummy = 0;
    };
    fNMaxBufferedNodes += nDummy;

    BuildBuffers();
    AssignBuffers();
}


////////////////////////////////////////////////////////////////////////
void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node)
{
    if (node != NULL && node->HasChildren() && node->GetLevel() != 0) {
        //check if this node is a member of the non-zero local coeff node set
        int special_id = fPrimaryNodes->GetSpecializedIDFromOrdinaryID(node->GetID());

        if (special_id != -1) {
            //get the number of children which are primary nodes
            unsigned int n_moment_sets = fCachedPrimaryNodeIDLists[special_id].size();

            //this nodes contains children that have non-zero multipole moments
            //if it is at the same tree level, and fits, we add it to the buffer
            int node_level = node->GetLevel();

            if (node_level != fCachedNodeLevel) {
                //we have entered a new tree level we need to execute the kernel and clear
                //the buffered nodes, before adding this node to the buffer
                //because the next batch will need to use different scale factors
                ExecuteBufferedAction();
                ClearBuffers();
                BufferNode(node);
                fCachedNodeLevel = node_level;
            }
            else {
                //now determine if we can fit the node into the current buffer
                if (fNBufferedNodes + n_moment_sets < fNMaxBufferedNodes) {
                    //all we have to do is buffer this node
                    BufferNode(node);
                }
                else {
                    //execute action on already buffered nodes
                    ExecuteBufferedAction();
                    ClearBuffers();
                    BufferNode(node);
                }
            }
        }
    }
}


void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::BufferNode(KFMNode<KFMElectrostaticNodeObjects>* node)
{
    //get special multipole node set
    int special_id = fPrimaryNodes->GetSpecializedIDFromOrdinaryID(node->GetID());

    //get the number of children which have non-zero multipole moments
    unsigned int n_moment_sets = fCachedPrimaryNodeIDLists[special_id].size();

    //fill in the node ids, block set ids, and parent ids for all nodes with moments sets
    for (unsigned int i = 0; i < n_moment_sets; i++) {
        fNodeIDBuffer.push_back(fCachedPrimaryNodeIDLists[special_id][i]);
        fBlockSetIDBuffer.push_back(fCachedBlockSetIDLists[special_id][i]);
        fParentIDBuffer.push_back(special_id);
        fNBufferedNodes++;
    }
}

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::ExecuteBufferedAction()
{
    if (fNBufferedNodes != 0) {
        fTransformKernel->setArg(0, fNBufferedNodes);       //n node
        fTransformKernel->setArg(1, fCachedNodeLevel + 1);  //tree level

        //write out the node id's
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIDBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       fNBufferedNodes * sizeof(unsigned int),
                                                                       &(fNodeIDBuffer[0]));

        //write out the block set id's
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBlockSetIDListBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       fNBufferedNodes * sizeof(unsigned int),
                                                                       &(fBlockSetIDBuffer[0]));

        //write out the parent ids
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentIDBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       fNBufferedNodes * sizeof(unsigned int),
                                                                       &(fParentIDBuffer[0]));

        unsigned int nDummy;
        unsigned int nLocal;
        unsigned int nGlobal;

        //run the transformation kernel
        nLocal = fNTransformLocal;
        nGlobal = fNBufferedNodes * fStride;
        nDummy = nLocal - (nGlobal % nLocal);
        if (nDummy == nLocal) {
            nDummy = 0;
        };
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fTransformKernel,
                                                                         cl::NullRange,
                                                                         cl::NDRange(nGlobal + nDummy),
                                                                         cl::NDRange(nLocal));
    }
}

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::ClearBuffers()
{
    fNBufferedNodes = 0;
    fNodeIDBuffer.clear();
    fBlockSetIDBuffer.clear();
    fParentIDBuffer.clear();
}

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::BuildBuffers()
{

    //    fL2LCoeffBufferCL; !!!!
    //    fSourceScaleFactorBufferCL; !!!!
    //    fTargetScaleFactorBufferCL; !!!!
    //    fNodeMomentBufferCL; //not created here
    //    fNodeIDBufferCL; !!!!
    //    fBlockSetIDListBufferCL; !!!!
    //    fParentIDBufferCL;

    size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

    //size of the response functions
    size_t L2L_size = fNTerms * fNTerms * fTotalSpatialSize;

    if (L2L_size * sizeof(CL_TYPE2) > max_buffer_size) {
        //TODO: add special handling if L2L coeff are too big to fit on the GPU
        kfmout << "KFMScalarMomentLocalToLocalConverter::BuildBuffers: Error. Cannot allocated buffer of size: "
               << L2L_size * sizeof(CL_TYPE2) << " on device with max allowable buffer size of: " << max_buffer_size
               << std::endl
               << kfmendl;
        kfmexit(1);
    }

    //create the L2L buffer
    CL_ERROR_TRY
    {
        fL2LCoeffBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                           CL_MEM_READ_ONLY,
                                           L2L_size * sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //write the L2L coefficients to the GPU
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fL2LCoeffBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   L2L_size * sizeof(CL_TYPE2),
                                                                   &(fRawL2LCoeff[0]));

    //no longer need a host side copy of the L2L coeff, swap with empty vector
    std::vector<std::complex<double>> temp;
    temp.swap(fRawL2LCoeff);

    //create the scale factor buffers
    size_t sf_size = ((fMaxTreeDepth + 1) * fStride);
    CL_ERROR_TRY
    {
        fSourceScaleFactorBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sf_size * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fTargetScaleFactorBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sf_size * sizeof(CL_TYPE));
    }
    CL_ERROR_CATCH

    //write the scale factors to the gpu
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSourceScaleFactorBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sf_size * sizeof(CL_TYPE),
                                                                   &(fSourceScaleFactorArray[0]));
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fTargetScaleFactorBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   sf_size * sizeof(CL_TYPE),
                                                                   &(fTargetScaleFactorArray[0]));

    //create the node id list buffer
    CL_ERROR_TRY
    {
        fNodeIDBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                         CL_MEM_READ_ONLY,
                                         fNMaxBufferedNodes * sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    //create the block set id list buffer
    CL_ERROR_TRY
    {
        fBlockSetIDListBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                 CL_MEM_READ_ONLY,
                                                 fNMaxBufferedNodes * sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    //create the parent id buffer
    CL_ERROR_TRY
    {
        fParentIDBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                           CL_MEM_READ_ONLY,
                                           fNMaxBufferedNodes * sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    std::vector<unsigned int> zero_buff;
    zero_buff.resize(fNMaxBufferedNodes, 0);

    //write out zeros to the id buffers
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIDBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNMaxBufferedNodes * sizeof(unsigned int),
                                                                   &(zero_buff[0]));

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBlockSetIDListBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNMaxBufferedNodes * sizeof(unsigned int),
                                                                   &(zero_buff[0]));

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentIDBufferCL,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fNMaxBufferedNodes * sizeof(unsigned int),
                                                                   &(zero_buff[0]));
}

////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::AssignBuffers()
{

    //ElectrostaticBatchedLocalToLocalTransformation(const unsigned int n_nodes,
    //                                        const unsigned int tree_level,
    //                                        __constant const CL_TYPE* source_scale_factor_array,
    //                                        __constant const CL_TYPE* target_scale_factor_array,
    //                                        __global CL_TYPE2* response_functions,
    //                                        __global unsigned int* node_ids,
    //                                        __global unsigned int* block_set_ids,
    //                                        __global unsigned int* parent_ids,
    //                                        __global CL_TYPE2* node_moments)


    fTransformKernel->setArg(0, 0);  //must be reset before running the kernel
    fTransformKernel->setArg(1, 0);  //must be reset before running the kernel
    fTransformKernel->setArg(2, *fSourceScaleFactorBufferCL);
    fTransformKernel->setArg(3, *fTargetScaleFactorBufferCL);
    fTransformKernel->setArg(4, *fL2LCoeffBufferCL);
    fTransformKernel->setArg(5, *fNodeIDBufferCL);
    fTransformKernel->setArg(6, *fBlockSetIDListBufferCL);
    fTransformKernel->setArg(7, *fParentIDBufferCL);
    fTransformKernel->setArg(8, *fNodeMomentBufferCL);
}

////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticBatchedLocalToLocalConverter_OpenCL::ConstructTransformKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath()
           << "/kEMField_KFMElectrostaticBatchedLocalToLocalTransformation_kernel.cl";

    //create the build flags
    std::stringstream ss;
    ss << " -D KFM_DEGREE=" << fDegree;
    ss << " -D KFM_REAL_STRIDE=" << fStride;
    ss << " -D KFM_COMPLEX_STRIDE=" << (fDegree + 1) * (fDegree + 1);
    ss << " -D KFM_SPATIAL_STRIDE=" << fTotalSpatialSize;

    std::string build_flags = ss.str();

    KOpenCLKernelBuilder k_builder;
    fTransformKernel =
        k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticBatchedLocalToLocalTransformation"), build_flags);

    //get n-local
    fNTransformLocal =
        fTransformKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    unsigned int preferredWorkgroupMultiple =
        fTransformKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            KOpenCLInterface::GetInstance()->GetDevice());

    if (preferredWorkgroupMultiple < fNTransformLocal) {
        fNTransformLocal = preferredWorkgroupMultiple;
    }
}


////////////////////////////////////////////////////////////////////////////////


}  // namespace KEMField
