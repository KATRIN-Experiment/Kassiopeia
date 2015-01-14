#ifndef KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL_H__
#define KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL_H__

#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

#include "KFMNodeCollector.hh"

#include "KFMReducedScalarMomentRemoteToLocalConverter.hh"
#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"

#include "KFMSpecialNodeSet.hh"
#include "KFMSpecialNodeSetCreator.hh"
#include "KFMNodeFlagValueInspector.hh"



namespace KEMField{

/**
*
*@file KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
*@class KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, typename SourceScalarMomentType, typename TargetScalarMomentType, typename KernelType, size_t SpatialNDIM, size_t NFLAGS >
class KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL:
public KFMReducedScalarMomentRemoteToLocalConverter<ObjectTypeList, SourceScalarMomentType, TargetScalarMomentType, KernelType, SpatialNDIM >
{
    public:

        typedef KFMReducedScalarMomentRemoteToLocalConverter<ObjectTypeList, SourceScalarMomentType, TargetScalarMomentType, KernelType, SpatialNDIM >
        KFMReducedScalarMomentRemoteToLocalConverterBaseType;

        KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL():
            KFMReducedScalarMomentRemoteToLocalConverterBaseType(),
            fConvolveKernel(NULL),
            fScaleKernel(NULL),
            fSparseAddKernel(NULL),
            fM2LCoeffBufferCL(NULL),
            fWorkspaceBufferCL(NULL),
            fSourceScaleFactorArray(NULL),
            fTargetScaleFactorArray(NULL),
            fSourceScaleFactorBufferCL(NULL),
            fTargetScaleFactorBufferCL(NULL),
            fNormalizationCoeffBufferCL(NULL),
            fReversedIndexArrayBufferCL(NULL),
            fReversedIndexArray(NULL),
            fNPrimaryNodes(0),
            fPrimaryLocalCoeffBufferCL(NULL),
            fNodeIDListBufferCL(NULL),
            fBlockSetIDListBufferCL(NULL)
        {
            this->fDFTCalcOpenCL = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<SpatialNDIM>();
            fHaveIntializedSparseAdd = false;
        }

        virtual ~KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL()
        {
            delete fDFTCalcOpenCL;
            delete fConvolveKernel;
            delete fScaleKernel;
            delete fSparseAddKernel;

            delete fM2LCoeffBufferCL;
            delete fWorkspaceBufferCL;
            delete fNormalizationCoeffBufferCL;

            delete[] fSourceScaleFactorArray;
            delete[] fTargetScaleFactorArray;
            delete fSourceScaleFactorBufferCL;
            delete fTargetScaleFactorBufferCL;
            delete fReversedIndexArrayBufferCL;
            delete[] fReversedIndexArray;

            delete fPrimaryLocalCoeffBufferCL;
            delete fNodeIDListBufferCL;
            delete fBlockSetIDListBufferCL;

            //delete the cached primary node indices
            for(unsigned int i=0; i<fCachedPrimaryNodeLists.size(); i++)
            {
                delete fCachedPrimaryNodeLists[i];
            }

            for(unsigned int i=0; i<fCachedBlockSetIDLists.size(); i++)
            {
                delete fCachedBlockSetIDLists[i];
            }
        }

        virtual void Prepare(KFMCubicSpaceTree<SpatialNDIM, ObjectTypeList>* tree)
        {
            if(!fHaveIntializedSparseAdd)
            {


                KFMCubicSpaceTreeProperties<SpatialNDIM>* tree_prop = tree->GetTreeProperties();
                unsigned int n_nodes = tree_prop->GetNNodes();

                KFMSpecialNodeSet<ObjectTypeList> primaryNodes;
                primaryNodes.SetTotalNumberOfNodes(n_nodes);

                //flag inspector determines if a node is primary or not
                KFMNodeFlagValueInspector<ObjectTypeList, NFLAGS> primary_flag_condition;
                primary_flag_condition.SetFlagIndex(0);
                primary_flag_condition.SetFlagValue(1);

                KFMSpecialNodeSetCreator<ObjectTypeList> set_creator;
                set_creator.SetSpecialNodeSet(&primaryNodes);

                //now we constuct the conditional actor
                KFMConditionalActor< KFMNode<ObjectTypeList> > conditional_actor;

                conditional_actor.SetInspectingActor(&primary_flag_condition);
                conditional_actor.SetOperationalActor(&set_creator);

                tree->ApplyCorecursiveAction(&conditional_actor);

                fPrimaryNodeLookUpTable.resize(n_nodes);
                fPrimaryNodeReverseLookUpTable.resize(primaryNodes.GetSize());

                fNPrimaryNodes = primaryNodes.GetSize();
                for(unsigned int i=0; i<n_nodes; i++)
                {
                    int index = primaryNodes.GetSpecializedIDFromOrdinaryID(i);
                    fPrimaryNodeLookUpTable[i] = index;
                    if( index != -1)
                    {
                        fPrimaryNodeReverseLookUpTable[index] = i;
                    }

//                    if( tree_prop->GetNodePrimaryStatus(i) )
//                    {
//                        fPrimaryNodeLookUpTable[i] = fNPrimaryNodes; //node id to primary id
//                        fPrimaryNodeReverseLookUpTable.push_back(i); //primary id to node id
//                        fNPrimaryNodes++;
//                    }
//                    else
//                    {
//                        fPrimaryNodeLookUpTable[i] = -1;
//                    }
                }

                //find the pointers to the primary nodes
                KFMNodeCollector<ObjectTypeList> node_collector;
                node_collector.SetListOfNodeIDs(&fPrimaryNodeReverseLookUpTable);
                tree->ApplyRecursiveAction(&node_collector);
                node_collector.GetNodeList(&fPrimaryNodes);

                //create space to cache the primary node ids
                fCachedPrimaryNodeLists.resize(n_nodes, NULL);
                fCachedBlockSetIDLists.resize(n_nodes, NULL);

                BuildSparseAddBuffers();
                AssignSparseAddBuffers();

                fHaveIntializedSparseAdd = true;
            }

            if(this->fInitialized)
            {
                //reset all primary node local coeff to zero and write to device
                unsigned int primary_size = fPrimaryLocalCoeff.size();

                for(unsigned int i=0; i<primary_size; i++)
                {
                    fPrimaryLocalCoeff[i] = std::complex<double>(0.,0.);
                }

                //write zeros out to the gpu
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fPrimaryLocalCoeffBufferCL, CL_TRUE, 0, primary_size*sizeof(CL_TYPE2), &(fPrimaryLocalCoeff[0]) );
            }

        };


        //this function is called after visiting the tree to finalize the tree state if needed
        virtual void Finalize(KFMCubicSpaceTree<SpatialNDIM, ObjectTypeList>* /*tree*/)
        {
            //read the primary node local coefficients back from the gpu;
            unsigned int primary_size = fNPrimaryNodes*(this->fNReducedTerms);
            KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fPrimaryLocalCoeffBufferCL, CL_TRUE, 0, primary_size*sizeof(CL_TYPE2), &(fPrimaryLocalCoeff[0]) );

            //now distribute the primary node moments
            for(unsigned int i=0; i<fNPrimaryNodes; i++)
            {
                KFMNode<ObjectTypeList>* node = fPrimaryNodes[i];
                TargetScalarMomentType* set = KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(node);
                if(set != NULL)
                {
                    std::complex<double> temp;
                    //we use raw ptr for speed
                    double* rmoments = &( (*(set->GetRealMoments()))[0] );
                    double* imoments = &( (*(set->GetImaginaryMoments()))[0] );
                    for(unsigned int j=0; j < this->fNReducedTerms; ++j)
                    {
                        temp = fPrimaryLocalCoeff[i*(this->fNReducedTerms) + j];
                        rmoments[j] = temp.real();
                        imoments[j] = temp.imag();
                    }
                }
            }
        };


        void SetPrimaryNodeIdLookUpTable(std::vector<unsigned int>* primary_look_up)
        {
            fPrimaryNodeLookUpTable = *primary_look_up;
        }


        virtual void Initialize()
        {
            if( !(this->fInitialized) )
            {
                if(this->fNReducedTerms != 0 && this->fDim != 0)
                {
                    CheckDeviceProperites();
                    KFMReducedScalarMomentRemoteToLocalConverterBaseType::Initialize(); //m2l coeff are calculated here


                    //intialize DFT calculator for array dimensions
                    fDFTCalcOpenCL->SetInput(this->fAllMultipoles);
                    fDFTCalcOpenCL->SetOutput(this->fAllLocalCoeff);

                    //all enqueue read/write buffers occur external to the DFT kernel execution
                    fDFTCalcOpenCL->Initialize();

                    //fill reversed array look up table
                    fReversedIndexArray = new unsigned int[this->fTotalSpatialSize];
                    this->fConjMultCalc->GetReversedIndexArray(fReversedIndexArray);

                    ConstructConvolutionKernel();
                    ConstructScaleKernel();
                    ConstructSparseAddKernel();
                    BuildBuffers();
                    AssignBuffers();
                }

                this->fInitialized = true;
            }
        }

        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                if( node->HasChildren() && this->ChildrenHaveNonZeroMoments(node) )
                {

                    //TODO! LEAVE MULTIPOLES ON GPU IN BUFFER
                    //collect the multipoles
                    this->fCollector->ApplyAction(node);

                    EnqueueWriteMultipoleMoments();

                    //if we have a scale invariant kernel
                    //factor the local coefficients depending on the tree level
                    if(this->fIsScaleInvariant)
                    {
                        unsigned int level = node->GetLevel() + 1;
                        fScaleKernel->setArg(3, level);
                        RescaleMultipoleMoments();
                    }

                    //convolve the multipoles with the response functions to get local coeff
                    Convolve();

                    if(this->fIsScaleInvariant)
                    {
                        RescaleLocalCoefficients();
                    }

                    CollectPrimaryNodeIdList(node);

                    ApplySparseAdditionKernel();
                }
            }
        }


    protected:

////////////////////////////////////////////////////////////////////////////////

        virtual void Convolve()
        {
            //first perform the forward dft on all the multipole coefficients
            fDFTCalcOpenCL->SetForward();
            fDFTCalcOpenCL->SetWriteOutHostDataFalse();
            fDFTCalcOpenCL->SetReadOutDataToHostFalse();
            fDFTCalcOpenCL->SetInput(this->fAllMultipoles);
            fDFTCalcOpenCL->SetOutput(this->fAllMultipoles);
            fDFTCalcOpenCL->ExecuteOperation();

            PointwiseMultiplyAndAddOpenCL();

            //now perform an inverse DFT on the x-formed local
            //coefficients to get the actual local coeff
            fDFTCalcOpenCL->SetWriteOutHostDataFalse();
            fDFTCalcOpenCL->SetReadOutDataToHostFalse();
            fDFTCalcOpenCL->SetInput(this->fAllLocalCoeff);
            fDFTCalcOpenCL->SetOutput(this->fAllLocalCoeff);
            fDFTCalcOpenCL->SetBackward();
            fDFTCalcOpenCL->ExecuteOperation();
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void BuildBuffers()
        {
            //size of the response functions
            size_t m2l_size = (this->fNResponseTerms)*(this->fTotalSpatialSize);

            CheckDeviceProperites();

            //create the m2l buffer
            fM2LCoeffBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, m2l_size*sizeof(CL_TYPE2));

            //write the M2L coefficients to the GPU
            size_t elements_in_buffer = (this->fNResponseTerms)*(this->fTotalSpatialSize);
            std::complex<double>* m2lptr = this->fPtrM2LCoeff;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fM2LCoeffBufferCL, CL_TRUE, 0, elements_in_buffer*sizeof(CL_TYPE2), m2lptr);

            //create the buffer for the normalization coefficients
            size_t norm_size = (this->fNTerms)*(this->fNTerms);
            this->fNormalizationCoeffBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, norm_size*sizeof(CL_TYPE2));

            //write the buffer containing the normalization coefficients
            std::complex<double>* ptr = this->fPtrNormalization;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNormalizationCoeffBufferCL, CL_TRUE, 0, norm_size*sizeof(CL_TYPE2), ptr);

            //size of the workspace on the gpu
            size_t workspace_size = (this->fNReducedTerms)*(this->fTotalSpatialSize);

            //create the workspace buffer
            this->fWorkspaceBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, workspace_size*sizeof(CL_TYPE2));

            //get the pointer to the DFT calculators GPU data buffer
            //we will use this to directly fill the buffer with the multipoles, and local coefficients
            //for FFTs while it is still on the GPU
            this->fFFTDataBufferCL = this->fDFTCalcOpenCL->GetDataBuffer();

            //create the reversed index look-up buffer
            fReversedIndexArrayBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, (this->fTotalSpatialSize)*sizeof(unsigned int));

            //fill the reversed index buffer
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fReversedIndexArrayBufferCL, CL_TRUE, 0, (this->fTotalSpatialSize)*sizeof(unsigned int), fReversedIndexArray);

            if(this->fIsScaleInvariant)
            {
                //create the scale factor arrays
                size_t sf_size = (this->fMaxTreeDepth + 1)*(this->fNReducedTerms);
                fSourceScaleFactorArray = new CL_TYPE[sf_size];
                fTargetScaleFactorArray = new CL_TYPE[sf_size];

                //fill them with the scale factors
                double level_side_length;
                double div_power;
                for(size_t level = 0; level <= this->fMaxTreeDepth; level++)
                {
                    div_power = std::pow( (double)(this->fDiv), level);
                    level_side_length = this->fLength/div_power;

                    //recompute the scale factors
                    std::complex<double> factor(level_side_length, 0.0);
                    for(int n=0; n <= this->fDegree; n++)
                    {
                        for(int m=0; m<=n; m++)
                        {
                            unsigned int csi = KFMScalarMultipoleExpansion::ComplexBasisIndex(n,m);
                            unsigned int rsi = KFMScalarMultipoleExpansion::RealBasisIndex(n,m);

                            std::complex<double> s;
                            //compute the needed re-scaling for this tree level
                            s = this->fScaleInvariantKernel->GetSourceScaleFactor(csi, factor );
                            fSourceScaleFactorArray[level*(this->fNReducedTerms) + rsi] = std::real(s);

                            s = this->fScaleInvariantKernel->GetTargetScaleFactor(csi, factor );
                            fTargetScaleFactorArray[level*(this->fNReducedTerms) + rsi] = std::real(s);
                        }
                    }
                }

                //create the scale factor buffers
                fSourceScaleFactorBufferCL
                = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sf_size*sizeof(CL_TYPE));

                fTargetScaleFactorBufferCL
                = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sf_size*sizeof(CL_TYPE));

                //write the scale factors to the gpu
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSourceScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), fSourceScaleFactorArray);

                //write the scale factors to the gpu
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fTargetScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), fTargetScaleFactorArray);



            }
        }

        void BuildSparseAddBuffers()
        {
            //create the primary local coefficients buffer
            unsigned int primary_size = fNPrimaryNodes*(this->fNReducedTerms);
            fPrimaryLocalCoeffBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, primary_size*sizeof(CL_TYPE2));
            //create space to read out the primary node local coeff buffers
            fPrimaryLocalCoeff.resize(primary_size);

            //create the buffer for the block-set's primary node ids
            fNodeIDListBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, (this->fTotalSpatialSize)*sizeof(unsigned int));
            //host space for the node ids
            fNodeIDList.resize(this->fTotalSpatialSize);

            //create the buffer of the valid nodes ids in the current block set
            fBlockSetIDListBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, (this->fTotalSpatialSize)*sizeof(unsigned int));
            //host space for the block set ids
            fBlockSetIDList.resize(this->fTotalSpatialSize);
        }

        void AssignSparseAddBuffers()
        {
            fSparseAddKernel->setArg(0, this->fTotalSpatialSize);
            fSparseAddKernel->setArg(1, this->fTotalSpatialSize);
            fSparseAddKernel->setArg(2, (unsigned int) this->fNReducedTerms);
            fSparseAddKernel->setArg(3, *fBlockSetIDListBufferCL);
            fSparseAddKernel->setArg(4, *fNodeIDListBufferCL);
            fSparseAddKernel->setArg(5, *fFFTDataBufferCL);
            fSparseAddKernel->setArg(6, *fPrimaryLocalCoeffBufferCL);
        }

////////////////////////////////////////////////////////////////////////////////

        void RescaleMultipoleMoments()
        {
            //if we have a scale invariant kernel, once we have computed the kernel reponse array once
            //we only have to re-scale the moments, we don't have to recompute the array at each tree level
            //any recomputation of the kernel reponse array for non-invariant kernels must be managed by an external class

            //compute size of the array
            unsigned int n_global = (this->fNReducedTerms)*(this->fTotalSpatialSize);

            //rescale the multipoles
            //pad out n-global to be a multiple of the n-local
            unsigned int nDummy = fNScaleLocal - (n_global%fNScaleLocal);
            if(nDummy == fNScaleLocal){nDummy = 0;};
            n_global += nDummy;
            cl::NDRange global(n_global);
            cl::NDRange local(fNScaleLocal);

            //set the scale factor argument
            fScaleKernel->setArg(4, *fSourceScaleFactorBufferCL);
            //now enqueue the kernel
            cl::Event scale_event;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fScaleKernel, cl::NullRange, global, local, NULL, &scale_event);
            scale_event.wait();
        }

////////////////////////////////////////////////////////////////////////////////

        void RescaleLocalCoefficients()
        {
            //compute size of the array
            unsigned int n_global = (this->fNReducedTerms)*(this->fTotalSpatialSize);

            //pad out n-global to be a multiple of the n-local
            unsigned int nDummy = fNScaleLocal - (n_global%fNScaleLocal);
            if(nDummy == fNScaleLocal){nDummy = 0;};
            n_global += nDummy;
            cl::NDRange global(n_global);
            cl::NDRange local(fNScaleLocal);

            //set scale factor argument
            fScaleKernel->setArg(4, *fTargetScaleFactorBufferCL);
            //now enqueue the kernel
            cl::Event scale_event;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fScaleKernel, cl::NullRange, global, local, NULL, &scale_event);
            scale_event.wait();
        }

////////////////////////////////////////////////////////////////////////////////

        void ApplySparseAdditionKernel()
        {
            //set the number of nodes we are processing
            fSparseAddKernel->setArg(0, fNPrimaryNodesCollected);

            //enqueue write the primary node ids
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIDListBufferCL, CL_TRUE, 0, fNPrimaryNodesCollected*sizeof(unsigned int), &(fNodeIDList[0]) );

            //enqueue write the block set ids
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBlockSetIDListBufferCL, CL_TRUE, 0, fNPrimaryNodesCollected*sizeof(unsigned int), &(fBlockSetIDList[0]) );

            //enqueue the sparse add kernel
            //pad out n-global to be a multiple of the n-local
            unsigned int n_sparse_global = fNPrimaryNodesCollected;
            unsigned int nDummy = fNSparseAddLocal - (n_sparse_global%fNSparseAddLocal);
            if(nDummy == fNSparseAddLocal){nDummy = 0;};
            n_sparse_global += nDummy;
            cl::NDRange global(n_sparse_global);
            cl::NDRange local(fNSparseAddLocal);

            cl::Event sparse_event;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fSparseAddKernel, cl::NullRange, global, local, NULL, &sparse_event);
            sparse_event.wait();
        }

////////////////////////////////////////////////////////////////////////////////

        void PointwiseMultiplyAndAddOpenCL()
        {
            //compute size of the array
            unsigned int array_size = (this->fNReducedTerms)*(this->fTotalSpatialSize);
            unsigned int n_global = array_size;

            //pad out n-global to be a multiple of the n-local
            unsigned int nDummy = fNConvolveLocal - (n_global%fNConvolveLocal);
            if(nDummy == fNConvolveLocal){nDummy = 0;};
            n_global += nDummy;

            cl::NDRange global(n_global);
            cl::NDRange local(fNConvolveLocal);

            cl::Event event;
            KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fConvolveKernel, cl::NullRange, global, local, NULL, &event);
            event.wait();

            //now copy the workspace data (pre-x-formed local) into the FFT buffer to perform inverse DFT
            KOpenCLInterface::GetInstance()->GetQueue().enqueueCopyBuffer(*fWorkspaceBufferCL, *fFFTDataBufferCL, size_t(0), size_t(0), array_size*sizeof(CL_TYPE2) );

        }


        void AssignBuffers()
        {
            //set the array size
            unsigned int array_size = (this->fNReducedTerms)*(this->fTotalSpatialSize);
            fConvolveKernel->setArg(0, array_size);

            //assign the expansion degree
            unsigned int deg = this->fDegree;
            fConvolveKernel->setArg(1, deg);

            //assign the array's spatial stride
            unsigned int spatial_stride = this->fTotalSpatialSize;
            fConvolveKernel->setArg(2, spatial_stride);

            //assign the remote moments buffer
            fConvolveKernel->setArg(3, *fFFTDataBufferCL);

            //set appropriate response function buffer
            fConvolveKernel->setArg(4, *fM2LCoeffBufferCL);

            //assign the local moment buffer
            fConvolveKernel->setArg(5, *fWorkspaceBufferCL);

            //assign the normalization buffer
            fConvolveKernel->setArg(6, *fNormalizationCoeffBufferCL);

            //assign the reversed index look up array buffer
            fConvolveKernel->setArg(7, *fReversedIndexArrayBufferCL);


            fScaleKernel->setArg(0, array_size );
            fScaleKernel->setArg(1, spatial_stride);
            fScaleKernel->setArg(2, (unsigned int) this->fNReducedTerms);
            fScaleKernel->setArg(3, (unsigned int) this->fMaxTreeDepth);
            fScaleKernel->setArg(4, *fSourceScaleFactorBufferCL);
            fScaleKernel->setArg(5, *fFFTDataBufferCL);
        }

        virtual void ConstructConvolutionKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMReducedScalarMomentRemoteToLocalConverter_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fConvolveKernel = k_builder.BuildKernel(clFile.str(), std::string("ReducedScalarMomentRemoteToLocalConverter") );

            //get n-local
            fNConvolveLocal = fConvolveKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void ConstructScaleKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMScalarMomentApplyScaleFactor_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fScaleKernel = k_builder.BuildKernel(clFile.str(), std::string("ScalarMomentApplyScaleFactor") );

            //get n-local
            fNScaleLocal = fScaleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void ConstructSparseAddKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMSparseScalarMomentAdd_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fSparseAddKernel = k_builder.BuildKernel(clFile.str(), std::string("SparseScalarMomentAdd") );

            //get n-local
            fNSparseAddLocal = fSparseAddKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
        }

////////////////////////////////////////////////////////////////////////////////

        void EnqueueWriteMultipoleMoments()
        {
            size_t moment_size = (this->fNReducedTerms)*(this->fTotalSpatialSize);
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fFFTDataBufferCL, CL_TRUE, 0, moment_size*sizeof(CL_TYPE2), this->fPtrMultipoles);
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void CheckDeviceProperites()
        {
            size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            size_t total_mem_size =  KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

            //size of the response functions
            size_t m2l_size = (this->fNResponseTerms)*(this->fTotalSpatialSize);

            if(m2l_size*sizeof(CL_TYPE2) > max_buffer_size)
            {
                //we cannot fit response_functions entirely on the gpu
                //even if we use multiple buffers
                size_t size_to_alloc_mb = ( m2l_size*sizeof(CL_TYPE2) )/(1024*1024);
                size_t max_size_mb = max_buffer_size/(1024*1024);
                size_t total_size_mb = total_mem_size/(1024*1024);

                kfmout<<"KFMReducedScalarMomentRemoteToLocalConverter::BuildBuffers: Error. Cannot allocate buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
                kfmexit(1);
            }
        }

        void CollectPrimaryNodeIdList(KFMNode<ObjectTypeList>* node)
        {
            //first see if we have the primary node ids cached;
            unsigned int node_id = node->GetID();
            if( fCachedPrimaryNodeLists[node_id] != NULL && fCachedBlockSetIDLists[node_id] != NULL )
            {
                fNPrimaryNodesCollected = fCachedPrimaryNodeLists[node_id]->size();

                for(unsigned int i=0; i<fNPrimaryNodesCollected; i++)
                {
                    fNodeIDList[i] = (*fCachedPrimaryNodeLists[node_id])[i];
                    fBlockSetIDList[i] = (*fCachedBlockSetIDLists[node_id])[i];

                }
                return;
            }

            //set number of primary nodes collected to zero
            fNPrimaryNodesCollected = 0;

            //set all primary nodes ids to 0
            for(unsigned int i=0; i<this->fTotalSpatialSize; i++){fNodeIDList[i] = 0;};

            unsigned int szpn[SpatialNDIM]; //parent neighbor spatial index
            unsigned int sznc[SpatialNDIM]; //neighbor child spatial index (within neighbor)

            int pn[SpatialNDIM]; //parent neighbor spatial index (relative position to original node)
            int lc[SpatialNDIM]; //global position in local coefficient array of this child

            unsigned int offset; //offset due to spatial indices from beginning of local coefficient array of this child

            //get all neighbors of this node
            KFMCubicSpaceNodeNeighborFinder<SpatialNDIM, ObjectTypeList>::GetAllNeighbors(node, this->fZeroMaskSize, &(this->fNeighbors));

            for(unsigned int n=0; n<(this->fNeighbors.size()); n++)
            {
                if(this->fNeighbors[n] != NULL)
                {
                    //compute relative index of this neighbor and store in pn array
                    KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(n, this->fNeighborDimensionSize, szpn);
                    for(unsigned int i=0; i<SpatialNDIM; i++)
                    {
                        pn[i] = (int)szpn[i] - this->fZeroMaskSize;
                    }

                    //loop over neighbors children
                    unsigned int n_children = this->fNeighbors[n]->GetNChildren();
                    for(unsigned int c = 0; c < n_children; c++)
                    {
                        this->fChild = this->fNeighbors[n]->GetChild(c);
                        if(this->fChild != NULL)
                        {
                            //get child's id
                            unsigned int child_id = this->fChild->GetID();

                            //look up if this child is a primary node
                            int child_primary_node_id = fPrimaryNodeLookUpTable[child_id];

                            if(child_primary_node_id != -1)
                            {
                                //we have a primary node, write it's primary id to the list
                                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(c, this->fChildDimensionSize, sznc);

                                //spatial index of local coefficients for this child
                                for(unsigned int i=0; i<SpatialNDIM; i++)
                                {
                                    lc[i] = (pn[i])*(this->fDiv) + (int)sznc[i];
                                }

                                offset = this->fLocalCoeff[0]->GetOffsetForIndices(lc);

                                fNodeIDList[fNPrimaryNodesCollected] = child_primary_node_id;
                                fBlockSetIDList[fNPrimaryNodesCollected] = offset;
                                fNPrimaryNodesCollected++;
                            }
                        }
                    }
                }
            }


            if( fCachedPrimaryNodeLists[node_id] == NULL && fCachedBlockSetIDLists[node_id] == NULL )
            {
                fCachedPrimaryNodeLists[node_id] = new std::vector<unsigned int>();
                fCachedBlockSetIDLists[node_id] = new std::vector<unsigned int>();
            }

            fCachedPrimaryNodeLists[node_id]->resize(fNPrimaryNodesCollected);
            fCachedBlockSetIDLists[node_id]->resize(fNPrimaryNodesCollected);

            for(unsigned int i=0; i<fNPrimaryNodesCollected; i++)
            {
                (*fCachedPrimaryNodeLists[node_id])[i] = fNodeIDList[i];
                (*fCachedBlockSetIDLists[node_id])[i] = fBlockSetIDList[i];
            }

        }


////////////////////////////////////////////////////////////////////////////////

        KFMBatchedMultidimensionalFastFourierTransform_OpenCL<SpatialNDIM>* fDFTCalcOpenCL;

        mutable cl::Kernel* fConvolveKernel;
        unsigned int fNConvolveLocal;

        mutable cl::Kernel* fScaleKernel;
        unsigned int fNScaleLocal;

        mutable cl::Kernel* fSparseAddKernel;
        unsigned int fNSparseAddLocal;

        //need a buffer to store the M2L coefficients on the GPU
        cl::Buffer* fM2LCoeffBufferCL;

        //need a buffer to copy the multpole moments into,
        //and to read the local coefficients out from
        //this is the input buffer of the batched FFT calculator, it is not owned
        cl::Buffer* fFFTDataBufferCL; //must be p^2*total_spatial_size

        //need a buffer to store the local coefficients on the GPU
        //this is a temporary buffer that only needs to operated on by the GPU
        //we copy this buffer into the batched FFT calculators buffer before
        //the final FFT to obtain the local coefficients
        cl::Buffer* fWorkspaceBufferCL; //must be p^2*total_spatial_size

        //scale factor buffers for scale invariant kernels
        CL_TYPE* fSourceScaleFactorArray;
        CL_TYPE* fTargetScaleFactorArray;
        cl::Buffer* fSourceScaleFactorBufferCL;
        cl::Buffer* fTargetScaleFactorBufferCL;

        //need a buffer to store the normalization coefficients on the GPU
        cl::Buffer* fNormalizationCoeffBufferCL;

        //buffer to store the indices of a reversed look-up
        cl::Buffer* fReversedIndexArrayBufferCL;
        unsigned int* fReversedIndexArray;

        //buffer to store all of the local coefficients for all primary nodes
        unsigned int fNPrimaryNodes;
        cl::Buffer* fPrimaryLocalCoeffBufferCL;
        std::vector< std::complex<double> > fPrimaryLocalCoeff;

        //number of valid (primary) nodes we have collected
        unsigned int fNPrimaryNodesCollected;

        //buffer to store the 'primary node ids' of the current block set under processing
        cl::Buffer* fNodeIDListBufferCL;
        std::vector<unsigned int> fNodeIDList;

        //buffer to store the local block-set id of the valid (primary) nodes
        cl::Buffer* fBlockSetIDListBufferCL;
        std::vector<unsigned int> fBlockSetIDList;

        //space to cache the ids of the primary nodes adjacent to the node being processes
        //these are indexed by node id
        std::vector< std::vector<unsigned int>* > fCachedPrimaryNodeLists;
        std::vector< std::vector<unsigned int>* > fCachedBlockSetIDLists;


        //vector to look up node id -> primary node id
        std::vector<unsigned int> fPrimaryNodeLookUpTable;
        std::vector<unsigned int> fPrimaryNodeReverseLookUpTable;

        bool fHaveIntializedSparseAdd;

        //pointers to the primary nodes
        std::vector< KFMNode<ObjectTypeList>* > fPrimaryNodes;


////////////////////////////////////////////////////////////////////////////////



};

}//end of KEMField namespace


#endif /* __KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL_H__ */
