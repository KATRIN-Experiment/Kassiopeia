#ifndef __KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL_H__
#define __KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL_H__

#include <vector>
#include <complex>

//core
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

//kernel
#include "KFMKernelResponseArray.hh"
#include "KFMKernelExpansion.hh"
#include "KFMScaleInvariantKernelExpansion.hh"
#include "KFMKernelResponseArrayTypes.hh"
#include "KFMScalarMultipoleExpansion.hh"

//core (opencl)
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

//kernel
#include "KFMScalarMultipoleExpansion.hh"

//math
#include "KFMPointCloud.hh"
#include "KFMMath.hh"
#include "KFMCube.hh"

#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"

//tree
#include "KFMSpecialNodeSet.hh"

//electrostatics
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticElementContainer.hh"


namespace KEMField
{

/**
*
*@file KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL.hh
*@class KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug  4 11:48:24 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL: public KFMNodeActor< KFMElectrostaticNode >
{
    public:

        KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL();
        virtual ~KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL();

        //set the world volume length
        void SetLength(double length){fWorldLength = length;};

        //set the maximum depth of the tree
        void SetMaxTreeDepth(unsigned int max_depth){fMaxTreeDepth = max_depth;};

        void SetNumberOfTermsInSeries(unsigned int n_terms)
        {
            fNTerms = n_terms;
            KFMScalarMultipoleExpansion expan;
            expan.SetNumberOfTermsInSeries(fNTerms);
            fDegree = expan.GetDegree();
            fStride = (fDegree+1)*(fDegree+2)/2;
            fNResponseTerms = (2*fDegree+1)*(2*fDegree+1);
        };

        void SetZeroMaskSize(int zeromasksize)
        {
            fZeroMaskSize = zeromasksize;
        }

        void SetNeighborOrder(int neighbor_order)
        {
            fNeighborOrder = std::fabs(neighbor_order);
            fNeighborStride = 2*fNeighborOrder + 1;
            fDim = 2*fDivisions*(fNeighborOrder + 1);
            fTotalSpatialSize = std::pow(fDim, KFMELECTROSTATICS_DIM);
        }

        void SetDivisions(int div)
        {
            fDivisions = div;
            fDim = 2*fDivisions*(fNeighborOrder + 1);
            fNChildren = std::pow(fDivisions, KFMELECTROSTATICS_DIM);
            fTotalSpatialSize = std::pow(fDim, KFMELECTROSTATICS_DIM);
        }

//        void SetParameters(KFMElectrostaticParameters params);
        void SetTree(KFMElectrostaticTree* tree);

        void SetNodeRemoteMomentBuffer(cl::Buffer* node_remote_moments){fNodeRemoteMomentBufferCL = node_remote_moments;};
        void SetNodeLocalMomentBuffer(cl::Buffer* node_local_moments){fNodeLocalMomentBufferCL = node_local_moments;};

        void SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set);
        void SetPrimaryNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* local_node_set);

        ////////////////////////////////////////////////////////////////////////
        void Initialize();

        void Prepare();
        void Finalize();
        void ReadOutLocalCoefficients();

        ////////////////////////////////////////////////////////////////////////
        virtual void ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node);

        virtual bool IsFinished() const {return true;};

    protected:

        void ConstructCachedNodeIdentityLists();
        void CheckDeviceProperites();

        void BuildBuffers();
        void AssignBuffers();

        void ConstructCopyAndScaleKernel();
        void ConstructTransformationKernel();
        void ConstructReduceAndScaleKernel();
        void ConstructZeroComplexArrayKernel();

        void ApplyCopyAndScaleKernel();
        void ApplyTransformationKernel();
        void ApplyReduceAndScaleKernel();


        void BufferNode(KFMNode<KFMElectrostaticNodeObjects>* node);
        void ExecuteBufferedAction();
        void ClearBuffers();

////////////////////////////////////////////////////////////////////////////////

        KFMElectrostaticTree* fTree;

        //parameters extracted from tree
        unsigned int fDegree;
        unsigned int fNTerms;
        unsigned int fStride;
        unsigned int fNResponseTerms;
        unsigned int fDivisions;

        //needed when computing scale factors
        unsigned int fTopLevelDivisions;
        unsigned int fLowerLevelDivisions;

        unsigned int fNChildren;
        unsigned int fZeroMaskSize;
        unsigned int fNeighborOrder;
        unsigned int fMaxTreeDepth;
        unsigned int fDim; //2*fDivisions*(fNeighborOrder+1)
        unsigned int fTotalSpatialSize;
        unsigned int fNeighborStride; //2*fNeighborOrder + 1
        unsigned int fNMaxNeighbors;
        double fWorldLength;
        double fFFTNormalization;

        //limits
        unsigned int fSpatialSize[KFMELECTROSTATICS_DIM];
        unsigned int fNeighborDimensionSize[KFMELECTROSTATICS_DIM];
        unsigned int fChildDimensionSize[KFMELECTROSTATICS_DIM];

        //size and limits on local moments
        unsigned int fTargetDimensionSize[KFMELECTROSTATICS_DIM+ 1];
        int fTargetLowerLimits[KFMELECTROSTATICS_DIM+ 1];
        int fTargetUpperLimits[KFMELECTROSTATICS_DIM+ 1];

        //size and limits on multipole moment
        unsigned int fSourceDimensionSize[KFMELECTROSTATICS_DIM+ 1];
        int fSourceLowerLimits[KFMELECTROSTATICS_DIM+ 1];
        int fSourceUpperLimits[KFMELECTROSTATICS_DIM+ 1];

        //dimensions and limits on m2l coefficients
        unsigned int fResponseDimensionSize[KFMELECTROSTATICS_DIM+ 1];
        int fLowerResponseLimits[KFMELECTROSTATICS_DIM+ 1];
        int fUpperResponseLimits[KFMELECTROSTATICS_DIM+ 1];

        //kernel for computing m2l coeff
        KFMKernelReducedResponseArray_3DLaplaceM2L* fKernelResponse;
        KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>* fScaleInvariantKernel;

        //raw arrays to store data for m2l coefficients and normalization coefficients
        std::vector< std::complex<double> > fRawM2LCoeff;
        std::vector< std::complex<double> > fRawTransposedM2LCoeff;
        KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM + 1>* fAllM2LCoeff;
        std::vector< KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM>* > fM2LCoeff;
        KFMMultidimensionalFastFourierTransform< KFMELECTROSTATICS_DIM >* fDFTCalc; //for calculating transformed m2l coeff

        //helper wrapper for multipole/local coeff array manipulation
        std::vector< std::complex<double> > fRawHelperArray;
        unsigned int fHelperDimensionSize[KFMELECTROSTATICS_DIM+1];
        KFMArrayWrapper< std::complex<double>, KFMELECTROSTATICS_DIM + 1>* fHelperArrayWrapper;

        ////////////////////////////////////////////////////////////////////////
        int fCachedNodeLevel;
        unsigned int fNMaxBufferedNodes;
        unsigned int fNBufferedNodes;
        unsigned int fDefaultWorkSize;
        std::vector<unsigned int> fBufferedNodeSpecialIDs;

        //OpenCL kernels and buffers
        mutable cl::Kernel* fCopyAndScaleKernel;
        unsigned int fNCopyAndScaleLocal;

        mutable cl::Kernel* fTransformationKernel;
        unsigned int fNTransformationLocal;

        mutable cl::Kernel* fReduceAndScaleKernel;
        unsigned int fNReduceAndScaleLocal;

        mutable cl::Kernel* fZeroComplexArrayKernel;
        unsigned int fNZeroComplexArrayLocal;

        //scale factor buffers for scale invariant kernels
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
        std::vector< CL_TYPE > fSourceScaleFactorArray;
        std::vector< CL_TYPE > fTargetScaleFactorArray;
        #pragma GCC diagnostic pop
        cl::Buffer* fSourceScaleFactorBufferCL;
        cl::Buffer* fTargetScaleFactorBufferCL;

        //OpenCL FFT calculator
        KFMBatchedMultidimensionalFastFourierTransform_OpenCL<KFMELECTROSTATICS_DIM>* fDFTCalcOpenCL_Forward;
        KFMBatchedMultidimensionalFastFourierTransform_OpenCL<KFMELECTROSTATICS_DIM>* fDFTCalcOpenCL_Inverse;

        //need a buffer to store the M2L coefficients on the GPU
        cl::Buffer* fM2LCoeffBufferCL;

        //need a buffer to copy the multpole moments into,
        //and to read the local coefficients out from
        //this is the input buffer of the batched FFT calculator, it is not owned
        cl::Buffer* fFFTDataBufferCL; //must be p^2*total_spatial_size*n_buffered_nodes

        //need a buffer to store the local coefficients on the GPU
        //this is a temporary buffer that only needs to operated on by the GPU
        //we copy this buffer into the batched FFT calculators buffer before
        //the final FFT to obtain the local coefficients
        cl::Buffer* fWorkspaceBufferCL; //must be p^2*total_spatial_size*n_buffered_nodes

        //buffer for normalization coefficients
        cl::Buffer* fACoeffBufferCL;

        //buffer to store the indices of a reversed look-up
        cl::Buffer* fReversedIndexArrayBufferCL;
        std::vector<unsigned int> fReversedIndexArray;

        //node sets (multipole/primary)
        unsigned int fNMultipoleNodes;
        unsigned int fNPrimaryNodes;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fMultipoleNodes;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fPrimaryNodes;

        //external buffers to store all of the
        //multipole moments and local coefficients of all relevant nodes
        cl::Buffer* fNodeLocalMomentBufferCL;
        cl::Buffer* fNodeRemoteMomentBufferCL;

        //only used by the Finalize() command when debugging
        std::vector< std::complex<double> > fPrimaryLocalCoeff;

        //buffer to store the 'multipole node ids' of the current block set under processing
        cl::Buffer* fMultipoleNodeIDListBufferCL;
        std::vector<unsigned int> fMultipoleNodeIDList;
        std::vector<unsigned int> fMultipoleNodeIDListStartIndexes;

        cl::Buffer* fMultipoleBlockSetIDListBufferCL;
        std::vector<unsigned int> fMultipoleBlockSetIDList;
        std::vector<unsigned int> fMultipoleBlockSetIDListStartIndexes;
        //space to cache the ids of the child multipole nodes for each parent that needs processing
        //these are indexed by multipole node id
        std::vector< std::vector<unsigned int> > fCachedMultipoleNodeIDLists;
        std::vector< std::vector<unsigned int> > fCachedMultipoleBlockSetIDLists;
        std::vector<int> fMultipoleNodeIDBuffer;
        std::vector<unsigned int> fMultipoleBlockSetIDBuffer;

        //buffer to store the 'primary node ids' of the current block set under processing
        cl::Buffer* fPrimaryNodeIDListBufferCL;
        std::vector<unsigned int> fPrimaryNodeIDList;
        std::vector<unsigned int> fPrimaryNodeIDListStartIndexes;

        cl::Buffer* fPrimaryBlockSetIDListBufferCL;
        std::vector<unsigned int> fPrimaryBlockSetIDList;
        std::vector<unsigned int> fPrimaryBlockSetIDListStartIndexes;


        //space to cache the ids of the primary nodes adjacent to the node being processes
        //these are indexed by node id
        std::vector< std::vector<unsigned int> > fCachedPrimaryNodeIDLists;
        std::vector< std::vector<unsigned int> > fCachedPrimaryBlockSetIDLists;

////////////////////////////////////////////////////////////////////////////////


};



}//end of kemfield namespace

#endif /* __KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL_H__ */
