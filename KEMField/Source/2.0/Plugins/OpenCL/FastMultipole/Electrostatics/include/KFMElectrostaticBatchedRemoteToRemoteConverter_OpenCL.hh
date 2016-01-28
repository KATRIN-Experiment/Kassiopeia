#ifndef KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL_H__
#define KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL_H__


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

//core (opencl)
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

//kernel
#include "KFMScalarMultipoleExpansion.hh"

////math
//#include "KFMPointCloud.hh"
//#include "KFMMath.hh"
//#include "KFMCube.hh"


//tree
#include "KFMSpecialNodeSet.hh"

//electrostatics
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
//#include "KFMElectrostaticElementContainer.hh"

namespace KEMField{

/**
*
*@file KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL.hh
*@class KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL: public KFMNodeActor< KFMElectrostaticNode >
{
    public:

        KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL();
        virtual ~KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL();

        void SetParameters(KFMElectrostaticParameters params);
        void SetTree(KFMElectrostaticTree* tree);

        void SetNodeMomentBuffer(cl::Buffer* node_moments){fNodeMomentBufferCL = node_moments;};
        void SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set);

        ////////////////////////////////////////////////////////////////////////
        void Initialize();

        ////////////////////////////////////////////////////////////////////////
        virtual void ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node);

        void Prepare();
        void Finalize();

        void CopyMomentsToDevice();
        void RecieveMomentsFromDevice();

    protected:

        void BufferNode(KFMNode<KFMElectrostaticNodeObjects>* node);
        void ExecuteBufferedAction();
        void ClearBuffers();

        void BuildBuffers();
        void AssignBuffers();

        void ConstructTransformationKernel();
        void ConstructReduceKernel();

////////////////////////////////////////////////////////////////////////////////

        KFMElectrostaticTree* fTree;

        //parameters extracted from tree
        unsigned int fDegree;
        unsigned int fNTerms;
        unsigned int fStride;
        unsigned int fDivisions;

        //needed when computing scale factors
        unsigned int fTopLevelDivisions;
        unsigned int fLowerLevelDivisions;

        unsigned int fZeroMaskSize;
        unsigned int fMaxTreeDepth;

        //buffering
        int fCachedNodeLevel;
        unsigned int fNMaxBufferedNodes;
        unsigned int fNMaxParentNodes;
        unsigned int fNBufferedNodes;
        unsigned int fNBufferedParentNodes;

        mutable cl::Kernel* fTransformationKernel;
        unsigned int fNTransformationLocal;

        mutable cl::Kernel* fReduceKernel;
        unsigned int fNReduceLocal;

        //need a buffer to store the (unormalized) M2M response functions on the GPU
        cl::Buffer* fM2MCoeffBufferCL;

        //temporary buffer to store transformed moments
        cl::Buffer* fTransformedMomentBufferCL;

        //scale factor buffers for scale invariant kernels
        double fWorldLength;
        std::vector< CL_TYPE > fSourceScaleFactorArray;
        std::vector< CL_TYPE > fTargetScaleFactorArray;
        cl::Buffer* fSourceScaleFactorBufferCL;
        cl::Buffer* fTargetScaleFactorBufferCL;

        //ptr to external buffer which stores all multipole moments
        cl::Buffer* fNodeMomentBufferCL;

        //buffer for the multipole node id's
        cl::Buffer* fNodeIDBufferCL;

        //buffer for the block set id's
        cl::Buffer* fBlockSetIDListBufferCL;

        //buffer of offsets to the transformed moment data of particular parent node (for reduction kernel)
        cl::Buffer* fParentNodeOffsetBufferCL;

        //buffer of number of child owned by a parent node (for reduction kernel)
        cl::Buffer* fNChildNodeBufferCL;

        //buffer of parent ids, (for reduction kernel)
        cl::Buffer* fParentNodeIDBufferCL;

        unsigned int fNMultipoleNodes;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fMultipoleNodes;

        //space to cache the ids of the child multipole nodes for each parent that needs processing
        //these are indexed by multipole node id
        std::vector< std::vector<unsigned int> > fCachedMultipoleNodeLists;
        std::vector< std::vector<unsigned int> > fCachedBlockSetIDLists;

        //vectors of offset, size, and ids for reduction kernel
        std::vector<unsigned int> fParentNodeOffsetBuffer;
        std::vector<unsigned int> fNChildBuffer;
        std::vector<unsigned int> fParentNodeIDBuffer;
        //for transformation kernel
        std::vector<unsigned int> fNodeIDBuffer;
        std::vector<unsigned int> fBlockSetIDBuffer;

        //M2M response calculator and data
        KFMKernelResponseArray_3DLaplaceM2M* fKernelResponse;
        KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>* fScaleInvariantKernel;
        std::vector< std::complex<double> > fRawM2MCoeff;
        KFMArrayWrapper<std::complex<double>,  KFMELECTROSTATICS_DIM + 2>* fM2MCoeff;

        //limits, and size
        int fLowerLimits[KFMELECTROSTATICS_DIM + 2];
        int fUpperLimits[KFMELECTROSTATICS_DIM + 2];
        unsigned int fDimensionSize[KFMELECTROSTATICS_DIM + 2];
        unsigned int fTotalSpatialSize;



////////////////////////////////////////////////////////////////////////////////


};


}



#endif /* __KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL_H__ */
