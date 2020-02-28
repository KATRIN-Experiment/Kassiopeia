#ifndef KFMElectrostaticRemoteToRemoteConverter_OpenCL_H__
#define KFMElectrostaticRemoteToRemoteConverter_OpenCL_H__


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

//math
#include "KFMPointCloud.hh"
#include "KFMMath.hh"
#include "KFMCube.hh"


//tree
#include "KFMSpecialNodeSet.hh"

//electrostatics
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticElementContainer.hh"

namespace KEMField{

/**
*
*@file KFMElectrostaticRemoteToRemoteConverter_OpenCL.hh
*@class KFMElectrostaticRemoteToRemoteConverter_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticRemoteToRemoteConverter_OpenCL: public KFMNodeActor< KFMElectrostaticNode >
{
    public:

        KFMElectrostaticRemoteToRemoteConverter_OpenCL();
        virtual ~KFMElectrostaticRemoteToRemoteConverter_OpenCL();

        void SetParameters(KFMElectrostaticParameters params);
        void SetTree(KFMElectrostaticTree* tree);

        void SetNodeMomentBuffer(cl::Buffer* node_moments){fNodeMomentBufferCL = node_moments;};
        void SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set);

        ////////////////////////////////////////////////////////////////////////
        void Initialize();

        ////////////////////////////////////////////////////////////////////////
        virtual void ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node);

        //functions for debugging, copies data from tree to device and back
        void Prepare();
        void Finalize();

    protected:

        void BuildBuffers();
        void AssignBuffers();

        void ConstructCopyAndScaleKernel();
        void ConstructTransformationKernel();
        void ConstructReduceAndScaleKernel();

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

        mutable cl::Kernel* fCopyAndScaleKernel;
        unsigned int fNCopyAndScaleLocal;

        mutable cl::Kernel* fTransformationKernel;
        unsigned int fNTransformationLocal;

        mutable cl::Kernel* fReduceAndScaleKernel;
        unsigned int fNReduceAndScaleLocal;

        //need a buffer to store the (unormalized) M2M response functions on the GPU
        cl::Buffer* fM2MCoeffBufferCL;

        //temporary buffer to store scaled but untransformed child moments
        cl::Buffer* fChildMomentBufferCL;

        //temporary buffer to store transformed child moments
        cl::Buffer* fTransformedChildMomentBufferCL;

        //scale factor buffers for scale invariant kernels
        double fWorldLength;
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
        std::vector< CL_TYPE > fSourceScaleFactorArray;
        std::vector< CL_TYPE > fTargetScaleFactorArray;
        #pragma GCC diagnostic pop
        cl::Buffer* fSourceScaleFactorBufferCL;
        cl::Buffer* fTargetScaleFactorBufferCL;

        //ptr to external buffer which stores all multipole moments
        cl::Buffer* fNodeMomentBufferCL;

        //buffer for the multipole node id's
        cl::Buffer* fNodeIDBufferCL;

        //buffer for the block set id's
        cl::Buffer* fBlockSetIDListBufferCL;

        unsigned int fNMultipoleNodes;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fMultipoleNodes;

        //space to cache the ids of the child multipole nodes for each parent that needs processing
        //these are indexed by multipole node id
        std::vector< std::vector<unsigned int> > fCachedMultipoleNodeLists;
        std::vector< std::vector<unsigned int> > fCachedBlockSetIDLists;

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



#endif /* __KFMElectrostaticRemoteToRemoteConverter_OpenCL_H__ */
