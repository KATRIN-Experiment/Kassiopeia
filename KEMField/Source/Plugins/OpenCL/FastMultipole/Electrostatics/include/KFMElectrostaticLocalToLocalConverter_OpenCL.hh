#ifndef __KFMElectrostaticLocalToLocalConverter_H__
#define __KFMElectrostaticLocalToLocalConverter_H__


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


namespace KEMField
{

/**
*
*@file KFMElectrostaticLocalToLocalConverter_OpenCL.hh
*@class KFMElectrostaticLocalToLocalConverter
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug  7 16:50:19 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticLocalToLocalConverter_OpenCL: public KFMNodeActor< KFMElectrostaticNode >
{
    public:
        KFMElectrostaticLocalToLocalConverter_OpenCL();
        virtual ~KFMElectrostaticLocalToLocalConverter_OpenCL();

        void SetParameters(KFMElectrostaticParameters params);
        void SetTree(KFMElectrostaticTree* tree);

        void SetNodeMomentBuffer(cl::Buffer* node_moments){fNodeMomentBufferCL = node_moments;};
        void SetPrimaryNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* primary_node_set);

        void Finalize();

        ////////////////////////////////////////////////////////////////////////
        void Initialize();

        ////////////////////////////////////////////////////////////////////////
        virtual void ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node);

    protected:

        void BuildBuffers();
        void AssignBuffers();

        void ConstructTransformKernel();

////////////////////////////////////////////////////////////////////////////////

        KFMElectrostaticTree* fTree;

        //parameters extracted from tree
        unsigned int fDegree;
        unsigned int fNTerms;
        unsigned int fStride;
        unsigned int fDivisions;

        //need when computing scale factors
        unsigned int fTopLevelDivisions;
        unsigned int fLowerLevelDivisions;

        unsigned int fZeroMaskSize;
        unsigned int fMaxTreeDepth;

        mutable cl::Kernel* fTransformKernel;
        unsigned int fNTransformLocal;

        //need a buffer to store the L2L response functions on the GPU
        cl::Buffer* fL2LCoeffBufferCL;

        //scale factor buffers for scale invariant kernels
        double fWorldLength;
        std::vector< CL_TYPE > fSourceScaleFactorArray;
        std::vector< CL_TYPE > fTargetScaleFactorArray;
        cl::Buffer* fSourceScaleFactorBufferCL;
        cl::Buffer* fTargetScaleFactorBufferCL;

        //ptr to external buffer which stores all local moments
        cl::Buffer* fNodeMomentBufferCL;

        //buffer for the primary node id's
        cl::Buffer* fNodeIDBufferCL;

        //buffer for the block set id's
        cl::Buffer* fBlockSetIDListBufferCL;

        unsigned int fNPrimaryNodes;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fPrimaryNodes;

        //space to cache the ids of the child multipole nodes for each parent that needs processing
        //these are indexed by multipole node id
        std::vector< std::vector<unsigned int> > fCachedPrimaryNodeIDLists;
        std::vector< std::vector<unsigned int> > fCachedBlockSetIDLists;

        //M2M response calculator and data
        KFMKernelResponseArray_3DLaplaceL2L* fKernelResponse;
        KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>* fScaleInvariantKernel;
        std::vector< std::complex<double> > fRawL2LCoeff;
        KFMArrayWrapper<std::complex<double>,  KFMELECTROSTATICS_DIM + 2>* fL2LCoeff;

        //limits, and size
        int fLowerLimits[KFMELECTROSTATICS_DIM + 2];
        int fUpperLimits[KFMELECTROSTATICS_DIM + 2];
        unsigned int fDimensionSize[KFMELECTROSTATICS_DIM + 2];
        unsigned int fTotalSpatialSize;

////////////////////////////////////////////////////////////////////////////////

};

}

#endif /* __KFMElectrostaticLocalToLocalConverter_H__ */
