#ifndef __KFMElectrostaticBatchedLocalToLocalConverter_OpenCL_H__
#define __KFMElectrostaticBatchedLocalToLocalConverter_OpenCL_H__


#include <complex>
#include <vector>

//core
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

//kernel
#include "KFMKernelExpansion.hh"
#include "KFMKernelResponseArray.hh"
#include "KFMKernelResponseArrayTypes.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

//core (opencl)
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

//kernel
#include "KFMScalarMultipoleExpansion.hh"

//tree
#include "KFMSpecialNodeSet.hh"

//electrostatics
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTree.hh"

namespace KEMField
{

/**
*
*@file KFMElectrostaticBatchedLocalToLocalConverter_OpenCL.hh
*@class KFMElectrostaticBatchedLocalToLocalConverter_OpenCL
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug  7 16:50:19 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticBatchedLocalToLocalConverter_OpenCL : public KFMNodeActor<KFMElectrostaticNode>
{
  public:
    KFMElectrostaticBatchedLocalToLocalConverter_OpenCL();
    virtual ~KFMElectrostaticBatchedLocalToLocalConverter_OpenCL();

    void SetParameters(KFMElectrostaticParameters params);
    void SetTree(KFMElectrostaticTree* tree);

    void SetNodeMomentBuffer(cl::Buffer* node_moments)
    {
        fNodeMomentBufferCL = node_moments;
    };
    void SetPrimaryNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* primary_node_set);

    void Prepare();
    void Finalize();

    ////////////////////////////////////////////////////////////////////////
    void Initialize();

    ////////////////////////////////////////////////////////////////////////
    virtual void ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node);

  protected:
    void BufferNode(KFMNode<KFMElectrostaticNodeObjects>* node);
    void ExecuteBufferedAction();
    void ClearBuffers();

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
    std::vector<unsigned int> fNodeIDBuffer;
    std::vector<unsigned int> fBlockSetIDBuffer;
    std::vector<unsigned int> fParentIDBuffer;

    mutable cl::Kernel* fTransformKernel;
    unsigned int fNTransformLocal;

    //need a buffer to store the L2L response functions on the GPU
    cl::Buffer* fL2LCoeffBufferCL;

    //scale factor buffers for scale invariant kernels
    double fWorldLength;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::vector<CL_TYPE> fSourceScaleFactorArray;
    std::vector<CL_TYPE> fTargetScaleFactorArray;
#pragma GCC diagnostic pop
    cl::Buffer* fSourceScaleFactorBufferCL;
    cl::Buffer* fTargetScaleFactorBufferCL;

    //ptr to external buffer which stores all local moments
    cl::Buffer* fNodeMomentBufferCL;

    //buffer for the primary node id's
    cl::Buffer* fNodeIDBufferCL;

    //buffer for the block set id's
    cl::Buffer* fBlockSetIDListBufferCL;

    //buffer for the parent node id's
    cl::Buffer* fParentIDBufferCL;

    unsigned int fNPrimaryNodes;
    KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* fPrimaryNodes;

    //space to cache the ids of the child multipole nodes for each parent that needs processing
    //these are indexed by multipole node id
    std::vector<std::vector<unsigned int>> fCachedPrimaryNodeIDLists;
    std::vector<std::vector<unsigned int>> fCachedBlockSetIDLists;

    //L2L response calculator and data
    KFMKernelResponseArray_3DLaplaceL2L* fKernelResponse;
    KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>* fScaleInvariantKernel;
    std::vector<std::complex<double>> fRawL2LCoeff;
    KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM + 2>* fL2LCoeff;

    //limits, and size
    int fLowerLimits[KFMELECTROSTATICS_DIM + 2];
    int fUpperLimits[KFMELECTROSTATICS_DIM + 2];
    unsigned int fDimensionSize[KFMELECTROSTATICS_DIM + 2];
    unsigned int fTotalSpatialSize;

    ////////////////////////////////////////////////////////////////////////////////
};

}  // namespace KEMField

#endif /* __KFMElectrostaticBatchedLocalToLocalConverter_OpenCL_H__ */
