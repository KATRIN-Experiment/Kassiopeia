#ifndef __KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL_H__
#define __KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL_H__


#include "KFMElectrostaticBatchedLocalToLocalConverter_OpenCL.hh"
#include "KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL.hh"
#include "KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticLocalToLocalConverter_OpenCL.hh"
#include "KFMElectrostaticMultipoleCalculator_OpenCL.hh"
#include "KFMElectrostaticMultipoleDistributor_OpenCL.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticRemoteToLocalConverter_OpenCL.hh"
#include "KFMElectrostaticRemoteToRemoteConverter_OpenCL.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMNodeObjectRemover.hh"
#include "KFMObjectRetriever.hh"
#include "KFMRemoteToLocalConverterInterface.hh"
#include "KFMSpecialNodeSet.hh"
#include "KFMSpecialNodeSetCreator.hh"
#include "KOpenCLInterface.hh"


namespace KEMField
{

/**
*
*@file KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh
*@class KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul 16 13:26:56 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL
{
  public:
    KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL();
    virtual ~KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL();

    //for evaluating work load weights
    void EvaluateWorkLoads(unsigned int divisions, unsigned int zeromask);
    double GetDiskWeight() const
    {
        return fDiskWeight;
    };
    double GetRamWeight() const
    {
        return fRamWeight;
    };
    double GetFFTWeight() const
    {
        return fFFTWeight;
    };

    //extracted electrode data
    void SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<3, 1>* container)
    {
        fContainer = container;
    };

    void SetParameters(KFMElectrostaticParameters params);

    void SetTree(KFMElectrostaticTree* tree);

    void InitializeMultipoleMoments();

    void InitializeLocalCoefficientsForPrimaryNodes();

    //needed when MPI is in used and all processes must communicate their
    //influence with the GPU
    void RecieveTopLevelLocalCoefficients();
    void SendTopLevelLocalCoefficients();

    void Initialize();

    void MapField();

    //individual operations, only to be used when the tree needs
    //to be modified in between steps (i.e. MPI)
    void ResetMultipoleMoments();
    void ComputeMultipoleMoments();
    void ResetLocalCoefficients();
    void ComputeMultipoleToLocal();
    void ComputeLocalToLocal();
    void ComputeLocalCoefficients();

  protected:
    void PrepareNodeSets();
    void AllocateBuffers();
    void CheckDeviceProperites();

    double ComputeDiskMatrixVectorProductWeight();
    double ComputeRamMatrixVectorProductWeight();
    double ComputeFFTWeight(unsigned int divisions, unsigned int zeromask);

#ifdef KEMFIELD_USE_REALTIME_CLOCK
    timespec diff(timespec start, timespec end)
    {
        timespec temp;
        if ((end.tv_nsec - start.tv_nsec) < 0) {
            temp.tv_sec = end.tv_sec - start.tv_sec - 1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        }
        else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        return temp;
    }
#endif

    ////////////////////////////////////////////////////////////////////////

    //data
    int fDegree;
    unsigned int fNTerms;
    int fDivisions;
    int fTopLevelDivisions;
    int fZeroMaskSize;
    int fMaximumTreeDepth;
    unsigned int fVerbosity;
    double fWorldLength;

    double fDiskWeight;
    double fRamWeight;
    double fFFTWeight;
    static const std::string fWeightFilePrefix;

    //if device type is an accelerator (e.g. intel xeon phi)
    //we need to use the batched kernels in order to obtain decent performance
    bool fUseBatchedKernels;

    //the tree object that the manager is to construct
    KFMElectrostaticTree* fTree;

    //container to the eletrostatic elements
    KFMElectrostaticElementContainerBase<3, 1>* fContainer;

    //moment initializers
    KFMElectrostaticLocalCoefficientInitializer* fLocalCoeffInitializer;
    KFMElectrostaticMultipoleInitializer* fMultipoleInitializer;

    //moment resetters
    KFMElectrostaticLocalCoefficientResetter* fLocalCoeffResetter;
    KFMElectrostaticMultipoleResetter* fMultipoleResetter;

    //special sets of nodes
    KFMSpecialNodeSet<KFMElectrostaticNodeObjects> fNonZeroMultipoleMomentNodes;
    KFMSpecialNodeSet<KFMElectrostaticNodeObjects> fPrimaryNodes;
    KFMSpecialNodeSet<KFMElectrostaticNodeObjects> fTopLevelPrimaryNodes;

    //the multipole moment calculator
    KFMElectrostaticMultipoleCalculator_OpenCL* fMultipoleCalculator;

    //the multipole up converter
    KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL* fM2MConverter_Batched;
    KFMElectrostaticRemoteToRemoteConverter_OpenCL* fM2MConverter;

    //the local coefficient calculator
    KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM,
                                       KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL>*
        fM2LConverterInterface_Batched;
    KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM,
                                       KFMElectrostaticRemoteToLocalConverter_OpenCL>* fM2LConverterInterface;

    //the local coefficient down converter
    KFMElectrostaticBatchedLocalToLocalConverter_OpenCL* fL2LConverter_Batched;
    KFMElectrostaticLocalToLocalConverter_OpenCL* fL2LConverter;


    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //need to allocate buffers for the multipole and local coefficients
    //as well as the appropriate node id's

    unsigned int fNMultipoleNodes;
    unsigned int fNPrimaryNodes;
    unsigned int fNTopLevelPrimaryNodes;
    unsigned int fNReducedTerms;
    cl::Buffer* fMultipoleBufferCL;
    cl::Buffer* fLocalCoeffBufferCL;
};

}  // namespace KEMField

#endif /* __KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL_H__ */
