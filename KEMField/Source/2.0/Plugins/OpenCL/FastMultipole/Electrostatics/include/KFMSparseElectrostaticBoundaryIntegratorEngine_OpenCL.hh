#ifndef __KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL_H__
#define __KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL_H__


#include "KOpenCLInterface.hh"

#include "KFMObjectRetriever.hh"
#include "KFMNodeObjectRemover.hh"

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticParameters.hh"

#include "KFMSpecialNodeSet.hh"
#include "KFMSpecialNodeSetCreator.hh"

#include "KFMElectrostaticMultipoleCalculator_OpenCL.hh"
#include "KFMElectrostaticMultipoleDistributor_OpenCL.hh"

#include "KFMElectrostaticRemoteToRemoteConverter_OpenCL.hh"


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

        //extracted electrode data
        void SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<3,1>* container){fContainer = container;};

        //access to the region tree
        void SetTree(KFMElectrostaticTree* tree);

        void Initialize();

        void MapField();

    protected:

        void PrepareNodeSets();
        void AllocateBuffers();
        void CheckDeviceProperites();

        //operations
        void SetParameters(KFMElectrostaticParameters params);
        void InitializeMultipoleMoments();
        void ResetMultipoleMoments();
        void ComputeMultipoleMoments();
        void ResetLocalCoefficients();
        void InitializeLocalCoefficients();
        void ComputeLocalCoefficients();

        ////////////////////////////////////////////////////////////////////////

        //data
        int fDegree;
        unsigned int fNTerms;
        int fDivisions;
        int fZeroMaskSize;
        int fMaximumTreeDepth;
        unsigned int fVerbosity;
        double fWorldLength;

        //the tree object that the manager is to construct
        KFMElectrostaticTree* fTree;

        //container to the eletrostatic elements
        KFMElectrostaticElementContainerBase<3,1>* fContainer;

        //moment initializers
        KFMElectrostaticLocalCoefficientInitializer* fLocalCoeffInitializer;
        KFMElectrostaticMultipoleInitializer* fMultipoleInitializer;

        //moment resetters
        KFMElectrostaticLocalCoefficientResetter* fLocalCoeffResetter;
        KFMElectrostaticMultipoleResetter* fMultipoleResetter;

        //special sets of nodes
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects> fNonZeroMultipoleMomentNodes;
        KFMSpecialNodeSet<KFMElectrostaticNodeObjects> fPrimaryNodes;

        //the multipole moment calculator
        KFMElectrostaticMultipoleCalculator_OpenCL* fMultipoleCalculator;
        KFMElectrostaticMultipoleDistributor_OpenCL* fMultipoleDistributor;


        //the multipole up converter
        //KFMElectrostaticRemoteToRemoteConverter* fM2MConverter;
        KFMElectrostaticRemoteToRemoteConverter_OpenCL* fM2MConverter;
        //the local coefficient calculator
        KFMElectrostaticRemoteToLocalConverter* fM2LConverter;
        //the local coefficient down converter
        KFMElectrostaticLocalToLocalConverter* fL2LConverter;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

        //need to allocate buffers for the multipole and local coefficients
        //as well as the appropriate node id's

        unsigned int fNMultipoleNodes;
        unsigned int fNPrimaryNodes;
        unsigned int fNReducedTerms;
        cl::Buffer* fMultipoleBufferCL;
        cl::Buffer* fLocalCoeffBufferCL;


};

}

#endif /* __KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL_H__ */
