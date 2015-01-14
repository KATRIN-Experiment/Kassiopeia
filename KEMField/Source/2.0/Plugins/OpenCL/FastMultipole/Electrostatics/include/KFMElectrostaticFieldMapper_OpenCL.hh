#ifndef __KFMElectrostaticFieldMapper_OpenCL_H__
#define __KFMElectrostaticFieldMapper_OpenCL_H__

#include "KFMObjectRetriever.hh"
#include "KFMNodeObjectRemover.hh"

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleBatchCalculatorBase.hh"
#include "KFMElectrostaticParameters.hh"


namespace KEMField
{

/**
*
*@file KFMElectrostaticFieldMapper_OpenCL.hh
*@class KFMElectrostaticFieldMapper_OpenCL
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul 16 09:53:50 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticFieldMapper_OpenCL
{
    public:
        KFMElectrostaticFieldMapper_OpenCL();
        virtual ~KFMElectrostaticFieldMapper_OpenCL();

        //extracted electrode data
        void SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<3,1>* container){fContainer = container;};

        //access to the region tree
        void SetTree(KFMElectrostaticTree* tree);

        void Initialize();

        void MapField();

    protected:

        //operations
        void SetParameters(KFMElectrostaticParameters params);
        void AssociateElementsAndNodes();
        void InitializeMultipoleMoments();
        void ComputeMultipoleMoments();
        void InitializeLocalCoefficients();
        void ComputeLocalCoefficients();
        void CleanUp();

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

        //element node associator
        KFMElectrostaticElementNodeAssociator* fElementNodeAssociator;
        //the multipole calculator
        KFMElectrostaticMultipoleBatchCalculatorBase* fBatchCalc;
        //the element's multipole distributor
        KFMElectrostaticElementMultipoleDistributor* fMultipoleDistributor;

        //the local coefficient initializer
        KFMElectrostaticLocalCoefficientInitializer* fLocalCoeffInitializer;

        //the multipole coefficient initializer
        KFMElectrostaticMultipoleInitializer* fMultipoleInitializer;

        //the multipole up converter
        KFMElectrostaticRemoteToRemoteConverter* fM2MConverter;
        //the local coefficient calculator
        KFMElectrostaticRemoteToLocalConverter* fM2LConverter;
        //the local coefficient down converter
        KFMElectrostaticLocalToLocalConverter* fL2LConverter;

        //container to the eletrostatic elements
        KFMElectrostaticElementContainerBase<3,1>* fContainer;

};


}

#endif /* __KFMElectrostaticFieldMapper_OpenCL_H__ */
