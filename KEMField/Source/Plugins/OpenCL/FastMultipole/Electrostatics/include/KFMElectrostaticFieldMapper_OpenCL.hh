#ifndef __KFMElectrostaticFieldMapper_OpenCL_H__
#define __KFMElectrostaticFieldMapper_OpenCL_H__

#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleBatchCalculatorBase.hh"
#include "KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMNodeObjectRemover.hh"
#include "KFMObjectRetriever.hh"
#include "KFMReducedScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMRemoteToLocalConverterInterface.hh"
#include "KFMScalarMomentLocalToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh"


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
    void SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<3, 1>* container)
    {
        fContainer = container;
    };

    //access to the region tree
    void SetTree(KFMElectrostaticTree* tree);

    void Initialize();

    void MapField();

  protected:
#ifdef USE_REDUCED_M2L
    typedef KFMReducedScalarMomentRemoteToLocalConverter_OpenCL<
        KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet,
        KFMResponseKernel_3DLaplaceM2L, 3>
        KFMElectrostaticRemoteToLocalConverter_OpenCL;
#else
    typedef KFMScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                                         KFMElectrostaticLocalCoefficientSet,
                                                         KFMResponseKernel_3DLaplaceM2L, 3>
        KFMElectrostaticRemoteToLocalConverter_OpenCL;
#endif

    typedef KFMScalarMomentLocalToLocalConverter_OpenCL<
        KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceL2L, 3>
        KFMElectrostaticLocalToLocalConverter_OpenCL;

    typedef KFMScalarMomentRemoteToRemoteConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                                          KFMResponseKernel_3DLaplaceM2M, 3>
        KFMElectrostaticRemoteToRemoteConverter_OpenCL;

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
    int fTopLevelDivisions;
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
    KFMElectrostaticRemoteToRemoteConverter_OpenCL* fM2MConverter;

    //the local coefficient calculator
    KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM,
                                       KFMElectrostaticRemoteToLocalConverter_OpenCL>* fM2LConverterInterface;
    // KFMElectrostaticRemoteToLocalConverter* fM2LConverter;

    //the local coefficient down converter
    KFMElectrostaticLocalToLocalConverter_OpenCL* fL2LConverter;

    //container to the eletrostatic elements
    KFMElectrostaticElementContainerBase<3, 1>* fContainer;
};


}  // namespace KEMField

#endif /* __KFMElectrostaticFieldMapper_OpenCL_H__ */
