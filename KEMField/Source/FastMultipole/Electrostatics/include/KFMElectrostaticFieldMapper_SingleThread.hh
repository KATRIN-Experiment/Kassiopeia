#ifndef KFMElectrostaticFieldMapper_SingleThread_HH__
#define KFMElectrostaticFieldMapper_SingleThread_HH__

#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleBatchCalculatorBase.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticRegionSizeEstimator.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMNodeObjectRemover.hh"
#include "KFMObjectRetriever.hh"


/*
*
*@file KFMElectrostaticFieldMapper_SingleThread.hh
*@class KFMElectrostaticFieldMapper_SingleThread
*@brief helper class to apply actions to a tree
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 15:05:27 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

namespace KEMField
{


class KFMElectrostaticFieldMapper_SingleThread
{
  public:
    KFMElectrostaticFieldMapper_SingleThread();
    virtual ~KFMElectrostaticFieldMapper_SingleThread();

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
    //operations
    void SetParameters(const KFMElectrostaticParameters& params);
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
    KFMElectrostaticRemoteToRemoteConverter* fM2MConverter;

    //the local coefficient calculator
    //        KFMElectrostaticRemoteToLocalConverter* fM2LConverter;
    KFMElectrostaticRemoteToLocalConverterInterface* fM2LConverterInterface;

    //the local coefficient down converter
    KFMElectrostaticLocalToLocalConverter* fL2LConverter;

    //container to the eletrostatic elements
    KFMElectrostaticElementContainerBase<3, 1>* fContainer;
};


}  // namespace KEMField


#endif /* KFMElectrostaticFieldMapper_SingleThread_H__ */
