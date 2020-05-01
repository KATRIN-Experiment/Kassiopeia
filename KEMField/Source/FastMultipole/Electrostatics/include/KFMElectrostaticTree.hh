#ifndef KFMElectrostaticTree_HH__
#define KFMElectrostaticTree_HH__

#include "KFMCubicSpaceTree.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMExternalIdentitySetSorter.hh"
#include "KFMIdentitySetSorter.hh"


//from kernel
#include "KFMResponseKernel_3DLaplaceL2L.hh"
#include "KFMResponseKernel_3DLaplaceM2L.hh"
#include "KFMResponseKernel_3DLaplaceM2M.hh"

//from tree
#include "KFMCollocationPointIdentitySetCreator.hh"
#include "KFMCubicSpaceNodeAdjacencyProgenitor.hh"
#include "KFMCubicSpaceTreeNavigator.hh"
#include "KFMElementLocalInfluenceRangeCollector.hh"
#include "KFMElementLocator.hh"
#include "KFMElementNodeAssociator.hh"
#include "KFMElementScalarMomentDistributor.hh"
#include "KFMIdentitySetCollector.hh"
#include "KFMIdentitySetListCreator.hh"
#include "KFMIdentitySetMerger.hh"
#include "KFMNearbyElementCounter.hh"
#include "KFMNodeFlagInitializer.hh"
#include "KFMNodeIdentityListCreator.hh"
#include "KFMNodeIdentityListRange.hh"
#include "KFMNodeIdentityListRangeAssociator.hh"
#include "KFMReducedScalarMomentRemoteToLocalConverter.hh"
#include "KFMRemoteToLocalConverterInterface.hh"
#include "KFMScalarMomentDistributor.hh"
#include "KFMScalarMomentInitializer.hh"
#include "KFMScalarMomentLocalToLocalConverter.hh"
#include "KFMScalarMomentRemoteToLocalConverter.hh"
#include "KFMScalarMomentRemoteToRemoteConverter.hh"
#include "KFMScalarMomentResetter.hh"

#include <string>

#define USE_REDUCED_M2L


namespace KEMField
{


//we operate on the tree with the following visitors
typedef KFMNearbyElementCounter<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>
    KFMElectrostaticNearbyElementCounter;

typedef KFMElementNodeAssociator<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>
    KFMElectrostaticElementNodeAssociator;

typedef KFMIdentitySetListCreator<KFMElectrostaticNodeObjects> KFMElectrostaticIdentitySetListCreator;

//distributor of element moments
typedef KFMElementScalarMomentDistributor<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                          KFMELECTROSTATICS_DIM>
    KFMElectrostaticElementMultipoleDistributor;


//initializers
typedef KFMScalarMomentInitializer<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>
    KFMElectrostaticMultipoleInitializer;

typedef KFMScalarMomentInitializer<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>
    KFMElectrostaticLocalCoefficientInitializer;

//resetters
typedef KFMScalarMomentResetter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>
    KFMElectrostaticMultipoleResetter;

typedef KFMScalarMomentResetter<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>
    KFMElectrostaticLocalCoefficientResetter;

//moment converters
typedef KFMScalarMomentRemoteToRemoteConverter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                               KFMResponseKernel_3DLaplaceM2M, KFMELECTROSTATICS_DIM>
    KFMElectrostaticRemoteToRemoteConverter;

typedef KFMScalarMomentLocalToLocalConverter<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet,
                                             KFMResponseKernel_3DLaplaceL2L, KFMELECTROSTATICS_DIM>
    KFMElectrostaticLocalToLocalConverter;

#ifdef USE_REDUCED_M2L
typedef KFMReducedScalarMomentRemoteToLocalConverter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                                     KFMElectrostaticLocalCoefficientSet,
                                                     KFMResponseKernel_3DLaplaceM2L, KFMELECTROSTATICS_DIM>
    KFMElectrostaticRemoteToLocalConverter;
#else
typedef KFMScalarMomentRemoteToLocalConverter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                              KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L,
                                              KFMELECTROSTATICS_DIM>
    KFMElectrostaticRemoteToLocalConverter;
#endif

//interface to m2l converters to handle different divisions on top level
typedef KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM,
                                           KFMElectrostaticRemoteToLocalConverter>
    KFMElectrostaticRemoteToLocalConverterInterface;

//navigator
typedef KFMCubicSpaceTreeNavigator<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM> KFMElectrostaticTreeNavigator;

//id set collector
typedef KFMIdentitySetMerger<KFMElectrostaticNodeObjects> KFMElectrostaticNodeIdentitySetMerger;

//inspector to determine node primacy
typedef KFMCubicSpaceNodeAdjacencyProgenitor<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>
    KFMElectrostaticAdjacencyProgenitor;


//sorters for the identity set, and external identity set
typedef KFMIdentitySetSorter<KFMElectrostaticNodeObjects> KFMElectrostaticIdentitySetSorter;

typedef KFMExternalIdentitySetSorter<KFMElectrostaticNodeObjects> KFMElectrostaticExternalIdentitySetSorter;

typedef KFMElementLocator<KFMElectrostaticNodeObjects> KFMElectrostaticElementLocator;

typedef KFMNodeIdentityListCreator<KFMElectrostaticNodeObjects> KFMElectrostaticNodeIdentityListCreator;

typedef KFMNodeIdentityListRangeAssociator<KFMElectrostaticNodeObjects> KFMElectrostaticNodeIdentityListRangeAssociator;

typedef KFMElementInfluenceRangeCollector<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>
    KFMElectrostaticElementInfluenceRangeCollector;

typedef KFMIdentitySetCollector<KFMElectrostaticNodeObjects> KFMElectrostaticNodeIdentitySetCollector;

typedef KFMCollocationPointIdentitySetCreator<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>
    KFMElectrostaticCollocationPointIdentitySetCreator;

//the local coefficient calculator
//KFMLocalCoefficientCalculator* fLocalCoeffCalculator;


/*
*
*@file KFMElectrostaticTree.hh
*@class KFMElectrostaticTree
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 15:00:38 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

//this is the type of tree we operate on
class KFMElectrostaticTree : public KFMCubicSpaceTree<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>
{
  public:
    KFMElectrostaticTree() : KFMCubicSpaceTree<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>()
    {
        ;
    }
    ~KFMElectrostaticTree() override{};

    void SetParameters(KFMElectrostaticParameters params)
    {
        fParameters = params;
    }

    KFMElectrostaticParameters GetParameters()
    {
        return fParameters;
    };

    std::string GetUniqueID() const
    {
        return fUniqueID;
    };
    void SetUniqueID(std::string unique_id)
    {
        fUniqueID = unique_id;
    };

  private:
    KFMElectrostaticParameters fParameters;
    std::string fUniqueID;
};


}  // namespace KEMField

#endif /* KFMElectrostaticTree_H__ */
