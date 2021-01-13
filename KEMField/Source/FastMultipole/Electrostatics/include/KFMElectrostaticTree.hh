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

using KFMElectrostaticElementNodeAssociator = KFMElementNodeAssociator<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>;

using KFMElectrostaticIdentitySetListCreator = KFMIdentitySetListCreator<KFMElectrostaticNodeObjects>;

//distributor of element moments
using KFMElectrostaticElementMultipoleDistributor =
    KFMElementScalarMomentDistributor<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMELECTROSTATICS_DIM>;


//initializers
using KFMElectrostaticMultipoleInitializer =
    KFMScalarMomentInitializer<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>;

using KFMElectrostaticLocalCoefficientInitializer =
    KFMScalarMomentInitializer<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>;

//resetters
using KFMElectrostaticMultipoleResetter =
    KFMScalarMomentResetter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>;

using KFMElectrostaticLocalCoefficientResetter =
    KFMScalarMomentResetter<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>;

//moment converters
using KFMElectrostaticRemoteToRemoteConverter =
    KFMScalarMomentRemoteToRemoteConverter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                           KFMResponseKernel_3DLaplaceM2M, KFMELECTROSTATICS_DIM>;

using KFMElectrostaticLocalToLocalConverter =
    KFMScalarMomentLocalToLocalConverter<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet,
                                         KFMResponseKernel_3DLaplaceL2L, KFMELECTROSTATICS_DIM>;

#ifdef USE_REDUCED_M2L
using KFMElectrostaticRemoteToLocalConverter =
    KFMReducedScalarMomentRemoteToLocalConverter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                                 KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L,
                                                 KFMELECTROSTATICS_DIM>;
#else
typedef KFMScalarMomentRemoteToLocalConverter<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet,
                                              KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L,
                                              KFMELECTROSTATICS_DIM>
    KFMElectrostaticRemoteToLocalConverter;
#endif

//interface to m2l converters to handle different divisions on top level
using KFMElectrostaticRemoteToLocalConverterInterface =
    KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM, KFMElectrostaticRemoteToLocalConverter>;

//navigator
using KFMElectrostaticTreeNavigator = KFMCubicSpaceTreeNavigator<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>;

//id set collector
using KFMElectrostaticNodeIdentitySetMerger = KFMIdentitySetMerger<KFMElectrostaticNodeObjects>;

//inspector to determine node primacy
using KFMElectrostaticAdjacencyProgenitor = KFMCubicSpaceNodeAdjacencyProgenitor<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>;


//sorters for the identity set, and external identity set
using KFMElectrostaticIdentitySetSorter = KFMIdentitySetSorter<KFMElectrostaticNodeObjects>;

using KFMElectrostaticExternalIdentitySetSorter = KFMExternalIdentitySetSorter<KFMElectrostaticNodeObjects>;

using KFMElectrostaticElementLocator = KFMElementLocator<KFMElectrostaticNodeObjects>;

using KFMElectrostaticNodeIdentityListCreator = KFMNodeIdentityListCreator<KFMElectrostaticNodeObjects>;

using KFMElectrostaticNodeIdentityListRangeAssociator = KFMNodeIdentityListRangeAssociator<KFMElectrostaticNodeObjects>;

using KFMElectrostaticElementInfluenceRangeCollector =
    KFMElementInfluenceRangeCollector<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>;

using KFMElectrostaticNodeIdentitySetCollector = KFMIdentitySetCollector<KFMElectrostaticNodeObjects>;

using KFMElectrostaticCollocationPointIdentitySetCreator =
    KFMCollocationPointIdentitySetCreator<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM>;

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
    ~KFMElectrostaticTree() override = default;
    ;

    void SetParameters(const KFMElectrostaticParameters& params)
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
    void SetUniqueID(const std::string& unique_id)
    {
        fUniqueID = unique_id;
    };

  private:
    KFMElectrostaticParameters fParameters;
    std::string fUniqueID;
};


}  // namespace KEMField

#endif /* KFMElectrostaticTree_H__ */
