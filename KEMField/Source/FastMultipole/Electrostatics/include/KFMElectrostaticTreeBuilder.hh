#ifndef __KFMElectrostaticTreeBuilder_H__
#define __KFMElectrostaticTreeBuilder_H__


#include "KFMObjectRetriever.hh"
#include "KFMNodeObjectRemover.hh"

#include "KFMCubicSpaceTree.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMCubicSpaceBallSorter.hh"
#include "KFMInsertionCondition.hh"

#include "KFMSubdivisionCondition.hh"
#include "KFMSubdivisionConditionAggressive.hh"
#include "KFMSubdivisionConditionBalanced.hh"
#include "KFMSubdivisionConditionGuided.hh"

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticElementContainerBase.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticRegionSizeEstimator.hh"

namespace KEMField
{


/**
*
*@file KFMElectrostaticTreeBuilder.hh
*@class KFMElectrostaticTreeBuilder
*@brief class responsible for constructing the tree's 'skeleton' (nodes and their relations)
*Does not compute moments or provide visitors to do this
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul  7 11:09:06 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticTreeBuilder
{
    public:
        KFMElectrostaticTreeBuilder()
        {
            fSubdivisionCondition = NULL;
            fSubdivisionConditionIsOwned = false;
        };

        virtual ~KFMElectrostaticTreeBuilder()
        {
            if(fSubdivisionConditionIsOwned)
            {
                delete fSubdivisionCondition;
            }
        };

        //extracted electrode data
        void SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM,KFMELECTROSTATICS_BASIS>* container);
        KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM,KFMELECTROSTATICS_BASIS>* GetElectrostaticElementContainer();

        //access to the region tree, tree builder does not own the tree!
        void SetTree(KFMElectrostaticTree* tree);
        KFMElectrostaticTree* GetTree();

        void SetSubdivisionCondition( KFMSubdivisionCondition<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* subdiv)
        {
            if(subdiv != NULL)
            {
                fSubdivisionCondition = subdiv;
                fSubdivisionConditionIsOwned = false;
            }
        }


        //build up the tree using these functions
        //typically these are applied in the same order as they are listed here
        void ConstructRootNode();
        void PerformSpatialSubdivision();
        void FlagNonZeroMultipoleNodes();
        void PerformAdjacencySubdivision();
        void FlagPrimaryNodes();
        void CollectDirectCallIdentities();
        void CollectDirectCallIdentitiesForPrimaryNodes();

    protected:

        /* data */
        int fDegree;
        unsigned int fNTerms;
        int fTopLevelDivisions;
        int fDivisions;
        int fZeroMaskSize;
        int fMaximumTreeDepth;
        double fInsertionRatio;
        unsigned int fVerbosity;
        double fRegionSizeFactor;

        bool fUseRegionEstimation;
        KFMPoint<3> fWorldCenter;
        double fWorldLength;

        //the tree object that the manager is to construct
        KFMElectrostaticTree* fTree;

        //subdivision condition
        KFMSubdivisionCondition<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* fSubdivisionCondition;
        bool fSubdivisionConditionIsOwned;

        //manager does not own this object!
        //container to the eletrostatic elements
        KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>* fContainer;
};




}

#endif /* __KFMElectrostaticTreeBuilder_H__ */
