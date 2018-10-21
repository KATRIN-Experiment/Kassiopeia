#ifndef __KFMElectrostaticTreeBuilder_MPI_H__
#define __KFMElectrostaticTreeBuilder_MPI_H__


#include "KFMObjectRetriever.hh"
#include "KFMNodeObjectRemover.hh"

#include "KFMCubicSpaceTree.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMCubicSpaceBallSorter.hh"
#include "KFMInsertionCondition.hh"
#include "KFMSubdivisionCondition.hh"

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticElementContainerBase.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticRegionSizeEstimator.hh"
#include "KFMCubicVolumeCollection.hh"

#include "KMPIInterface.hh"

namespace KEMField
{


/**
*
*@file KFMElectrostaticTreeBuilder_MPI.hh
*@class KFMElectrostaticTreeBuilder_MPI
*@brief class responsible for constructing the tree's 'skeleton' (nodes and their relations)
*Does not compute moments or provide visitors to do this
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul  7 11:09:06 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticTreeBuilder_MPI
{
    public:
        KFMElectrostaticTreeBuilder_MPI()
        {
            fSubdivisionCondition = NULL;
            fSubdivisionConditionIsOwned = false;
        };

        virtual ~KFMElectrostaticTreeBuilder_MPI()
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

        void SetFFTWeight(double fft_weight){fFFTWeight = fft_weight;};
        void SetSparseMatrixWeight(double mx_weight){fSparseMatrixWeight = mx_weight;};

        //build up the tree using these functions
        //typically these are applied in the same order as they are listed here
        void ConstructRootNode();
        void PerformSpatialSubdivision();
        void FlagNonZeroMultipoleNodes();
        void PerformAdjacencySubdivision();
        void FlagPrimaryNodes();

        void DetermineSourceNodes();
        void DetermineTargetNodes();
        void RemoveExtraneousData();

        void CollectDirectCallIdentitiesForPrimaryNodes();

        const KFMCubicVolumeCollection<KFMELECTROSTATICS_DIM>* GetSourceVolume() const {return &fSourceVolume;};
        const KFMCubicVolumeCollection<KFMELECTROSTATICS_DIM>* GetTargetVolume() const {return &fTargetVolume;};

        void GetSourceNodeIndexes(std::vector<unsigned int>* source_node_indexes) const;
        void GetTargetNodeIndexes(std::vector<unsigned int>* target_node_indexes) const;

    protected:

        /* data */
        int fDegree;
        unsigned int fNTerms;
        int fDivisions;
        int fTopLevelDivisions;
        int fZeroMaskSize;
        int fMaximumTreeDepth;
        double fInsertionRatio;
        unsigned int fVerbosity;
        double fRegionSizeFactor;

        double fFFTWeight;
        double fSparseMatrixWeight;

        //MPI
        unsigned int fNSourceNodes;
        unsigned int fNTargetNodes;
        std::vector< KFMElectrostaticNode* > fSourceNodeCollection;
        std::vector< KFMElectrostaticNode* > fNonSourceNodeCollection;
        std::vector< KFMElectrostaticNode* > fTargetNodeCollection;
        std::vector< KFMElectrostaticNode* > fNonTargetNodeCollection;
        KFMCubicVolumeCollection<KFMELECTROSTATICS_DIM> fSourceVolume;
        KFMCubicVolumeCollection<KFMELECTROSTATICS_DIM> fTargetVolume;


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

#endif /* __KFMElectrostaticTreeBuilder_MPI_H__ */
