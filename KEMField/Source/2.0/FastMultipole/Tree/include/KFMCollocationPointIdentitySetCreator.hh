#ifndef __KFMCollocationPointIdentitySetCreator_H__
#define __KFMCollocationPointIdentitySetCreator_H__

#include "KFMObjectRetriever.hh"
#include "KFMCollocationPointIdentitySet.hh"

#include "KFMCubicSpaceTree.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMCubicSpaceTreeNavigator.hh"

#include "KFMMessaging.hh"


namespace KEMField
{

/**
*
*@file KFMCollocationPointIdentitySetCreator.hh
*@class KFMCollocationPointIdentitySetCreator
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Sep 29 14:03:15 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList, unsigned int SpatialNDIM >
class KFMCollocationPointIdentitySetCreator
{
    public:
        KFMCollocationPointIdentitySetCreator():fTree(NULL),fRootNode(NULL){};
        virtual ~KFMCollocationPointIdentitySetCreator(){};

        void SetTree(KFMCubicSpaceTree<SpatialNDIM, ObjectTypeList>* tree)
        {
            fTree = tree;
            fRootNode = fTree->GetRootNode();
        };

        void SortCollocationPoints(const std::vector< const KFMPoint<SpatialNDIM>* >* c_points)
        {
            //we assume that the placement of the collocation point in the list is its identity index
            unsigned int size = c_points->size();
            for(unsigned int id=0; id<size; id++)
            {
                fNavigator.SetPoint(c_points->at(id));
                fNavigator.ApplyAction(fRootNode);

                if(fNavigator.Found())
                {
                    KFMNode<ObjectTypeList>* node = fNavigator.GetLeafNode();

                    //we add this id to the leaf node's collocation point id list
                    KFMCollocationPointIdentitySet* cpid_set = NULL;
                    cpid_set = KFMObjectRetriever<ObjectTypeList, KFMCollocationPointIdentitySet >::GetNodeObject(node);

                    if(cpid_set == NULL)
                    {
                        //create this collocation point id set and add it to this node
                        cpid_set = new KFMCollocationPointIdentitySet();
                        KFMObjectRetriever<ObjectTypeList, KFMCollocationPointIdentitySet >::SetNodeObject(cpid_set, node);
                    }

                    //add this id to the cp id set
                    cpid_set->AddID(id);
                }
                else
                {
                    //warning, abort!
                    kfmout<<"KFMCollocationPointIdentitySetCreator::SortCollocationPoints: Abort! Point not found in tree region. "<<kfmendl;
                    kfmexit(1);
                }
            }
        };

        void SortCollocationPoints(const std::vector< const KFMPoint<SpatialNDIM>* >* c_points, const std::vector<unsigned int>* c_point_ids)
        {
            //we assume that the placement of the collocation point in the list is its identity index
            unsigned int size = c_points->size();
            for(unsigned int i=0; i<size; i++)
            {
                unsigned int id = c_point_ids->at(i);
                fNavigator.SetPoint(c_points->at(i));
                fNavigator.ApplyAction(fRootNode);

                if(fNavigator.Found())
                {
                    KFMNode<ObjectTypeList>* node = fNavigator.GetLeafNode();

                    //we add this id to the leaf node's collocation point id list
                    KFMCollocationPointIdentitySet* cpid_set = NULL;
                    cpid_set = KFMObjectRetriever<ObjectTypeList, KFMCollocationPointIdentitySet >::GetNodeObject(node);

                    if(cpid_set == NULL)
                    {
                        //create this collocation point id set and add it to this node
                        cpid_set = new KFMCollocationPointIdentitySet();
                        KFMObjectRetriever<ObjectTypeList, KFMCollocationPointIdentitySet >::SetNodeObject(cpid_set, node);
                    }

                    //add this id to the cp id set
                    cpid_set->AddID(id);
                }
                else
                {
                    //warning, abort!
                    kfmout<<"KFMCollocationPointIdentitySetCreator::SortCollocationPoints: Abort! Point not found in tree region. "<<kfmendl;
                    kfmexit(1);
                }
            }
        };




    protected:
        /* data */

        KFMCubicSpaceTree<SpatialNDIM, ObjectTypeList>* fTree;
        KFMNode<ObjectTypeList>* fRootNode;
        KFMCubicSpaceTreeNavigator<ObjectTypeList, SpatialNDIM> fNavigator;

};


}

#endif /* __KFMCollocationPointIdentitySetCreator_H__ */
