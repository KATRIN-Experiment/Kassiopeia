#ifndef KFMCubicSpaceBallSorter_HH__
#define KFMCubicSpaceBallSorter_HH__

#include <cmath>

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMObjectContainer.hh"
#include "KFMIdentitySet.hh"

#include "KFMBall.hh"
#include "KFMInsertionCondition.hh"

namespace KEMField
{

/*
*
*@file KFMCubicSpaceBallSorter.hh
*@class KFMCubicSpaceBallSorter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Aug 25 19:32:36 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM, typename ObjectTypeList>
class KFMCubicSpaceBallSorter: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMCubicSpaceBallSorter(){};
        virtual ~KFMCubicSpaceBallSorter(){};

        void SetInsertionCondition(const KFMInsertionCondition<NDIM>* cond){fCondition = cond;};
        const KFMInsertionCondition<NDIM>* GetInsertionCondition(){return fCondition;};

        void SetBoundingBallContainer(const KFMObjectContainer< KFMBall<NDIM > >* ball_container){fBallContainer = ball_container;};

        virtual void ApplyAction( KFMNode< ObjectTypeList >* node)
        {

            //in this function we distribute the bounding balls owned by a node
            //on to its children depending on the insertion condition
            //it can only be applied to a node which has children

            if(node->HasChildren())
            {
                //get the bounding ball list class id's from the node
                KFMIdentitySet* bball_list = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet >::GetNodeObject(node);
                std::vector<unsigned int> original_id_list;
                bball_list->GetIDs(&original_id_list);

                //now we iterate over the node's list of bounding balls
                //and decide which ones should be pushed downwards to a child
                const KFMBall<NDIM>* bball;
                unsigned int n_children = node->GetNChildren();
                std::vector< std::vector<unsigned int> > child_id_list; //id list for each child
                std::vector< unsigned int > updated_id_list; //new id list for the parent after redistribution
                child_id_list.resize(n_children);
                KFMCube<NDIM>* child_cube;
                KFMNode<ObjectTypeList>* child;

                unsigned int list_size = original_id_list.size();
                for(unsigned int i=0; i<list_size; i++)
                {
                    bball = fBallContainer->GetObjectWithID(original_id_list[i]);
                    bool parent_owns_ball = true;

                    for(unsigned int j=0; j<n_children; j++)
                    {
                        child = node->GetChild(j);
                        child_cube = KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM> >::GetNodeObject(child);

                        if(fCondition->CanInsertBallInCube(bball, child_cube)) //if condition is satified then this (child) node now owns the bball
                        {
                            child_id_list[j].push_back(original_id_list[i]);
                            parent_owns_ball = false;
                            break;
                        }
                    }

                    if(parent_owns_ball)
                    {
                        updated_id_list.push_back(original_id_list[i]);
                    }
                }

                //now we update the bounding ball list for this node
                bball_list->SetIDs(&updated_id_list);

                //update/create the list for each child
                KFMIdentitySet* child_bball_list;
                for(unsigned int j=0; j<n_children; j++)
                {
                    child = node->GetChild(j);
                    child_bball_list = KFMObjectRetriever<ObjectTypeList,  KFMIdentitySet >::GetNodeObject(child);

                    delete child_bball_list; //delete it if it already exists

                    child_bball_list = new KFMIdentitySet();
                    child_bball_list->SetIDs( &(child_id_list[j]) );

                    KFMObjectRetriever<ObjectTypeList, KFMIdentitySet  >::SetNodeObject(child_bball_list, child);
                }

            }

        }


    private:

        const KFMObjectContainer< KFMBall<NDIM> >* fBallContainer;

        const KFMInsertionCondition<NDIM>* fCondition;

};


}

#endif /* KFMCubicSpaceBallSorter_H__ */
