#ifndef KGSpaceTree_HH__
#define KGSpaceTree_HH__

#include "KGTree.hh"

#include "KGSpaceTreeProperties.hh"
#include "KGSpaceNodeProgenitor.hh"
#include "KGNodeObjectNullifier.hh"

namespace KGeoBag
{


/*
*
*@file KGSpaceTree.hh
*@class KGSpaceTree
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 23:09:58 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

//IMPORTANT!
//The ObjectTypeList must contain the types: KGSpaceTreeProperties<NDIM> and KGCube<NDIM>


template<unsigned int NDIM, typename ObjectTypeList> //this is the same typelist passed to the node type
class KGSpaceTree: public KGTree<ObjectTypeList>
{
    public:

        KGSpaceTree()
        {
            fTreeProperties = new KGSpaceTreeProperties<NDIM>();
            fCompoundActor = new KGCompoundActor< KGNode<ObjectTypeList> >();
            fProgenitor = new KGSpaceNodeProgenitor<NDIM,ObjectTypeList>();
            fConditionalProgenitor = new KGConditionalActor< KGNode<ObjectTypeList> >();
            fConditionalProgenitor->SetOperationalActor(fProgenitor);

            //first actor in the compound actors list is always the conditional progenitor
            fCompoundActor->AddActor(fConditionalProgenitor);

        }

        virtual ~KGSpaceTree()
        {
            delete fTreeProperties;
            //reset the space tree properties pointer to null for all nodes
            KGNodeObjectNullifier<ObjectTypeList, KGSpaceTreeProperties<NDIM> > treePropertyNullifier;
            this->ApplyCorecursiveAction(&treePropertyNullifier);
            delete fCompoundActor;
            delete fProgenitor;
            delete fConditionalProgenitor;
        }

        //access to manipulate tree properties
        KGSpaceTreeProperties<NDIM>* GetTreeProperties(){return fTreeProperties;};

        //set the condition on the use of the progenitor
        void SetSubdivisionCondition(KGInspectingActor< KGNode<ObjectTypeList> >* inspector)
        {
            fConditionalProgenitor->SetInspectingActor(inspector);
        }

        void AddPostSubdivisionAction(KGNodeActor< KGNode<ObjectTypeList> >* actor)
        {
            //adds an additional action to be performed each node after the conditional progenator acts upon it
            fCompoundActor->AddActor(actor);
        }

        void ConstructTree() //recursively apply the progenating action to the root node
        {
            this->ApplyRecursiveAction(fCompoundActor); //compound actor contains the progenating actor and all post actions
        }

        KGSpaceNodeProgenitor<NDIM,ObjectTypeList>* GetProgenitor(){return fProgenitor;};



    private:

        KGCompoundActor< KGNode<ObjectTypeList> >* fCompoundActor;
        KGConditionalActor< KGNode<ObjectTypeList> >* fConditionalProgenitor;
        KGSpaceNodeProgenitor<NDIM,ObjectTypeList>* fProgenitor;
        KGSpaceTreeProperties<NDIM>* fTreeProperties;

};



}//end of KGeoBag


#endif /* KGSpaceTree_H__ */
