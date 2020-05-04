#ifndef KFMCubicSpaceTree_HH__
#define KFMCubicSpaceTree_HH__

#include "KFMCubicSpaceNodeProgenitor.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMNodeObjectNullifier.hh"
#include "KFMTree.hh"

namespace KEMField
{


/*
*
*@file KFMCubicSpaceTree.hh
*@class KFMCubicSpaceTree
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 23:09:58 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

//IMPORTANT!
//The ObjectTypeList must contain the types: KFMCubicSpaceTreeProperties<NDIM> and KFMCube<NDIM>


template<unsigned int NDIM, typename ObjectTypeList>  //this is the same typelist passed to the node type
class KFMCubicSpaceTree : public KFMTree<ObjectTypeList>
{
  public:
    KFMCubicSpaceTree()
    {
        fTreeProperties = new KFMCubicSpaceTreeProperties<NDIM>();
        fCompoundActor = new KFMCompoundActor<KFMNode<ObjectTypeList>>();
        fProgenitor = new KFMCubicSpaceNodeProgenitor<NDIM, ObjectTypeList>();
        fConditionalProgenitor = new KFMConditionalActor<KFMNode<ObjectTypeList>>();
        fConditionalProgenitor->SetOperationalActor(fProgenitor);

        //first actor in the compound actors list is always the conditional progenitor
        fCompoundActor->AddActor(fConditionalProgenitor);
    }

    ~KFMCubicSpaceTree() override
    {
        delete fTreeProperties;
        //reset the space tree properties pointer to null for all nodes
        KFMNodeObjectNullifier<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>> treePropertyNullifier;
        this->ApplyCorecursiveAction(&treePropertyNullifier);
        delete fCompoundActor;
        delete fProgenitor;
        delete fConditionalProgenitor;
    }

    //access to manipulate tree properties
    KFMCubicSpaceTreeProperties<NDIM>* GetTreeProperties()
    {
        return fTreeProperties;
    };

    //set the condition on the use of the progenitor
    void SetSubdivisionCondition(KFMInspectingActor<KFMNode<ObjectTypeList>>* inspector)
    {
        fConditionalProgenitor->SetInspectingActor(inspector);
    }

    void AddPostSubdivisionAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor)
    {
        //adds an additional action to be performed each node after the conditional progenator acts upon it
        fCompoundActor->AddActor(actor);
    }

    void ConstructTree()  //recursively apply the progenating action to the root node
    {
        this->ApplyRecursiveAction(
            fCompoundActor);  //compound actor contains the progenating actor and all post actions
    }

    KFMCubicSpaceNodeProgenitor<NDIM, ObjectTypeList>* GetProgenitor()
    {
        return fProgenitor;
    };


  private:
    KFMCompoundActor<KFMNode<ObjectTypeList>>* fCompoundActor;
    KFMConditionalActor<KFMNode<ObjectTypeList>>* fConditionalProgenitor;
    KFMCubicSpaceNodeProgenitor<NDIM, ObjectTypeList>* fProgenitor;
    KFMCubicSpaceTreeProperties<NDIM>* fTreeProperties;
};


}  // namespace KEMField


#endif /* KFMCubicSpaceTree_H__ */
