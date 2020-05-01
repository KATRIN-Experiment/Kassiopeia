#ifndef KFMTree_HH__
#define KFMTree_HH__


#include "KFMCompoundActor.hh"
#include "KFMConditionalActor.hh"
#include "KFMCorecursiveActor.hh"
#include "KFMInspectingActor.hh"
#include "KFMNode.hh"
#include "KFMNodeFinder.hh"
#include "KFMRecursiveActor.hh"

namespace KEMField
{


/*
*
*@file KFMTree.hh
*@class KFMTree
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 23:09:58 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList>  //this is the same typelist passed to the node type
class KFMTree
{
  public:
    KFMTree()
    {
        fRootNode = new KFMNode<ObjectTypeList>();
        fRecursiveActor = new KFMRecursiveActor<KFMNode<ObjectTypeList>>();
        fCorecursiveActor = new KFMCorecursiveActor<KFMNode<ObjectTypeList>>();
        fRestricAction = false;
    }

    virtual ~KFMTree()
    {
        delete fRootNode;
        delete fRecursiveActor;
        delete fCorecursiveActor;
    }

    KFMNode<ObjectTypeList>* GetRootNode()
    {
        return fRootNode;
    };
    const KFMNode<ObjectTypeList>* GetRootNode() const
    {
        return fRootNode;
    };
    void ReplaceRootNode(KFMNode<ObjectTypeList>* root)
    {
        delete fRootNode;
        fRootNode = root;
    }

    void RestrictActionBehavior(bool restrictAction)
    {
        //if fRestricAction is false, then the recursive and co-recursive
        //actors will be applied to the entire tree
        //if it is false then the actions will be restricted to a specific
        //set of nodes
        fRestricAction = restrictAction;
    }

    void SetReducedActionCollection(std::vector<KFMNode<ObjectTypeList>*>* node_collection)
    {
        //the collection of nodes to which recursive and corecursive actions will
        //be applied in the case that fRestricAction is true

        //it is important that when this set is constructed that none of the
        //nodes contained in the collection are children, or sub-children
        //of another node in the set, otherwise nodes may be visited twice
        //or more by the same action

        //this set can change depending on the action being applied
        fReducedActionSet = *node_collection;
    }

    void ApplyRecursiveAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor,
                              bool isForward = true)  //default direction is forward
    {
        if (fRestricAction) {
            ApplyRestrictedRecursiveAction(actor, isForward);
        }
        else {
            ApplyFullRecursiveAction(actor, isForward);
        }
    }

    void ApplyCorecursiveAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor, bool isForward = true)
    {
        if (fRestricAction) {
            ApplyRestrictedCorecursiveAction(actor, isForward);
        }
        else {
            ApplyFullCorecursiveAction(actor, isForward);
        }
    }


    KFMNode<ObjectTypeList>* GetNodeByID(unsigned int id)
    {
        KFMNodeFinder<ObjectTypeList> node_finder;
        node_finder.SetID(id);
        fCorecursiveActor->SetOperationalActor(&node_finder);
        fCorecursiveActor->ApplyAction(fRootNode);
        return node_finder.GetNode();
    }


  protected:
    void ApplyFullRecursiveAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor, bool isForward)
    {
        if (isForward) {
            fRecursiveActor->VisitParentBeforeChildren();
            fRecursiveActor->SetOperationalActor(actor);
            fRecursiveActor->ApplyAction(fRootNode);
        }
        else {
            fRecursiveActor->VisitChildrenBeforeParent();
            fRecursiveActor->SetOperationalActor(actor);
            fRecursiveActor->ApplyAction(fRootNode);
        }
    }

    void ApplyFullCorecursiveAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor, bool isForward)
    {
        if (isForward) {
            fCorecursiveActor->VisitParentBeforeChildren();
            fCorecursiveActor->SetOperationalActor(actor);
            fCorecursiveActor->ApplyAction(fRootNode);
        }
        else {
            fCorecursiveActor->VisitChildrenBeforeParent();
            fCorecursiveActor->SetOperationalActor(actor);
            fCorecursiveActor->ApplyAction(fRootNode);
        }
    }

    void ApplyRestrictedRecursiveAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor, bool isForward)
    {
        unsigned int n_actionable_nodes = fReducedActionSet.size();
        for (unsigned int i = 0; i < n_actionable_nodes; i++) {
            if (isForward) {
                fRecursiveActor->VisitParentBeforeChildren();
                fRecursiveActor->SetOperationalActor(actor);
                fRecursiveActor->ApplyAction(fReducedActionSet[i]);
            }
            else {
                fRecursiveActor->VisitChildrenBeforeParent();
                fRecursiveActor->SetOperationalActor(actor);
                fRecursiveActor->ApplyAction(fReducedActionSet[i]);
            }
        }
    }

    void ApplyRestrictedCorecursiveAction(KFMNodeActor<KFMNode<ObjectTypeList>>* actor, bool isForward)
    {
        unsigned int n_actionable_nodes = fReducedActionSet.size();
        for (unsigned int i = 0; i < n_actionable_nodes; i++) {
            if (isForward) {
                fCorecursiveActor->VisitParentBeforeChildren();
                fCorecursiveActor->SetOperationalActor(actor);
                fCorecursiveActor->ApplyAction(fRootNode);
            }
            else {
                fCorecursiveActor->VisitChildrenBeforeParent();
                fCorecursiveActor->SetOperationalActor(actor);
                fCorecursiveActor->ApplyAction(fRootNode);
            }
        }
    }


    KFMNode<ObjectTypeList>* fRootNode;
    KFMRecursiveActor<KFMNode<ObjectTypeList>>* fRecursiveActor;
    KFMCorecursiveActor<KFMNode<ObjectTypeList>>* fCorecursiveActor;

    //used when restricting tree actions to specific node sub-sets
    bool fRestricAction;
    std::vector<KFMNode<ObjectTypeList>*> fReducedActionSet;
};


}  // namespace KEMField


#endif /* KFMTree_H__ */
