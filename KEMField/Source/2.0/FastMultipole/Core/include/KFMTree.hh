#ifndef KFMTree_HH__
#define KFMTree_HH__


#include "KFMNode.hh"
#include "KFMNodeFinder.hh"
#include "KFMRecursiveActor.hh"
#include "KFMCorecursiveActor.hh"
#include "KFMInspectingActor.hh"
#include "KFMCompoundActor.hh"
#include "KFMConditionalActor.hh"

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

template<typename ObjectTypeList> //this is the same typelist passed to the node type
class KFMTree
{
    public:

        KFMTree()
        {
            fRootNode = new KFMNode<ObjectTypeList>();
            fRecursiveActor = new KFMRecursiveActor< KFMNode<ObjectTypeList> >();
            fCorecursiveActor = new KFMCorecursiveActor< KFMNode<ObjectTypeList> >();
        }

        virtual ~KFMTree()
        {
            //TODO fix this memory leak!

            delete fRootNode;
            delete fRecursiveActor;
            delete fCorecursiveActor;
        }

        KFMNode<ObjectTypeList>* GetRootNode(){return fRootNode;};
        const KFMNode<ObjectTypeList>* GetRootNode() const {return fRootNode;};
        void ReplaceRootNode( KFMNode<ObjectTypeList>* root)
        {
            delete fRootNode; //TODO fix this memory leak
            fRootNode = root;
        }


        void ApplyRecursiveAction(KFMNodeActor< KFMNode<ObjectTypeList>  >* actor, bool isForward = true) //default direction is forward
        {
            if(isForward)
            {
                fRecursiveActor->VisitParentBeforeChildren();
                fRecursiveActor->SetOperationalActor(actor);
                fRecursiveActor->ApplyAction(fRootNode);
            }
            else
            {
                fRecursiveActor->VisitChildrenBeforeParent();
                fRecursiveActor->SetOperationalActor(actor);
                fRecursiveActor->ApplyAction(fRootNode);
            }
        }

        void ApplyCorecursiveAction(KFMNodeActor< KFMNode<ObjectTypeList>  >* actor)
        {
                fCorecursiveActor->SetOperationalActor(actor);
                fCorecursiveActor->ApplyAction(fRootNode);
        }

        KFMNode<ObjectTypeList>* GetNodeByID(unsigned int id)
        {
            KFMNodeFinder<ObjectTypeList> node_finder;
            node_finder.SetID(id);
            fCorecursiveActor->SetOperationalActor(&node_finder);
            fCorecursiveActor->ApplyAction(fRootNode);
            return node_finder.GetNode();
        }


    private:


        KFMNode<ObjectTypeList>* fRootNode;
        KFMRecursiveActor< KFMNode<ObjectTypeList> >* fRecursiveActor;
        KFMCorecursiveActor< KFMNode<ObjectTypeList> >* fCorecursiveActor;


};



}//end of KEMField


#endif /* KFMTree_H__ */
