#ifndef KGTree_HH__
#define KGTree_HH__


#include "KGNode.hh"
#include "KGRecursiveActor.hh"
#include "KGCorecursiveActor.hh"
#include "KGInspectingActor.hh"
#include "KGCompoundActor.hh"
#include "KGConditionalActor.hh"

namespace KGeoBag
{


/*
*
*@file KGTree.hh
*@class KGTree
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 23:09:58 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> //this is the same typelist passed to the node type
class KGTree
{
    public:

        KGTree()
        {
            fRootNode = new KGNode<ObjectTypeList>();
            fRecursiveActor = new KGRecursiveActor< KGNode<ObjectTypeList> >();
            fCorecursiveActor = new KGCorecursiveActor< KGNode<ObjectTypeList> >();
        }

        virtual ~KGTree()
        {
            delete fRootNode;
            delete fRecursiveActor;
            delete fCorecursiveActor;
        }

        KGNode<ObjectTypeList>* GetRootNode(){return fRootNode;};
        const KGNode<ObjectTypeList>* GetRootNode() const {return fRootNode;};
        void ReplaceRootNode( KGNode<ObjectTypeList>* root)
        {
            delete fRootNode;
            fRootNode = root;
        }


        void ApplyRecursiveAction(KGNodeActor< KGNode<ObjectTypeList>  >* actor, bool isForward = true) //default direction is forward
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

        void ApplyCorecursiveAction(KGNodeActor< KGNode<ObjectTypeList>  >* actor)
        {
                fCorecursiveActor->SetOperationalActor(actor);
                fCorecursiveActor->ApplyAction(fRootNode);
        }


    private:


        KGNode<ObjectTypeList>* fRootNode;
        KGRecursiveActor< KGNode<ObjectTypeList> >* fRecursiveActor;
        KGCorecursiveActor< KGNode<ObjectTypeList> >* fCorecursiveActor;


};



}//end of KGeoBag


#endif /* KGTree_H__ */
