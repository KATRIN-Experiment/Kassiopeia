#ifndef KGConditionalActor_HH__
#define KGConditionalActor_HH__

#include "KGNodeActor.hh"
#include "KGInspectingActor.hh"

namespace KGeoBag
{

/*
*
*@file KGConditionalActor.hh
*@class KGConditionalActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 11:19:56 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType>
class KGConditionalActor: public KGNodeActor<NodeType>
{
    public:
        KGConditionalActor(){;};
        virtual ~KGConditionalActor(){;};


        //this visitor should not modify the state of the node
        //just determine if it satisfies a certain condition
        void SetInspectingActor(KGInspectingActor<NodeType>* inspectActor)
        {
            if(inspectActor != NULL)//avoid a disaster
            {
                fInspectingActor = inspectActor;
            }
        }


        //this visitor performs some sort of action on the node
        //if the inspecting visitor is satisfied
        void SetOperationalActor(KGNodeActor<NodeType>* opActor)
        {
            if(opActor != this && opActor != NULL)//avoid a disaster
            {
                fOperationalActor = opActor;
            }
        }


        virtual void ApplyAction(NodeType* node)
        {
            if( fInspectingActor->ConditionIsSatisfied(node) )
            {
                fOperationalActor->ApplyAction(node);
            }
        }


    private:

        KGNodeActor<NodeType>* fOperationalActor;
        KGInspectingActor<NodeType>* fInspectingActor;

};


}//end of KGeoBag

#endif /* KGConditionalActor_H__ */
