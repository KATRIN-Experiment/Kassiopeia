#ifndef KFMConditionalActor_HH__
#define KFMConditionalActor_HH__

#include "KFMNodeActor.hh"
#include "KFMInspectingActor.hh"

namespace KEMField
{

/*
*
*@file KFMConditionalActor.hh
*@class KFMConditionalActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 11:19:56 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType>
class KFMConditionalActor: public KFMNodeActor<NodeType>
{
    public:
        KFMConditionalActor(){;};
        virtual ~KFMConditionalActor(){;};


        //this visitor should not modify the state of the node
        //just determine if it satisfies a certain condition
        void SetInspectingActor(KFMInspectingActor<NodeType>* inspectActor)
        {
            if(inspectActor != NULL)//avoid a disaster
            {
                fInspectingActor = inspectActor;
            }
        }


        //this visitor performs some sort of action on the node
        //if the inspecting visitor is satisfied
        void SetOperationalActor(KFMNodeActor<NodeType>* opActor)
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

        KFMNodeActor<NodeType>* fOperationalActor;
        KFMInspectingActor<NodeType>* fInspectingActor;

};


}//end of KEMField

#endif /* KFMConditionalActor_H__ */
