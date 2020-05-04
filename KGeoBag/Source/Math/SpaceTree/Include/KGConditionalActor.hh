#ifndef KGConditionalActor_HH__
#define KGConditionalActor_HH__

#include "KGInspectingActor.hh"
#include "KGNodeActor.hh"

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

template<typename NodeType> class KGConditionalActor : public KGNodeActor<NodeType>
{
  public:
    KGConditionalActor()
    {
        ;
    };
    ~KGConditionalActor() override
    {
        ;
    };


    //this visitor should not modify the state of the node
    //just determine if it satisfies a certain condition
    void SetInspectingActor(KGInspectingActor<NodeType>* inspectActor)
    {
        if (inspectActor != nullptr)  //avoid a disaster
        {
            fInspectingActor = inspectActor;
        }
    }


    //this visitor performs some sort of action on the node
    //if the inspecting visitor is satisfied
    void SetOperationalActor(KGNodeActor<NodeType>* opActor)
    {
        if (opActor != this && opActor != nullptr)  //avoid a disaster
        {
            fOperationalActor = opActor;
        }
    }


    void ApplyAction(NodeType* node) override
    {
        if (fInspectingActor->ConditionIsSatisfied(node)) {
            fOperationalActor->ApplyAction(node);
        }
    }


  private:
    KGNodeActor<NodeType>* fOperationalActor;
    KGInspectingActor<NodeType>* fInspectingActor;
};


}  // namespace KGeoBag

#endif /* KGConditionalActor_H__ */
