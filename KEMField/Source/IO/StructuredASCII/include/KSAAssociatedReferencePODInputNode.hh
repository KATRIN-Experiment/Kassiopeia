#ifndef KSAAssociatedReferencePODInputNode_HH__
#define KSAAssociatedReferencePODInputNode_HH__

#include "KSACallbackTypes.hh"
#include "KSAPODInputNode.hh"

#define AddKSAInputFor(class, var, type)                                                                               \
    node->AddChild(new KSAAssociatedReferencePODInputNode<class, type, &class ::Set##var>(std::string(#var), this))

namespace KEMField
{


/**
*
*@file KSAAssociatedReferencePODInputNode.hh
*@class KSAAssociatedReferencePODInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan  3 22:10:43 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType&)>
class KSAAssociatedReferencePODInputNode : public KSAPODInputNode<SetType>
{
  public:
    KSAAssociatedReferencePODInputNode(const std::string& name, CallType* call_ptr) : KSAPODInputNode<SetType>(name)
    {
        fCallPtr = call_ptr;
    };

    ~KSAAssociatedReferencePODInputNode() override
    {
        ;
    };

    void FinalizeObject() override
    {
        fCallback(fCallPtr, this->fValue);
    }

  protected:
    CallType* fCallPtr;
    KSAPassByConstantReferenceSet<CallType, SetType, memberFunction> fCallback;
};


}  // namespace KEMField

#endif /* KSAAssociatedReferencePODInputNode_H__ */
