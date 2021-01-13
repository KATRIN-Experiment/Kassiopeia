#ifndef KSAAssociatedPointerPODOutputNode_HH__
#define KSAAssociatedPointerPODOutputNode_HH__


#include "KSACallbackTypes.hh"
#include "KSAPODOutputNode.hh"

namespace KEMField
{

#define AddKSAOutputForPointer(class, var, type)                                                                       \
    node->AddChild(new KSAAssociatedPointerPODOutputNode<class, type, &class ::Get##var>(std::string(#var), this))

/**
*
*@file KSAAssociatedPointerPODOutputNode.hh
*@class KSAAssociatedPointerPODOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 21:12:33 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename ReturnType, const ReturnType* (CallType::*memberFunction)() const>
class KSAAssociatedPointerPODOutputNode : public KSAPODOutputNode<ReturnType>
{
  public:
    KSAAssociatedPointerPODOutputNode(const std::string& name, const CallType* call_ptr) :
        KSAPODOutputNode<ReturnType>(name)
    {
        KSAConstantReturnByPointerGet<CallType, ReturnType, memberFunction> callback;
        KSAPODOutputNode<ReturnType>::SetValue(callback(call_ptr));
    }

    ~KSAAssociatedPointerPODOutputNode() override
    {
        ;
    };

  protected:
};


}  // namespace KEMField

#endif /* KSAAssociatedPointerPODOutputNode_H__ */
