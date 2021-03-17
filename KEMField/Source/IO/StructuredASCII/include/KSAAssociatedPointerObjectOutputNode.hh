#ifndef KSAAssociatedPointerObjectOutputNode_HH__
#define KSAAssociatedPointerObjectOutputNode_HH__

#include "KSACallbackTypes.hh"
#include "KSAObjectOutputNode.hh"

namespace KEMField
{


/**
*
*@file KSAAssociatedPointerObjectOutputNode.hh
*@class KSAAssociatedPointerObjectOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Dec 30 23:07:40 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename ReturnType, const ReturnType* (CallType::*memberFunction)() const>
class KSAAssociatedPointerObjectOutputNode :
    public KSAObjectOutputNode<ReturnType, KSAIsDerivedFrom<ReturnType, KSAFixedSizeInputOutputObject>::Is>
{
  public:
    KSAAssociatedPointerObjectOutputNode(const std::string& name, const CallType* call_ptr) :
        KSAObjectOutputNode<ReturnType, KSAIsDerivedFrom<ReturnType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        KSAConstantReturnByPointerGet<CallType, ReturnType, memberFunction> callback;
        KSAObjectOutputNode<ReturnType, KSAIsDerivedFrom<ReturnType, KSAFixedSizeInputOutputObject>::Is>::
            AttachObjectToNode(callback(call_ptr));
    }

    ~KSAAssociatedPointerObjectOutputNode() override
    {
        ;
    };


  protected:
};


}  // namespace KEMField


#endif /* KSAAssociatedPointerObjectOutputNode_H__ */
