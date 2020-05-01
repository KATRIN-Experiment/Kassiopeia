#ifndef KSAAssociatedPointerPODArrayOutputNode_HH__
#define KSAAssociatedPointerPODArrayOutputNode_HH__

#include "KSACallbackTypes.hh"
#include "KSAPODArrayOutputNode.hh"

namespace KEMField
{


/**
*
*@file KSAAssociatedPointerPODArrayOutputNode.hh
*@class KSAAssociatedPointerPODArrayOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 21:12:33 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename ReturnType, const ReturnType* (CallType::*memberFunction)() const>
class KSAAssociatedPointerPODArrayOutputNode : public KSAPODArrayOutputNode<ReturnType>
{
  public:
    KSAAssociatedPointerPODArrayOutputNode(std::string name, unsigned int arr_size, const CallType* call_ptr) :
        KSAPODArrayOutputNode<ReturnType>(name, arr_size)
    {
        KSAConstantReturnByPointerGet<CallType, ReturnType, memberFunction> callback;
        KSAPODArrayOutputNode<ReturnType>::SetValue(callback(call_ptr));
    }

    virtual ~KSAAssociatedPointerPODArrayOutputNode()
    {
        ;
    };

  protected:
};


}  // namespace KEMField

#endif /* KSAAssociatedPointerPODArrayOutputNode_H__ */
