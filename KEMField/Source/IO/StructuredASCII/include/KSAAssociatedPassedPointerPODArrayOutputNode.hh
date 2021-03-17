#ifndef KSAAssociatedPassedPointerPODArrayOutputNode_HH__
#define KSAAssociatedPassedPointerPODArrayOutputNode_HH__


#include "KSACallbackTypes.hh"
#include "KSAPODArrayOutputNode.hh"

#define AddKSAOutputForArray(class, var, type, size)                                                                   \
    node->AddChild(                                                                                                    \
        new KSAAssociatedPassedPointerPODArrayOutputNode<class, type, &class ::Get##var##Array>(std::string(#var),     \
                                                                                                size,                  \
                                                                                                this))


namespace KEMField
{


/**
*
*@file KSAAssociatedPassedPointerPODArrayOutputNode.hh
*@class KSAAssociatedPassedPointerPODArrayOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 21:12:33 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename ReturnType, void (CallType::*memberFunction)(ReturnType*) const>
class KSAAssociatedPassedPointerPODArrayOutputNode : public KSAPODArrayOutputNode<ReturnType>
{
  public:
    KSAAssociatedPassedPointerPODArrayOutputNode(const std::string& name, unsigned int arr_size,
                                                 const CallType* call_ptr) :
        KSAPODArrayOutputNode<ReturnType>(name, arr_size)
    {
        KSAConstantReturnByPassedPointerGet<CallType, ReturnType, memberFunction> callback;
        auto* val = new ReturnType[this->fArraySize];
        callback(call_ptr, val);
        KSAPODArrayOutputNode<ReturnType>::SetValue(val);
        delete[] val;
    }

    ~KSAAssociatedPassedPointerPODArrayOutputNode() override
    {
        ;
    };

  protected:
};


}  // namespace KEMField

#endif /* KSAAssociatedPassedPointerPODArrayOutputNode_H__ */
