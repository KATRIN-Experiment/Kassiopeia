#ifndef KSAAssociatedPointerPODInputNode_HH__
#define KSAAssociatedPointerPODInputNode_HH__

#include "KSACallbackTypes.hh"
#include "KSAPODInputNode.hh"

#define AddKSAInputForPointer(class, var, type)                                                                        \
    node->AddChild(new KSAAssociatedPointerPODInputNode<class, type, &class ::Set##var>(std::string(#var), this))

namespace KEMField
{


/**
*
*@file KSAAssociatedPointerPODInputNode.hh
*@class KSAAssociatedPointerPODInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan  3 22:10:43 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType*)>
class KSAAssociatedPointerPODInputNode : public KSAPODInputNode<SetType>
{
  public:
    KSAAssociatedPointerPODInputNode(const std::string& name, CallType* call_ptr) : KSAPODInputNode<SetType>(name)
    {
        fCallPtr = call_ptr;
    };

    ~KSAAssociatedPointerPODInputNode() override
    {
        ;
    };

    void FinalizeObject() override
    {
        fCallback(fCallPtr, &(this->fValue));
    }

  protected:
    CallType* fCallPtr;
    KSAPassByConstantPointerSet<CallType, SetType, memberFunction> fCallback;
};


}  // namespace KEMField

#endif /* KSAAssociatedPointerPODInputNode_H__ */
