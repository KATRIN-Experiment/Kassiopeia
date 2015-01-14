#ifndef KSAAssociatedValuePODOutputNode_HH__
#define KSAAssociatedValuePODOutputNode_HH__

#include "KSACallbackTypes.hh"
#include "KSAPODOutputNode.hh"

#define AddKSAOutputFor(class,var,type) \
  node->AddChild(new KSAAssociatedValuePODOutputNode< class, type, &class::Get ## var>(std::string(#var), this) )

namespace KEMField{


/**
*
*@file KSAAssociatedValuePODOutputNode.hh
*@class KSAAssociatedValuePODOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 21:12:33 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename ReturnType, ReturnType (CallType::*memberFunction)() const>
class KSAAssociatedValuePODOutputNode: public KSAPODOutputNode< ReturnType >
{
    public:
        KSAAssociatedValuePODOutputNode(std::string name, const CallType* call_ptr):KSAPODOutputNode< ReturnType >(name)
        {
            KSAConstantReturnByValueGet< CallType, ReturnType, memberFunction > callback;
            KSAPODOutputNode< ReturnType >::SetValue(callback(call_ptr));
        };

        virtual ~KSAAssociatedValuePODOutputNode(){;};

    protected:

};



}//end of kemfield namespace

#endif /* KSAAssociatedValuePODOutputNode_H__ */
