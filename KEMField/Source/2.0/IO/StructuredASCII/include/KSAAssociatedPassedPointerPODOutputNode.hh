#ifndef KSAAssociatedPassedPointerPODOutputNode_HH__
#define KSAAssociatedPassedPointerPODOutputNode_HH__


#include "KSACallbackTypes.hh"
#include "KSAPODOutputNode.hh"

namespace KEMField{


/**
*
*@file KSAAssociatedPassedPointerPODOutputNode.hh
*@class KSAAssociatedPassedPointerPODOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Dec 29 21:12:33 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename CallType, typename ReturnType, void (CallType::*memberFunction)(ReturnType* ) const >
class KSAAssociatedPassedPointerPODOutputNode: public KSAPODOutputNode< ReturnType >
{
    public:

        KSAAssociatedPassedPointerPODOutputNode(std::string name, const CallType* call_ptr):KSAPODOutputNode< ReturnType >( name )
        {
            KSAConstantReturnByPassedPointerGet< CallType, ReturnType, memberFunction > callback;
            ReturnType val;
            callback(call_ptr, &val);
            KSAPODOutputNode< ReturnType >::SetValue(&val);
        }

        virtual ~KSAAssociatedPassedPointerPODOutputNode(){;};

    protected:

};



}//end of kemfield namespace

#endif /* KSAAssociatedPassedPointerPODOutputNode_H__ */
