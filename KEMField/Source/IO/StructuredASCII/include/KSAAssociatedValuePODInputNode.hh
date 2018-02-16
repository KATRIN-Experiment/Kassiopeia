#ifndef KSAAssociatedValuePODInputNode_HH__
#define KSAAssociatedValuePODInputNode_HH__

#include "KSAPODInputNode.hh"
#include "KSACallbackTypes.hh"

namespace KEMField{


/**
*
*@file KSAAssociatedValuePODInputNode.hh
*@class KSAAssociatedValuePODInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan  3 22:10:43 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/



template< typename CallType, typename SetType, void (CallType::*memberFunction)(SetType) >
class KSAAssociatedValuePODInputNode: public KSAPODInputNode<SetType>
{
    public:

        KSAAssociatedValuePODInputNode(std::string name, CallType* call_ptr):KSAPODInputNode< SetType >(name)
        {
            fCallPtr = call_ptr;
        };

        virtual ~KSAAssociatedValuePODInputNode(){;};

        void FinalizeObject()
        {
            fCallback(fCallPtr, this->fValue);
        }

    protected:

        CallType* fCallPtr;
        KSAPassByValueSet< CallType, SetType, memberFunction > fCallback;

};


}

#endif /* KSAAssociatedValuePODInputNode_H__ */
