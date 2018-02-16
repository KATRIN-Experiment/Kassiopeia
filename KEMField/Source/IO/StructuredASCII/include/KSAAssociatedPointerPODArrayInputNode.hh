#ifndef KSAAssociatedPointerPODArrayInputNode_HH__
#define KSAAssociatedPointerPODArrayInputNode_HH__

#include "KSAPODInputNode.hh"
#include "KSACallbackTypes.hh"

#define AddKSAInputForArray(class,var,type,size) \
  node->AddChild(new KSAAssociatedPointerPODArrayInputNode<class, type, &class::Set ## var ## Array>( std::string(#var), size, this) )

namespace KEMField{


/**
*
*@file KSAAssociatedPointerPODArrayInputNode.hh
*@class KSAAssociatedPointerPODArrayInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan  3 22:10:43 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/



template< typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType*) >
class KSAAssociatedPointerPODArrayInputNode: public KSAPODInputNode< std::vector<SetType> >
{
    public:

        KSAAssociatedPointerPODArrayInputNode(std::string name, unsigned int arr_size, CallType* call_ptr):KSAPODInputNode< std::vector< SetType > >(name),fArraySize(arr_size)
        {
            fCallPtr = call_ptr;
        };

        virtual ~KSAAssociatedPointerPODArrayInputNode(){;};

        void FinalizeObject()
        {
            fArray = new SetType[fArraySize];
            for(unsigned int i=0; i<fArraySize; i++)
            {
                fArray[i] = (this->fValue)[i];
            }

            fCallback(fCallPtr, fArray );

            delete[] fArray;
        }

    protected:

        CallType* fCallPtr;
        unsigned int fArraySize;
        SetType* fArray;
        KSAPassByConstantPointerSet< CallType, SetType, memberFunction > fCallback;

};


}

#endif /* KSAAssociatedPointerPODArrayInputNode_H__ */
