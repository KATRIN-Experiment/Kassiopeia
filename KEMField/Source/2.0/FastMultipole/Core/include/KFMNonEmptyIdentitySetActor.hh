#ifndef __KFMNonEmptyIdentitySetActor_H__
#define __KFMNonEmptyIdentitySetActor_H__

#include "KFMNode.hh"
#include "KFMInspectingActor.hh"
#include "KFMIdentitySet.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/**
*
*@file KFMNonEmptyIdentitySetActor.hh
*@class KFMNonEmptyIdentitySetActor
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul 14 11:05:31 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList>
class KFMNonEmptyIdentitySetActor: public KFMInspectingActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMNonEmptyIdentitySetActor(){};
        virtual ~KFMNonEmptyIdentitySetActor(){};

        virtual bool ConditionIsSatisfied( KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                KFMIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);
                if(set != NULL)
                {
                    if(set->GetSize() != 0)
                    {
                        return true;
                    }
                }
            }
            return false;
        }


    protected:
        /* data */
};


}

#endif /* __KFMNonEmptyIdentitySetActor_H__ */
