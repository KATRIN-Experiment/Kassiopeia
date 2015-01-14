#ifndef __KFMNodeChildToParentFlagValueInitializer_H__
#define __KFMNodeChildToParentFlagValueInitializer_H__


#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMNodeFlags.hh"
#include "KFMNodeFlagValueInitializer.hh"


namespace KEMField
{

/**
*
*@file KFMNodeChildToParentFlagValueInitializer.hh
*@class KFMNodeChildToParentFlagValueInitializer
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 12 16:01:07 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, unsigned int NFLAGS>
class KFMNodeChildToParentFlagValueInitializer: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMNodeChildToParentFlagValueInitializer(){};
        virtual ~KFMNodeChildToParentFlagValueInitializer(){};

        void SetFlagIndex(unsigned int flag_index){fValueInitializer.SetFlagIndex(flag_index);};
        void SetFlagValue(char value){fValueInitializer.SetFlagValue(value);};

        virtual void ApplyAction( KFMNode<ObjectTypeList>* node)
        {
            //set the flag value for this node
            fValueInitializer.ApplyAction(node);

            //now succesively apply the value initializer to all parents of this node
            KFMNode<ObjectTypeList>* parent = node->GetParent();

            while(parent != NULL)
            {
                fValueInitializer.ApplyAction(parent);
                parent = parent->GetParent();
            }
        }


    protected:

        KFMNodeFlagValueInitializer<ObjectTypeList, NFLAGS> fValueInitializer;

};

}


#endif /* __KFMNodeChildToParentFlagValueInitializer_H__ */
