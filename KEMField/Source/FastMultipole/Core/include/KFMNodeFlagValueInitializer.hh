#ifndef __KFMNodeFlagValueInitializer_H__
#define __KFMNodeFlagValueInitializer_H__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMNodeFlags.hh"


namespace KEMField
{

/**
*
*@file KFMNodeFlagValueInitializer.hh
*@class KFMNodeFlagValueInitializer
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 12 16:01:07 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, unsigned int NFLAGS>
class KFMNodeFlagValueInitializer: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMNodeFlagValueInitializer(){};
        virtual ~KFMNodeFlagValueInitializer(){};

        void SetFlagIndex(unsigned int flag_index){fFlagIndex = flag_index;};
        void SetFlagValue(char value){fValue = value;};

        virtual void ApplyAction( KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                KFMNodeFlags<NFLAGS>* flags = KFMObjectRetriever<ObjectTypeList, KFMNodeFlags<NFLAGS> >::GetNodeObject(node);
                if(flags == NULL)
                {
                    flags = new KFMNodeFlags<NFLAGS>();
                    KFMObjectRetriever<ObjectTypeList, KFMNodeFlags<NFLAGS> >::SetNodeObject(flags, node);
                }

                flags->SetFlag(fFlagIndex, fValue );
            }
        }

    protected:

        unsigned int fFlagIndex;
        char fValue;


};

}


#endif /* __KFMNodeFlagValueInitializer_H__ */
