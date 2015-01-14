#ifndef __KFMNodeFlagValueInspector_H__
#define __KFMNodeFlagValueInspector_H__

#include "KFMNode.hh"
#include "KFMInspectingActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMNodeFlags.hh"




namespace KEMField
{

/**
*
*@file KFMNodeFlagValueInspector.hh
*@class KFMNodeFlagValueInspector
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Jul 13 17:13:25 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList, unsigned int NFLAGS>
class KFMNodeFlagValueInspector: public KFMInspectingActor< KFMNode<ObjectTypeList> >
{
    public:

        KFMNodeFlagValueInspector()
        {
            fFlagIndex = 0;
            fValue = 0;
        };

        virtual ~KFMNodeFlagValueInspector(){};

        void SetFlagIndex(unsigned int flag_index){fFlagIndex = flag_index;};
        void SetFlagValue(char value){fValue = value;};

        //needs to answer this question about whether this node statisfies a condition
        virtual bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                KFMNodeFlags<NFLAGS>* flags = KFMObjectRetriever<ObjectTypeList, KFMNodeFlags<NFLAGS> >::GetNodeObject(node);
                if(flags != NULL)
                {
                    if(flags->GetFlag(fFlagIndex) == fValue){return true;};
                }
            }

            return false;

        }


    protected:

        unsigned int fFlagIndex;
        char fValue;


};

}

#endif /* __KFMNodeFlagValueInspector_H__ */
