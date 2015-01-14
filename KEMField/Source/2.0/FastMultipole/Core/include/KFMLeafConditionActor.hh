#ifndef KFMLeafConditionActor_HH__
#define KFMLeafConditionActor_HH__

#include "KFMNode.hh"
#include "KFMInspectingActor.hh"

namespace KEMField
{

/*
*
*@file KFMLeafConditionActor.hh
*@class KFMLeafConditionActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 11:56:28 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType>
class KFMLeafConditionActor: public KFMInspectingActor<NodeType>
{
    public:
        KFMLeafConditionActor(){fSwitch = true;};
        virtual ~KFMLeafConditionActor(){};

        void SetTrueOnLeafNodes(){fSwitch = true;}
        void SetFalseOnLeafNodes(){fSwitch = false;};

        //needs to answer this question about whether this node statisfies a condition
        virtual bool ConditionIsSatisfied(NodeType* node)
        {
            if( node->HasChildren() )
            {
                return !fSwitch;
            }
            else
            {
                return fSwitch;
            }
        }


    private:

        bool fSwitch;

};


}

#endif /* KFMLeafConditionActor_H__ */
