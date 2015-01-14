#ifndef KGLeafConditionActor_HH__
#define KGLeafConditionActor_HH__

#include "KGNode.hh"
#include "KGInspectingActor.hh"

namespace KGeoBag
{

/*
*
*@file KGLeafConditionActor.hh
*@class KGLeafConditionActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 11:56:28 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType>
class KGLeafConditionActor: public KGInspectingActor<NodeType>
{
    public:
        KGLeafConditionActor(){fSwitch = true;};
        virtual ~KGLeafConditionActor(){};

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

#endif /* KGLeafConditionActor_H__ */
