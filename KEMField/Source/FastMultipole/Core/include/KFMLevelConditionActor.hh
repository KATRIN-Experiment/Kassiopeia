#ifndef KFMLevelConditionActor_HH__
#define KFMLevelConditionActor_HH__

#include "KFMNode.hh"
#include "KFMInspectingActor.hh"

namespace KEMField
{

/*
*
*@file KFMLevelConditionActor.hh
*@class KFMLevelConditionActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 11:56:28 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType>
class KFMLevelConditionActor: public KFMInspectingActor<NodeType>
{
    public:
        KFMLevelConditionActor(){fLevel = 0; fSwitch = 0;};
        virtual ~KFMLevelConditionActor(){};

        void SetLevel(unsigned int l){fLevel = l;};

        void SetEqualOrGreaterThan(){fSwitch = 0;};
        void SetEqualOrLessThan(){fSwitch = 1;};
        void SetGreaterThan(){fSwitch = 2;};
        void SetLessThan(){fSwitch = 3;};

        //needs to answer this question about whether this node statisfies a condition
        virtual bool ConditionIsSatisfied(NodeType* node)
        {
            switch(fSwitch)
            {
                case 0:
                    return (node->GetLevel() >= fLevel);
                break;
                case 1:
                    return (node->GetLevel() <= fLevel);
                break;
                case 2:
                    return (node->GetLevel() > fLevel);
                break;
                case 3:
                    return (node->GetLevel() < fLevel);
                break;
                default:
                    return false;
                break;
            }

            return false;

        }


    private:

        unsigned int fLevel;
        unsigned int fSwitch;


};


}

#endif /* KFMLevelConditionActor_H__ */
