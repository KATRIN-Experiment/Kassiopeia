#ifndef __KFMCompoundInspectingActor_H__
#define __KFMCompoundInspectingActor_H__

#include "KFMInspectingActor.hh"

#include <vector>

namespace KEMField
{

/**
*
*@file KFMCompoundInspectingActor.hh
*@class KFMCompoundInspectingActor
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul 14 11:54:53 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename NodeType> class KFMCompoundInspectingActor : public KFMInspectingActor<NodeType>
{
  public:
    KFMCompoundInspectingActor()
    {
        fAndOr = true;
    };

    ~KFMCompoundInspectingActor() override = default;
    ;

    void AddInspectingActor(KFMInspectingActor<NodeType>* actor)
    {
        if (actor != this) {
            fInspectingActors.push_back(actor);
        }
    };

    void UseAndCondition()
    {
        fAndOr = true;
    };
    void UseOrCondition()
    {
        fAndOr = false;
    };

    //needs to answer this question about whether this node statisfies a condition
    bool ConditionIsSatisfied(NodeType* node) override
    {
        bool result = fAndOr;

        for (unsigned int i = 0; i < fInspectingActors.size(); i++) {
            if (fAndOr) {
                //use and condition
                if (result && fInspectingActors[i]->ConditionIsSatisfied(node)) {
                    result = true;
                }
                else {
                    result = false;
                }
            }
            else {
                //use or condition
                if (result || fInspectingActors[i]->ConditionIsSatisfied(node)) {
                    result = true;
                }
                else {
                    result = false;
                }
            }
        }

        if (fInspectingActors.size() != 0) {
            return result;
        }
        else {
            return false;
        }
    }

  private:
    bool fAndOr;  //true = and, false = or;
    std::vector<KFMInspectingActor<NodeType>*> fInspectingActors;
};


}  // namespace KEMField

#endif /* __KFMCompoundInspectingActor_H__ */
