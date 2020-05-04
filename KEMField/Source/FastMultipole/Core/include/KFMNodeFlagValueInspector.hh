#ifndef __KFMNodeFlagValueInspector_H__
#define __KFMNodeFlagValueInspector_H__

#include "KFMInspectingActor.hh"
#include "KFMNode.hh"
#include "KFMNodeFlags.hh"
#include "KFMObjectRetriever.hh"


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

template<typename ObjectTypeList, unsigned int NFLAGS>
class KFMNodeFlagValueInspector : public KFMInspectingActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeFlagValueInspector()
    {
        fFlagIndex = 0;
        fValue = 0;
    };

    ~KFMNodeFlagValueInspector() override{};

    void SetFlagIndex(unsigned int flag_index)
    {
        fFlagIndex = flag_index;
    };
    void SetFlagValue(char value)
    {
        fValue = value;
    };

    //needs to answer this question about whether this node statisfies a condition
    bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            KFMNodeFlags<NFLAGS>* flags = KFMObjectRetriever<ObjectTypeList, KFMNodeFlags<NFLAGS>>::GetNodeObject(node);
            if (flags != nullptr) {
                if (flags->GetFlag(fFlagIndex) == fValue) {
                    return true;
                };
            }
        }

        return false;
    }


  protected:
    unsigned int fFlagIndex;
    char fValue;
};

}  // namespace KEMField

#endif /* __KFMNodeFlagValueInspector_H__ */
