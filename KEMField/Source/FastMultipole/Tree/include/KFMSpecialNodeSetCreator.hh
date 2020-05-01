#ifndef KFMSpecialNodeSetCreator_HH__
#define KFMSpecialNodeSetCreator_HH__


#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMSpecialNodeSet.hh"

#include <map>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMSpecialNodeSetCreator.hh
*@class KFMSpecialNodeSetCreator
*@brief this set creator must be controlled externally by a conditional actor
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Jun 29 15:37:47 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMSpecialNodeSetCreator : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMSpecialNodeSetCreator(){};
    virtual ~KFMSpecialNodeSetCreator(){};

    void SetSpecialNodeSet(KFMSpecialNodeSet<ObjectTypeList>* node_set)
    {
        fNodeSet = node_set;
    };

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            fNodeSet->AddNode(node);
        }
    }

  protected:
    KFMSpecialNodeSet<ObjectTypeList>* fNodeSet;
};

}  // namespace KEMField

#endif /* KFMSpecialNodeSetCreator_H__ */
