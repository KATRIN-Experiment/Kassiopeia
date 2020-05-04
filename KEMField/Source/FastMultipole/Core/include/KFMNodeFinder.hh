#ifndef __KFMNodeFinder_H__
#define __KFMNodeFinder_H__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"

namespace KEMField
{

/**
*
*@file KFMNodeFinder.hh
*@class KFMNodeFinder
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jul 22 12:44:19 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMNodeFinder : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeFinder() : fID(0), fNode(NULL){};
    virtual ~KFMNodeFinder(){};

    void SetID(int id)
    {
        fID = id;
    };

    KFMNode<ObjectTypeList>* GetNode()
    {
        return fNode;
    };

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            if (node->GetID() == fID) {
                fNode = node;
            };
        }
    }


  protected:
    /* data */

    int fID;
    KFMNode<ObjectTypeList>* fNode;
};

}  // namespace KEMField

#endif /* __KFMNodeFinder_H__ */
