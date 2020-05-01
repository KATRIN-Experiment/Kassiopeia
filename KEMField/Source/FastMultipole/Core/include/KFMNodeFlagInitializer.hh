#ifndef __KFMNodeFlagInitializer_H__
#define __KFMNodeFlagInitializer_H__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMNodeFlags.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/**
*
*@file KFMNodeFlagInitializer.hh
*@class KFMNodeFlagInitializer
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Jul  6 16:02:46 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, unsigned int NFLAGS>
class KFMNodeFlagInitializer : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeFlagInitializer(){};
    virtual ~KFMNodeFlagInitializer(){};

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            KFMNodeFlags<NFLAGS>* flags = KFMObjectRetriever<ObjectTypeList, KFMNodeFlags<NFLAGS>>::GetNodeObject(node);
            if (flags == NULL) {
                flags = new KFMNodeFlags<NFLAGS>();
                KFMObjectRetriever<ObjectTypeList, KFMNodeFlags<NFLAGS>>::SetNodeObject(flags, node);
            }
        }
    }

  protected:
};

}  // namespace KEMField

#endif /* __KFMNodeFlagInitializer_H__ */
