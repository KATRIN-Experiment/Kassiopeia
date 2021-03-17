#ifndef KFMObjectRetriever_HH__
#define KFMObjectRetriever_HH__

#include "KFMNode.hh"
#include "KTypelist.hh"

namespace KEMField
{

/*
*
*@file KFMObjectRetriever.hh
*@class KFMObjectRetriever
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug 16 09:56:25 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename ObjectType> class KFMObjectRetriever
{
  public:
    KFMObjectRetriever() = default;
    ;
    virtual ~KFMObjectRetriever() = default;
    ;

    static ObjectType* GetNodeObject(KFMNode<ObjectTypeList>* node)
    {
        return static_cast<KFMObjectHolder<ObjectType>*>(node)->fObject;
    }

    static const ObjectType* GetNodeObject(const KFMNode<ObjectTypeList>* node)
    {
        return static_cast<KFMObjectHolder<ObjectType>*>(node)->fObject;
    }

    static void SetNodeObject(ObjectType* obj_ptr, KFMNode<ObjectTypeList>* node)
    {
        static_cast<KFMObjectHolder<ObjectType>*>(node)->fObject = obj_ptr;
    }

  private:
};


}  // namespace KEMField


#endif /* KFMObjectRetriever_H__ */
