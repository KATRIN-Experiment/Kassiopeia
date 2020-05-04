#ifndef KGObjectRetriever_HH__
#define KGObjectRetriever_HH__

#include "KGNode.hh"
#include "KGTypelist.hh"

namespace KGeoBag
{

/*
*
*@file KGObjectRetriever.hh
*@class KGObjectRetriever
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug 16 09:56:25 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename ObjectType> class KGObjectRetriever
{
  public:
    KGObjectRetriever(){};
    virtual ~KGObjectRetriever(){};

    static ObjectType* GetNodeObject(KGNode<ObjectTypeList>* node)
    {
        return static_cast<KGObjectHolder<ObjectType>*>(node)->fObject;
    }

    static const ObjectType* GetNodeObject(const KGNode<ObjectTypeList>* node)
    {
        return static_cast<KGObjectHolder<ObjectType>*>(node)->fObject;
    }

    static void SetNodeObject(ObjectType* obj_ptr, KGNode<ObjectTypeList>* node)
    {
        static_cast<KGObjectHolder<ObjectType>*>(node)->fObject = obj_ptr;
    }

  private:
};


}  // namespace KGeoBag


#endif /* KGObjectRetriever_H__ */
