#ifndef KTYPEOPERATION_H_
#define KTYPEOPERATION_H_

#include "KTypeList.h"
#include "KTypeLogic.h"

namespace katrin
{

//KLength function for typelists

//degenerate definition
template<class XTypeList> class KLength;

//basic definition
template<class XHeadType, class XTailType> class KLength<KTypeList<XHeadType, XTailType>>
{
  public:
    enum
    {
        Value = KLength<XTailType>::Value + 1
    };
};

//terminal definition
template<class XHeadType> class KLength<KTypeList<XHeadType, KTypeNull>>
{
  public:
    enum
    {
        Value = 1
    };
};


//AtFront function for typelists

//degenerate definition
template<class XTypeList> class KAtFront;

//basic definition
template<class XHeadType, class XTailType> class KAtFront<KTypeList<XHeadType, XTailType>>
{
  public:
    using Type = XHeadType;
};


//KAtBack function for typelists

//degenerate definition
template<class XTypeList> class KAtBack;

//basic definition
template<class XHeadType, class XTailType> class KAtBack<KTypeList<XHeadType, XTailType>>
{
  public:
    using Type = typename KAtBack<XTailType>::Type;
};

//terminal definiton (end of typelist reached)
template<class XHeadType> class KAtBack<KTypeList<XHeadType, KTypeNull>>
{
  public:
    using Type = XHeadType;
};


//KSAt function for typelists

//degenerate definition
template<class XTypeList, int XIndex> class KSAt;

//basic definition
template<class XHeadType, class XTailType, int XIndex> class KSAt<KTypeList<XHeadType, XTailType>, XIndex>
{
  public:
    using Type = typename KSAt<XTailType, XIndex - 1>::Type;
};

//terminal definition (index is reached)
template<class XHeadType, class XTailType> class KSAt<KTypeList<XHeadType, XTailType>, 0>
{
  public:
    using Type = XHeadType;
};

//terminal definition (end of typelist reached)
template<int XIndex> class KSAt<KTypeNull, XIndex>
{
  public:
    using Type = KTypeNull;
};


//KPushBack function for typelists

//degenerate definition
template<class XInserted, class XTarget> class KPushBack;

//basic definition (inserted is an undifferentiated type)
template<class XInserted, class XTypeAtIndex, class XNext> class KPushBack<XInserted, KTypeList<XTypeAtIndex, XNext>>
{
  public:
    using Type = KTypeList<XTypeAtIndex, typename KPushBack<XInserted, XNext>::Type>;
};

//basic definition (inserted is a typelist)
template<class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext>
class KPushBack<KTypeList<XFirstInserted, XNextInserted>, KTypeList<XTypeAtIndex, XNext>>
{
  public:
    using Type = KTypeList<XTypeAtIndex, typename KPushBack<KTypeList<XFirstInserted, XNextInserted>, XNext>::Type>;
};

//terminal definition (inserted is an undifferentiated type, end of typelist reached)
template<class XInserted, class XTypeAtIndex> class KPushBack<XInserted, KTypeList<XTypeAtIndex, KTypeNull>>
{
  public:
    using Type = KTypeList<XTypeAtIndex, KTypeList<XInserted, KTypeNull>>;
};

//terminal definition (inserted is a typelist, end of typelist reached)
template<class XFirstInserted, class XNextInserted, class XTypeAtIndex>
class KPushBack<KTypeList<XFirstInserted, XNextInserted>, KTypeList<XTypeAtIndex, KTypeNull>>
{
  public:
    using Type = KTypeList<XTypeAtIndex, KTypeList<XFirstInserted, XNextInserted>>;
};


//KPushFront function for typelists

//degenerate definition
template<class XInserted, class XTarget> class KPushFront;

//basic definition (inserted is an undifferentiated type)
template<class XInserted, class XTypeAtIndex, class XNext> class KPushFront<XInserted, KTypeList<XTypeAtIndex, XNext>>
{
  public:
    using Type = KTypeList<XInserted, KTypeList<XTypeAtIndex, XNext>>;
};

//basic definition (inserted is a typelist)
template<class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext>
class KPushFront<KTypeList<XFirstInserted, XNextInserted>, KTypeList<XTypeAtIndex, XNext>>
{
  public:
    using Type = KPushBack<KTypeList<XTypeAtIndex, XNext>, KTypeList<XFirstInserted, XNextInserted>>;
};

//terminal definition
template<class XTypeAtIndex, class XNext> class KPushFront<KTypeNull, KTypeList<XTypeAtIndex, XNext>>
{
  public:
    using Type = KTypeList<XTypeAtIndex, XNext>;
};
//terminal definition
template<class XTypeAtIndex, class XNext> class KPushFront<KTypeList<XTypeAtIndex, XNext>, KTypeNull>
{
  public:
    using Type = KTypeList<XTypeAtIndex, XNext>;
};
//terminal definition
template<> class KPushFront<KTypeNull, KTypeNull>
{
  public:
    using Type = KTypeNull;
};


//KInsert function for typelists

//degenerate definition
template<class XInserted, class XTarget, int XIndex> class KInsert;

//basic definition (inserted is an undifferentiated type, target is a typelist)
template<class XInserted, class XTypeAtIndex, class XNext, int XIndex>
class KInsert<XInserted, KTypeList<XTypeAtIndex, XNext>, XIndex>
{
  public:
    using Type = KTypeList<XTypeAtIndex, typename KInsert<XInserted, XNext, XIndex - 1>::Type>;
};

//basic definition (inserted is a typelist, target is a typelist)
template<class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext, int XIndex>
class KInsert<KTypeList<XFirstInserted, XNextInserted>, KTypeList<XTypeAtIndex, XNext>, XIndex>
{
  public:
    using Type =
        KTypeList<XTypeAtIndex, typename KInsert<KTypeList<XFirstInserted, XNextInserted>, XNext, XIndex - 1>::Type>;
};

//terminal definition (inserted is an undifferentiated type, target is a typelist, index is reached)
template<class XInserted, class XTypeAtIndex, class XNext> class KInsert<XInserted, KTypeList<XTypeAtIndex, XNext>, 0>
{
  public:
    using Type = KTypeList<XInserted, KTypeList<XTypeAtIndex, XNext>>;
};

//terminal definition (inserted is a typelist, target is a typelist, index is reached)
template<class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext>
class KInsert<KTypeList<XFirstInserted, XNextInserted>, KTypeList<XTypeAtIndex, XNext>, 0>
{
  public:
    using Type = typename KPushBack<KTypeList<XTypeAtIndex, XNext>, KTypeList<XFirstInserted, XNextInserted>>::Type;
};

//KHas function for typelists

//degenerate definition
template<class XQuery, class XTarget> class KHas;

//nonmatching definition
template<class XQuery, class XType, class XNext> class KHas<XQuery, KTypeList<XType, XNext>>
{
  public:
    enum
    {
        Value = KHas<XQuery, XNext>::Value
    };
};

//matching definition
template<class XQuery, class XNext> class KHas<XQuery, KTypeList<XQuery, XNext>>
{
  public:
    enum
    {
        Value = 1
    };
};

//terminal definition
template<class XQuery> class KHas<XQuery, KTypeNull>
{
  public:
    enum
    {
        Value = 0
    };
};

//KNotHas function for typelists

//degenerate definition
template<class XQuery, class XTarget> class KNotHas;

//nonmatching definition
template<class XQuery, class XType, class XNext> class KNotHas<XQuery, KTypeList<XType, XNext>>
{
  public:
    enum
    {
        Value = KNotHas<XQuery, XNext>::Value
    };
};

//matching definition
template<class XQuery, class XNext> class KNotHas<XQuery, KTypeList<XQuery, XNext>>
{
  public:
    enum
    {
        Value = 0
    };
};

//terminal definition
template<class XQuery> class KNotHas<XQuery, KTypeNull>
{
  public:
    enum
    {
        Value = 1
    };
};

//KUnique function for typelists

//degenerate definition
template<class XTarget> class KUnique;

template<class XHead, class XTail> class KUnique<KTypeList<XHead, XTail>>
{
  public:
    using Type = typename KTypeIf<KHas<XHead, typename KUnique<XTail>::Type>::Value, typename KUnique<XTail>::Type,
                                  KTypeList<XHead, typename KUnique<XTail>::Type>>::Type;
};

template<> class KUnique<KTypeNull>
{
  public:
    using Type = KTypeNull;
};
}  // namespace katrin

#endif
