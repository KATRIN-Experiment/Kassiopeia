#ifndef KTYPEOPERATION_H_
#define KTYPEOPERATION_H_

#include "KTypeList.h"
#include "KTypeLogic.h"

namespace katrin
{

    //KLength function for typelists

    //degenerate definition
    template< class XTypeList >
    class KLength;

    //basic definition
    template< class XHeadType, class XTailType >
    class KLength< KTypeList< XHeadType, XTailType > >
    {
        public:
            enum { Value = KLength< XTailType >::Value + 1 };
    };

    //terminal definition
    template< class XHeadType >
    class KLength< KTypeList< XHeadType, KTypeNull > >
    {
        public:
            enum { Value = 1 };
    };


    //AtFront function for typelists

    //degenerate definition
    template< class XTypeList >
    class KAtFront;

    //basic definition
    template< class XHeadType, class XTailType >
    class KAtFront< KTypeList< XHeadType, XTailType > >
    {
        public:
            typedef XHeadType Type;
    };


    //KAtBack function for typelists

    //degenerate definition
    template< class XTypeList >
    class KAtBack;

    //basic definition
    template< class XHeadType, class XTailType >
    class KAtBack< KTypeList< XHeadType, XTailType > >
    {
        public:
            typedef typename KAtBack< XTailType >::Type Type;
    };

    //terminal definiton (end of typelist reached)
    template< class XHeadType >
    class KAtBack< KTypeList< XHeadType, KTypeNull > >
    {
        public:
            typedef XHeadType Type;
    };


    //KSAt function for typelists

    //degenerate definition
    template< class XTypeList, int XIndex >
    class KSAt;

    //basic definition
    template< class XHeadType, class XTailType, int XIndex >
    class KSAt< KTypeList< XHeadType, XTailType >, XIndex >
    {
        public:
            typedef typename KSAt< XTailType, XIndex - 1 >::Type Type;
    };

    //terminal definition (index is reached)
    template< class XHeadType, class XTailType >
    class KSAt< KTypeList< XHeadType, XTailType >, 0 >
    {
        public:
            typedef XHeadType Type;
    };

    //terminal definition (end of typelist reached)
    template< int XIndex >
    class KSAt< KTypeNull, XIndex >
    {
        public:
            typedef KTypeNull Type;
    };


    //KPushBack function for typelists

    //degenerate definition
    template< class XInserted, class XTarget >
    class KPushBack;

    //basic definition (inserted is an undifferentiated type)
    template< class XInserted, class XTypeAtIndex, class XNext >
    class KPushBack< XInserted, KTypeList< XTypeAtIndex, XNext > >
    {
        public:
            typedef KTypeList< XTypeAtIndex, typename KPushBack< XInserted, XNext >::Type > Type;
    };

    //basic definition (inserted is a typelist)
    template< class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext >
    class KPushBack< KTypeList< XFirstInserted, XNextInserted >, KTypeList< XTypeAtIndex, XNext > >
    {
        public:
            typedef KTypeList< XTypeAtIndex, typename KPushBack< KTypeList< XFirstInserted, XNextInserted >, XNext >::Type > Type;
    };

    //terminal definition (inserted is an undifferentiated type, end of typelist reached)
    template< class XInserted, class XTypeAtIndex >
    class KPushBack< XInserted, KTypeList< XTypeAtIndex, KTypeNull > >
    {
        public:
            typedef KTypeList< XTypeAtIndex, KTypeList< XInserted, KTypeNull > > Type;
    };

    //terminal definition (inserted is a typelist, end of typelist reached)
    template< class XFirstInserted, class XNextInserted, class XTypeAtIndex >
    class KPushBack< KTypeList< XFirstInserted, XNextInserted >, KTypeList< XTypeAtIndex, KTypeNull > >
    {
        public:
            typedef KTypeList< XTypeAtIndex, KTypeList< XFirstInserted, XNextInserted > > Type;
    };


    //KPushFront function for typelists

    //degenerate definition
    template< class XInserted, class XTarget >
    class KPushFront;

    //basic definition (inserted is an undifferentiated type)
    template< class XInserted, class XTypeAtIndex, class XNext >
    class KPushFront< XInserted, KTypeList< XTypeAtIndex, XNext > >
    {
        public:
            typedef KTypeList< XInserted, KTypeList< XTypeAtIndex, XNext > > Type;
    };

    //basic definition (inserted is a typelist)
    template< class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext >
    class KPushFront< KTypeList< XFirstInserted, XNextInserted >, KTypeList< XTypeAtIndex, XNext > >
    {
        public:
            typedef KPushBack< KTypeList< XTypeAtIndex, XNext >, KTypeList< XFirstInserted, XNextInserted > > Type;
    };

    //terminal definition
    template< class XTypeAtIndex, class XNext >
    class KPushFront< KTypeNull, KTypeList< XTypeAtIndex, XNext > >
    {
        public:
            typedef KTypeList< XTypeAtIndex, XNext > Type;
    };
    //terminal definition
    template< class XTypeAtIndex, class XNext >
    class KPushFront< KTypeList< XTypeAtIndex, XNext >, KTypeNull >
    {
        public:
            typedef KTypeList< XTypeAtIndex, XNext > Type;
    };
    //terminal definition
    template<>
    class KPushFront< KTypeNull, KTypeNull >
    {
        public:
            typedef KTypeNull Type;
    };


    //KInsert function for typelists

    //degenerate definition
    template< class XInserted, class XTarget, int XIndex >
    class KInsert;

    //basic definition (inserted is an undifferentiated type, target is a typelist)
    template< class XInserted, class XTypeAtIndex, class XNext, int XIndex >
    class KInsert< XInserted, KTypeList< XTypeAtIndex, XNext >, XIndex >
    {
        public:
            typedef KTypeList< XTypeAtIndex, typename KInsert< XInserted, XNext, XIndex - 1 >::Type > Type;
    };

    //basic definition (inserted is a typelist, target is a typelist)
    template< class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext, int XIndex >
    class KInsert< KTypeList< XFirstInserted, XNextInserted >, KTypeList< XTypeAtIndex, XNext >, XIndex >
    {
        public:
            typedef KTypeList< XTypeAtIndex, typename KInsert< KTypeList< XFirstInserted, XNextInserted >, XNext, XIndex - 1 >::Type > Type;
    };

    //terminal definition (inserted is an undifferentiated type, target is a typelist, index is reached)
    template< class XInserted, class XTypeAtIndex, class XNext >
    class KInsert< XInserted, KTypeList< XTypeAtIndex, XNext >, 0 >
    {
        public:
            typedef KTypeList< XInserted, KTypeList< XTypeAtIndex, XNext > > Type;
    };

    //terminal definition (inserted is a typelist, target is a typelist, index is reached)
    template< class XFirstInserted, class XNextInserted, class XTypeAtIndex, class XNext >
    class KInsert< KTypeList< XFirstInserted, XNextInserted >, KTypeList< XTypeAtIndex, XNext >, 0 >
    {
        public:
            typedef typename KPushBack< KTypeList< XTypeAtIndex, XNext >, KTypeList< XFirstInserted, XNextInserted > >::Type Type;
    };

    //KHas function for typelists

    //degenerate definition
    template< class XQuery, class XTarget >
    class KHas;

    //nonmatching definition
    template< class XQuery, class XType, class XNext >
    class KHas< XQuery, KTypeList< XType, XNext > >
    {
        public:
            enum
            {
                Value = KHas< XQuery, XNext >::Value
            };
    };

    //matching definition
    template< class XQuery, class XNext >
    class KHas< XQuery, KTypeList< XQuery, XNext > >
    {
        public:
            enum
            {
                Value = 1
            };
    };

    //terminal definition
    template< class XQuery >
    class KHas< XQuery, KTypeNull >
    {
        public:
            enum
            {
                Value = 0
            };
    };

    //KNotHas function for typelists

    //degenerate definition
    template< class XQuery, class XTarget >
    class KNotHas;

    //nonmatching definition
    template< class XQuery, class XType, class XNext >
    class KNotHas< XQuery, KTypeList< XType, XNext > >
    {
        public:
            enum
            {
                Value = KNotHas< XQuery, XNext >::Value
            };
    };

    //matching definition
    template< class XQuery, class XNext >
    class KNotHas< XQuery, KTypeList< XQuery, XNext > >
    {
        public:
            enum
            {
                Value = 0
            };
    };

    //terminal definition
    template< class XQuery >
    class KNotHas< XQuery, KTypeNull >
    {
        public:
            enum
            {
                Value = 1
            };
    };

    //KUnique function for typelists

    //degenerate definition
    template< class XTarget >
    class KUnique;

    template< class XHead, class XTail >
    class KUnique< KTypeList< XHead, XTail > >
    {
        public:
            typedef typename KTypeIf< KHas< XHead, typename KUnique< XTail >::Type >::Value, typename KUnique< XTail >::Type, KTypeList< XHead, typename KUnique< XTail >::Type > >::Type Type;

    };

    template<>
    class KUnique< KTypeNull >
    {
        public:
            typedef KTypeNull Type;
    };
}

#endif
