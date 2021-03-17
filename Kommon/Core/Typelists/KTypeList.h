#ifndef KTYPELIST_H_
#define KTYPELIST_H_

#include "KTypeNull.h"

namespace katrin
{

//KTypeList type

//degenerate definition
template<class XHeadType, class XTailType> class KTypeList;


//basic definition
template<class XHeadType, class XNextHeadType, class XNextTailType>
class KTypeList<XHeadType, KTypeList<XNextHeadType, XNextTailType>>
{
  public:
    using HeadType = XHeadType;
    using TailTypes = KTypeList<XNextHeadType, XNextTailType>;
};

//terminal definiton
template<class XHeadType> class KTypeList<XHeadType, KTypeNull>
{
  public:
    using HeadType = XHeadType;
    using TailTypes = KTypeNull;
};

////////////////////////////////////////////////////////////////////////////////

// The following section of code pertaining to KGenScatterHierarchy is
// copied directly from /KEMField/Source/2.0/Core/KTypeList.hh
// and is a reiteration of Alexandrescu's typelist construct (see "Modern
// C++ Design: Generic Programming and Design Patterns Applied" by Andrei
// Alexandrescu, 2001), with only a few modifications.
// 3.13 Class Generation with Typelists
template<class TList, template<class> class Unit> class KGenScatterHierarchy;
// GenScatterHierarchy specialization: Typelist to Unit
template<class T1, class T2, template<class> class Unit>
class KGenScatterHierarchy<KTypeList<T1, T2>, Unit> :
    public KGenScatterHierarchy<T1, Unit>,
    public KGenScatterHierarchy<T2, Unit>
{
  public:
    using TList = KTypeList<T1, T2>;
    using LeftBase = KGenScatterHierarchy<T1, Unit>;
    using RightBase = KGenScatterHierarchy<T2, Unit>;
    template<typename T> struct Rebind
    {
        using Result = Unit<T>;
    };
};
// Pass an atomic type (non-typelist) to Unit
template<class AtomicType, template<class> class Unit> class KGenScatterHierarchy : public Unit<AtomicType>
{
    using LeftBase = Unit<AtomicType>;
    template<typename T> struct Rebind
    {
        using Result = Unit<T>;
    };
};
// Do nothing for NullType
template<template<class> class Unit> class KGenScatterHierarchy<KTypeNull, Unit>
{
    template<typename T> struct Rebind
    {
        using Result = Unit<T>;
    };
};

////////////////////////////////////////////////////////////////////////////////


//KTypeList construction macros
#define KTYPELIST0()                                 KTypeNull
#define KTYPELIST1(one)                              ::katrin::KTypeList<one, KTypeNull>
#define KTYPELIST2(one, two)                         ::katrin::KTypeList<one, KTYPELIST1(two)>
#define KTYPELIST3(one, two, three)                  ::katrin::KTypeList<one, KTYPELIST2(two, three)>
#define KTYPELIST4(one, two, three, four)            ::katrin::KTypeList<one, KTYPELIST3(two, three, four)>
#define KTYPELIST5(one, two, three, four, five)      ::katrin::KTypeList<one, KTYPELIST4(two, three, four, five)>
#define KTYPELIST6(one, two, three, four, five, six) ::katrin::KTypeList<one, KTYPELIST5(two, three, four, five, six)>
#define KTYPELIST7(one, two, three, four, five, six, seven)                                                            \
    ::katrin::KTypeList<one, KTYPELIST6(two, three, four, five, six, seven)>
#define KTYPELIST8(one, two, three, four, five, six, seven, eight)                                                     \
    ::katrin::KTypeList<one, KTYPELIST7(two, three, four, five, six, seven, eight)>
#define KTYPELIST9(one, two, three, four, five, six, seven, eight, nine)                                               \
    ::katrin::KTypeList<one, KTYPELIST8(two, three, four, five, six, seven, eight, nine)>
#define KTYPELIST10(one, two, three, four, five, six, seven, eight, nine, ten)                                         \
    ::katrin::KTypeList<one, KTYPELIST9(two, three, four, five, six, seven, eight, nine, ten)>
#define KTYPELIST11(one, two, three, four, five, six, seven, eight, nine, ten, eleven)                                 \
    ::katrin::KTypeList<one, KTYPELIST10(two, three, four, five, six, seven, eight, nine, ten, eleven)>
#define KTYPELIST12(one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve)                         \
    ::katrin::KTypeList<one, KTYPELIST11(two, three, four, five, six, seven, eight, nine, ten, eleven, twelve)>
#define KTYPELIST13(one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen)               \
    ::katrin::KTypeList<one,                                                                                           \
                        KTYPELIST12(two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen)>
#define KTYPELIST14(one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen)     \
    ::katrin::KTypeList<                                                                                               \
        one,                                                                                                           \
        KTYPELIST13(two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen)>
#define KTYPELIST15(one,                                                                                               \
                    two,                                                                                               \
                    three,                                                                                             \
                    four,                                                                                              \
                    five,                                                                                              \
                    six,                                                                                               \
                    seven,                                                                                             \
                    eight,                                                                                             \
                    nine,                                                                                              \
                    ten,                                                                                               \
                    eleven,                                                                                            \
                    twelve,                                                                                            \
                    thirteen,                                                                                          \
                    fourteen,                                                                                          \
                    fifteen)                                                                                           \
    ::katrin::KTypeList<one,                                                                                           \
                        KTYPELIST14(two,                                                                               \
                                    three,                                                                             \
                                    four,                                                                              \
                                    five,                                                                              \
                                    six,                                                                               \
                                    seven,                                                                             \
                                    eight,                                                                             \
                                    nine,                                                                              \
                                    ten,                                                                               \
                                    eleven,                                                                            \
                                    twelve,                                                                            \
                                    thirteen,                                                                          \
                                    fourteen,                                                                          \
                                    fifteen)>

}  // namespace katrin

#endif
