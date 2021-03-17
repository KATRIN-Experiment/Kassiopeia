#ifndef KGTYPELIST_DEF
#define KGTYPELIST_DEF

// 3.3 Linearizing Typelist Creation

#define KGTYPELIST_1(T1)                             KGeoBag::KGTypelist<T1, KGeoBag::KGNullType>
#define KGTYPELIST_2(T1, T2)                         KGeoBag::KGTypelist<T1, KGTYPELIST_1(T2)>
#define KGTYPELIST_3(T1, T2, T3)                     KGeoBag::KGTypelist<T1, KGTYPELIST_2(T2, T3)>
#define KGTYPELIST_4(T1, T2, T3, T4)                 KGeoBag::KGTypelist<T1, KGTYPELIST_3(T2, T3, T4)>
#define KGTYPELIST_5(T1, T2, T3, T4, T5)             KGeoBag::KGTypelist<T1, KGTYPELIST_4(T2, T3, T4, T5)>
#define KGTYPELIST_6(T1, T2, T3, T4, T5, T6)         KGeoBag::KGTypelist<T1, KGTYPELIST_5(T2, T3, T4, T5, T6)>
#define KGTYPELIST_7(T1, T2, T3, T4, T5, T6, T7)     KGeoBag::KGTypelist<T1, KGTYPELIST_6(T2, T3, T4, T5, T6, T7)>
#define KGTYPELIST_8(T1, T2, T3, T4, T5, T6, T7, T8) KGeoBag::KGTypelist<T1, KGTYPELIST_7(T2, T3, T4, T5, T6, T7, T8)>
#define KGTYPELIST_9(T1, T2, T3, T4, T5, T6, T7, T8, T9)                                                               \
    KGeoBag::KGTypelist<T1, KGTYPELIST_8(T2, T3, T4, T5, T6, T7, T8, T9)>
#define KGTYPELIST_10(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)                                                         \
    KGeoBag::KGTypelist<T1, KGTYPELIST_9(T2, T3, T4, T5, T6, T7, T8, T9, T10)>
#define KGTYPELIST_11(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)                                                    \
    KGeoBag::KGTypelist<T1, KGTYPELIST_10(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)>
#define KGTYPELIST_12(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)                                               \
    KGeoBag::KGTypelist<T1, KGTYPELIST_11(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)>
#define KGTYPELIST_13(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)                                          \
    KGeoBag::KGTypelist<T1, KGTYPELIST_12(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)>
#define KGTYPELIST_14(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)                                     \
    KGeoBag::KGTypelist<T1, KGTYPELIST_13(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)>
#define KGTYPELIST_15(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)                                \
    KGeoBag::KGTypelist<T1, KGTYPELIST_14(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)>
#define KGTYPELIST_16(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16)                           \
    KGeoBag::KGTypelist<T1, KGTYPELIST_15(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16)>
#define KGTYPELIST_17(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17)                      \
    KGeoBag::KGTypelist<T1, KGTYPELIST_16(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17)>
#define KGTYPELIST_18(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18)                 \
    KGeoBag::KGTypelist<T1, KGTYPELIST_17(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18)>
#define KGTYPELIST_19(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19)            \
    KGeoBag::KGTypelist<                                                                                               \
        T1,                                                                                                            \
        KGTYPELIST_18(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19)>
#define KGTYPELIST_20(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20)       \
    KGeoBag::KGTypelist<                                                                                               \
        T1,                                                                                                            \
        KGTYPELIST_19(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20)>

namespace KGeoBag
{

/**
* @struct KGTypelist
*
* @brief KGeoBag's typelist implementation (taken from KGeoBag)
*
* KGTypelist is a reiteration of Alexandrescu's typelist construct (see "Modern
* C++ Design: Generic Programming and Design Patterns Applied" by Andrei
* Alexandrescu, 2001), with only a few modifications.
*
* @author T.J. Corona
*/

// 2.9 NullTypes & EmptyTypes

class KGNullType
{};
struct KGEmptyType
{};

// 3.2 Defining Typelists

template<class T, class U> struct KGTypelist
{
    using Head = T;
    using Tail = U;
};

// 3.4 Calculating Length

template<class TList> struct Length;
template<> struct Length<KGNullType>
{
    enum
    {
        value = 0
    };
};
template<class T, class U> struct Length<KGTypelist<T, U>>
{
    enum
    {
        value = 1 + Length<U>::value
    };
};

// 3.6 Indexed Access

template<class TList, unsigned int index> struct TypeAt;
template<class Head, class Tail> struct TypeAt<KGTypelist<Head, Tail>, 0>
{
    using Result = Head;
};
template<class Head, class Tail, unsigned int i> struct TypeAt<KGTypelist<Head, Tail>, i>
{
    using Result = typename TypeAt<Tail, i - 1>::Result;
};

// 3.7 Searching Typelists

template<class TList, class T> struct IndexOf;
template<class T> struct IndexOf<KGNullType, T>
{
    enum
    {
        value = -1
    };
};
template<class T, class Tail> struct IndexOf<KGTypelist<T, Tail>, T>
{
    enum
    {
        value = 0
    };
};
template<class Head, class Tail, class T> struct IndexOf<KGTypelist<Head, Tail>, T>
{
  private:
    enum
    {
        temp = IndexOf<Tail, T>::value
    };

  public:
    enum
    {
        value = (temp == -1 ? -1 : 1 + temp)
    };
};

// 3.8 Appending to Typelists

template<class TList, class T> struct Append;
template<> struct Append<KGNullType, KGNullType>
{
    using Result = KGNullType;
};
template<class T> struct Append<KGNullType, T>
{
    using Result = KGeoBag::KGTypelist<T, KGeoBag::KGNullType>;
};
template<class Head, class Tail> struct Append<KGNullType, KGTypelist<Head, Tail>>
{
    using Result = KGTypelist<Head, Tail>;
};
template<class Head, class Tail, class T> struct Append<KGTypelist<Head, Tail>, T>
{
    using Result = KGTypelist<Head, typename Append<Tail, T>::Result>;
};

// 3.9 Erasing a Type from a Typelist

template<class TList, class T> struct Erase;
template<class T> struct Erase<KGNullType, T>
{
    using Result = KGNullType;
};
template<class T, class Tail> struct Erase<KGTypelist<T, Tail>, T>
{
    using Result = Tail;
};
template<class Head, class Tail, class T> struct Erase<KGTypelist<Head, Tail>, T>
{
    using Result = KGTypelist<Head, typename Erase<Tail, T>::Result>;
};

// (My own code) Removing one typelist from another

template<class TList, class TListRemove> struct RemoveTypelist;
template<class TList> struct RemoveTypelist<TList, KGNullType>
{
    using Result = TList;
};
template<class TList, class Head, class Tail> struct RemoveTypelist<TList, KGTypelist<Head, Tail>>
{
    using Result = typename RemoveTypelist<typename Erase<TList, Head>::Result, Tail>::Result;
};

template<class TList, class T> struct EraseAll;
template<class T> struct EraseAll<KGNullType, T>
{
    using Result = KGNullType;
};
template<class T, class Tail> struct EraseAll<KGTypelist<T, Tail>, T>
{
    // Go all the way down the list removing the type
    using Result = typename EraseAll<Tail, T>::Result;
};
template<class Head, class Tail, class T> struct EraseAll<KGTypelist<Head, Tail>, T>
{
    // Go all the way down the list removing the type
    using Result = KGTypelist<Head, typename EraseAll<Tail, T>::Result>;
};

// 3.10 Erasing Duplicates

template<class TList> struct NoDuplicates;
template<> struct NoDuplicates<KGNullType>
{
    using Result = KGNullType;
};
template<class Head, class Tail> struct NoDuplicates<KGTypelist<Head, Tail>>
{
  private:
    using L1 = typename NoDuplicates<Tail>::Result;
    using L2 = typename Erase<L1, Head>::Result;

  public:
    using Result = KGTypelist<Head, L2>;
};

// 3.11 Replacing an Element in a Typelist
template<class TList, class T, class U> struct Replace;
template<class T, class U> struct Replace<KGNullType, T, U>
{
    using Result = KGNullType;
};
template<class T, class Tail, class U> struct Replace<KGTypelist<T, Tail>, T, U>
{
    using Result = KGTypelist<U, Tail>;
};
template<class Head, class Tail, class T, class U> struct Replace<KGTypelist<Head, Tail>, T, U>
{
    using Result = KGTypelist<Head, typename Replace<Tail, T, U>::Result>;
};

// 3.13 Class Generation with Typelists
template<class TList, template<class> class Unit> class KGGenScatterHierarchy;
// GenScatterHierarchy specialization: Typelist to Unit
template<class T1, class T2, template<class> class Unit>
class KGGenScatterHierarchy<KGTypelist<T1, T2>, Unit> :
    public KGGenScatterHierarchy<T1, Unit>,
    public KGGenScatterHierarchy<T2, Unit>
{
  public:
    using TList = KGTypelist<T1, T2>;
    using LeftBase = KGGenScatterHierarchy<T1, Unit>;
    using RightBase = KGGenScatterHierarchy<T2, Unit>;
    template<typename T> struct Rebind
    {
        using Result = Unit<T>;
    };
};
// Pass an atomic type (non-typelist) to Unit
template<class AtomicType, template<class> class Unit> class KGGenScatterHierarchy : public Unit<AtomicType>
{
    using LeftBase = Unit<AtomicType>;
    template<typename T> struct Rebind
    {
        using Result = Unit<T>;
    };
};
// Do nothing for NullType
template<template<class> class Unit> class KGGenScatterHierarchy<KGNullType, Unit>
{
    template<typename T> struct Rebind
    {
        using Result = Unit<T>;
    };
};

template<class T, class H> typename H::template Rebind<T>::Result& KField(H& obj)
{
    return obj;
}

template<class T, class H> const typename H::template Rebind<T>::Result& KField(const H& obj)
{
    return obj;
}

// GenScatterHierarchy with an additional template parameter
template<class TList, class Parameter, template<class, class> class Unit> class KGGenScatterHierarchyWithParameter;
// GenScatterHierarchy specialization: Typelist to Unit
template<class T1, class T2, class Parameter, template<class, class> class Unit>
class KGGenScatterHierarchyWithParameter<KGTypelist<T1, T2>, Parameter, Unit> :
    public KGGenScatterHierarchyWithParameter<T1, Parameter, Unit>,
    public KGGenScatterHierarchyWithParameter<T2, Parameter, Unit>
{
  public:
    using TList = KGTypelist<T1, T2>;
    using LeftBase = KGGenScatterHierarchyWithParameter<T1, Parameter, Unit>;
    using RightBase = KGGenScatterHierarchyWithParameter<T2, Parameter, Unit>;
    template<typename T> struct Rebind
    {
        using Result = Unit<T, Parameter>;
    };
};
// Pass an atomic type (non-typelist) to Unit
template<class AtomicType, class Parameter, template<class, class> class Unit>
class KGGenScatterHierarchyWithParameter : public Unit<AtomicType, Parameter>
{
    using LeftBase = Unit<AtomicType, Parameter>;
    template<typename T> struct Rebind
    {
        using Result = Unit<T, Parameter>;
    };
};
// Do nothing for NullType
template<class Parameter, template<class, class> class Unit>
class KGGenScatterHierarchyWithParameter<KGNullType, Parameter, Unit>
{
    template<typename T> struct Rebind
    {
        using Result = Unit<T, Parameter>;
    };
};

// GenLinearHierarchy
template<class TList, template<class AtomicType, class Base> class Unit, class Root = KGEmptyType>
class KGGenLinearHierarchy;

template<class T1, class T2, template<class, class> class Unit, class Root>
class KGGenLinearHierarchy<KGTypelist<T1, T2>, Unit, Root> : public Unit<T1, KGGenLinearHierarchy<T2, Unit, Root>>
{};

template<class T, template<class, class> class Unit, class Root>
class KGGenLinearHierarchy<KGTypelist<T, KGNullType>, Unit, Root> : public Unit<T, Root>
{};

// GenLinearHierarchy with an additional template parameter
template<class TList, class Parameter, template<class AtomicType, class, class Base> class Unit,
         class Root = KGEmptyType>
class KGGenLinearHierarchyWithParameter;

template<class T1, class T2, class Parameter, template<class, class, class> class Unit, class Root>
class KGGenLinearHierarchyWithParameter<KGTypelist<T1, T2>, Parameter, Unit, Root> :
    public Unit<T1, Parameter, KGGenLinearHierarchyWithParameter<T2, Parameter, Unit, Root>>
{};

template<class T, class Parameter, template<class, class, class> class Unit, class Root>
class KGGenLinearHierarchyWithParameter<KGTypelist<T, KGNullType>, Parameter, Unit, Root> :
    public Unit<T, Parameter, Root>
{};
}  // namespace KGeoBag

#endif /* KGTYPELIST_DEF */
