#ifndef KTYPELIST_DEF
#define KTYPELIST_DEF

// 3.3 Linearizing Typelist Creation

#define KTYPELIST_1(T1) \
  KEMField::KTypelist<T1,KEMField::KNullType>
#define KTYPELIST_2(T1,T2) \
  KEMField::KTypelist<T1,KTYPELIST_1(T2) >
#define KTYPELIST_3(T1,T2,T3) \
  KEMField::KTypelist<T1,KTYPELIST_2(T2,T3) >
#define KTYPELIST_4(T1,T2,T3,T4) \
  KEMField::KTypelist<T1,KTYPELIST_3(T2,T3,T4) >
#define KTYPELIST_5(T1,T2,T3,T4,T5) \
  KEMField::KTypelist<T1,KTYPELIST_4(T2,T3,T4,T5) >
#define KTYPELIST_6(T1,T2,T3,T4,T5,T6) \
  KEMField::KTypelist<T1,KTYPELIST_5(T2,T3,T4,T5,T6) >
#define KTYPELIST_7(T1,T2,T3,T4,T5,T6,T7) \
  KEMField::KTypelist<T1,KTYPELIST_6(T2,T3,T4,T5,T6,T7) >
#define KTYPELIST_8(T1,T2,T3,T4,T5,T6,T7,T8) \
  KEMField::KTypelist<T1,KTYPELIST_7(T2,T3,T4,T5,T6,T7,T8) >
#define KTYPELIST_9(T1,T2,T3,T4,T5,T6,T7,T8,T9) \
  KEMField::KTypelist<T1,KTYPELIST_8(T2,T3,T4,T5,T6,T7,T8,T9) >
#define KTYPELIST_10(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10) \
  KEMField::KTypelist<T1,KTYPELIST_9(T2,T3,T4,T5,T6,T7,T8,T9,T10) >
#define KTYPELIST_11(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11) \
  KEMField::KTypelist<T1,KTYPELIST_10(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11) >
#define KTYPELIST_12(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12) \
  KEMField::KTypelist<T1,KTYPELIST_11(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12) >
#define KTYPELIST_13(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13) \
  KEMField::KTypelist<T1,KTYPELIST_12(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13) >
#define KTYPELIST_14(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14) \
  KEMField::KTypelist<T1,KTYPELIST_13(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14) >
#define KTYPELIST_15(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15) \
  KEMField::KTypelist<T1,KTYPELIST_14(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15) >
#define KTYPELIST_16(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16) \
  KEMField::KTypelist<T1,KTYPELIST_15(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16) >
#define KTYPELIST_17(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17) \
  KEMField::KTypelist<T1,KTYPELIST_16(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17) >
#define KTYPELIST_18(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18) \
  KEMField::KTypelist<T1,KTYPELIST_17(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18) >
#define KTYPELIST_19(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19) \
  KEMField::KTypelist<T1,KTYPELIST_18(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19) >
#define KTYPELIST_20(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20) \
  KEMField::KTypelist<T1,KTYPELIST_19(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20) >

namespace KEMField
{

/**
* @struct KTypelist
*
* @brief KEMField's typelist implementation.
*
* KTypelist is a reiteration of Alexandrescu's typelist construct (see "Modern
* C++ Design: Generic Programming and Design Patterns Applied" by Andrei
* Alexandrescu, 2001), with only a few modifications.
*
* @author T.J. Corona
*/

  // 2.9 NullTypes & EmptyTypes

  class KNullType {};
  struct KEmptyType {};

  // 3.2 Defining Typelists

  template <class T, class U>
  struct KTypelist
  {
    typedef T Head;
    typedef U Tail;
  };

  // 3.4 Calculating Length

  template <class TList> struct Length;
  template <> struct Length<KNullType>
  {
    enum { value = 0 };
  };
  template <class T, class U>
  struct Length< KTypelist<T,U> >
  {
    enum {value = 1 + Length<U>::value };
  };

  // 3.6 Indexed Access

  template <class TList, unsigned int index> struct TypeAt;
  template <class Head, class Tail>
  struct TypeAt<KTypelist<Head,Tail>,0>
  {
    typedef Head Result;
  };
  template <class Head, class Tail, unsigned int i>
  struct TypeAt<KTypelist<Head, Tail>, i>
  {
    typedef typename TypeAt<Tail,i-1>::Result Result;
  };

  // 3.7 Searching Typelists

  template <class TList, class T> struct IndexOf;
  template <class T>
  struct IndexOf<KNullType, T>
  {
    enum { value = -1 };
  };
  template <class T, class Tail>
  struct IndexOf<KTypelist<T, Tail>, T>
  {
    enum { value = 0 };
  };
  template <class Head, class Tail, class T>
  struct IndexOf<KTypelist<Head, Tail>, T>
  {
  private:
    enum { temp = IndexOf<Tail, T>::value };
  public:
    enum { value = (temp == -1 ? -1 : 1 + temp) };
  };

  // 3.8 Appending to Typelists

  template <class TList, class T> struct Append;
  template <> struct Append<KNullType, KNullType>
  {
    typedef KNullType Result;
  };
  template <class T> struct Append<KNullType, T>
  {
    typedef KTYPELIST_1(T) Result;
  };
  template <class Head, class Tail>
  struct Append<KNullType, KTypelist<Head, Tail> >
  {
    typedef KTypelist<Head, Tail> Result;
  };
  template <class Head, class Tail, class T>
  struct Append<KTypelist<Head, Tail>, T>
  {
    typedef KTypelist<Head, typename Append<Tail, T>::Result>
    Result;
  };

  // 3.9 Erasing a Type from a Typelist

  template <class TList, class T> struct Erase;
  template <class T>
  struct Erase<KNullType, T>
  {
    typedef KNullType Result;
  };
  template <class T, class Tail>
  struct Erase<KTypelist<T, Tail>, T>
  {
    typedef Tail Result;
  };
  template <class Head, class Tail, class T>
  struct Erase<KTypelist<Head, Tail>, T>
  {
    typedef KTypelist<Head, typename Erase<Tail, T>::Result>
    Result;
  };

  // (My own code) Removing one typelist from another

  template <class TList, class TListRemove> struct RemoveTypelist;
  template <class TList>
  struct RemoveTypelist<TList,KNullType>
  {
    typedef TList Result;
  };
  template <class TList,class Head,class Tail>
  struct RemoveTypelist<TList,KTypelist<Head,Tail> >
  {
    typedef typename RemoveTypelist<typename Erase<TList,Head>::Result,Tail>::Result Result;
  };

  template <class TList, class T> struct EraseAll;
  template <class T>
  struct EraseAll<KNullType, T>
  {
    typedef KNullType Result;
  };
  template <class T, class Tail>
  struct EraseAll<KTypelist<T, Tail>, T>
  {
    // Go all the way down the list removing the type
    typedef typename EraseAll<Tail, T>::Result Result;
  };
  template <class Head, class Tail, class T>
  struct EraseAll<KTypelist<Head, Tail>, T>
  {
    // Go all the way down the list removing the type
    typedef KTypelist<Head, typename EraseAll<Tail, T>::Result>
    Result;
  };

  // 3.10 Erasing Duplicates

  template <class TList> struct NoDuplicates;
  template <> struct NoDuplicates<KNullType>
  {
    typedef KNullType Result;
  };
  template <class Head, class Tail>
  struct NoDuplicates< KTypelist<Head, Tail> >
  {
  private:
    typedef typename NoDuplicates<Tail>::Result L1;
    typedef typename Erase<L1, Head>::Result L2;
  public:
    typedef KTypelist<Head, L2> Result;
  };

  // 3.11 Replacing an Element in a Typelist
  template <class TList, class T, class U> struct Replace;
  template <class T, class U>
  struct Replace<KNullType, T, U>
  {
    typedef KNullType Result;
  };
  template <class T, class Tail, class U>
  struct Replace<KTypelist<T, Tail>, T, U>
  {
    typedef KTypelist<U, Tail> Result;
  };
  template <class Head, class Tail, class T, class U>
  struct Replace<KTypelist<Head, Tail>, T, U>
  {
    typedef KTypelist<Head, typename Replace<Tail, T, U>::Result>
    Result;
  };

  // 3.13 Class Generation with Typelists
  template <class TList, template <class> class Unit>
  class KGenScatterHierarchy;
  // GenScatterHierarchy specialization: Typelist to Unit
  template <class T1, class T2, template <class> class Unit>
  class KGenScatterHierarchy<KTypelist<T1, T2>, Unit>
    : public KGenScatterHierarchy<T1, Unit>,
      public KGenScatterHierarchy<T2, Unit>
  {
  public:
    typedef KTypelist<T1, T2> TList;
    typedef KGenScatterHierarchy<T1, Unit> LeftBase;
    typedef KGenScatterHierarchy<T2, Unit> RightBase;
    template <typename T> struct Rebind
    {
      typedef Unit<T> Result;
    };
  };
  // Pass an atomic type (non-typelist) to Unit
  template <class AtomicType, template <class> class Unit>
  class KGenScatterHierarchy : public Unit<AtomicType>
  {
    typedef Unit<AtomicType> LeftBase;
    template <typename T> struct Rebind
    {
      typedef Unit<T> Result;
    };
  };
  // Do nothing for NullType
  template <template <class> class Unit>
  class KGenScatterHierarchy<KNullType, Unit>
  {
    template <typename T> struct Rebind
    {
      typedef Unit<T> Result;
    };
  };

  template <class T, class H>
  typename H::template Rebind<T>::Result& KField(H& obj)
  {
    return obj;
  }

  template <class T, class H>
  const typename H::template Rebind<T>::Result& KField(const H& obj)
  {
    return obj;
  }

  // GenScatterHierarchy with an additional template parameter
  template <class TList, class Parameter, template <class,class> class Unit>
  class KGenScatterHierarchyWithParameter;
  // GenScatterHierarchy specialization: Typelist to Unit
  template <class T1, class T2, class Parameter, template <class,class> class Unit>
  class KGenScatterHierarchyWithParameter<KTypelist<T1, T2>, Parameter, Unit>
    : public KGenScatterHierarchyWithParameter<T1, Parameter, Unit>,
      public KGenScatterHierarchyWithParameter<T2, Parameter, Unit>
  {
  public:
    typedef KTypelist<T1, T2> TList;
    typedef KGenScatterHierarchyWithParameter<T1, Parameter, Unit> LeftBase;
    typedef KGenScatterHierarchyWithParameter<T2, Parameter, Unit> RightBase;
    template <typename T> struct Rebind
    {
      typedef Unit<T,Parameter> Result;
    };
  };
  // Pass an atomic type (non-typelist) to Unit
  template <class AtomicType, class Parameter, template <class,class> class Unit>
  class KGenScatterHierarchyWithParameter : public Unit<AtomicType,Parameter>
  {
    typedef Unit<AtomicType,Parameter> LeftBase;
    template <typename T> struct Rebind
    {
      typedef Unit<T,Parameter> Result;
    };
  };
  // Do nothing for NullType
  template <class Parameter,template <class,class> class Unit>
  class KGenScatterHierarchyWithParameter<KNullType, Parameter, Unit>
  {
    template <typename T> struct Rebind
    {
      typedef Unit<T,Parameter> Result;
    };
  };

  // GenLinearHierarchy
  template <class TList, template <class AtomicType, class Base> class Unit, class Root = KEmptyType>
  class KGenLinearHierarchy;

  template <class T1,class T2,template <class, class> class Unit,class Root>
  class KGenLinearHierarchy<KTypelist<T1, T2>, Unit, Root>
    : public Unit< T1, KGenLinearHierarchy<T2, Unit, Root> >
  {
  };

  template <class T,template <class, class> class Unit,class Root>
  class KGenLinearHierarchy<KTypelist<T, KNullType>, Unit, Root>
    : public Unit<T, Root>
  {
  };

  // GenLinearHierarchy with an additional template parameter
  template <class TList,class Parameter,template <class AtomicType, class, class Base> class Unit,class Root = KEmptyType>
  class KGenLinearHierarchyWithParameter;

  template <class T1,class T2,class Parameter,template <class, class, class> class Unit,class Root>
  class KGenLinearHierarchyWithParameter<KTypelist<T1, T2>, Parameter, Unit, Root>
    : public Unit< T1, Parameter, KGenLinearHierarchyWithParameter<T2, Parameter, Unit, Root> >
  {
  };

  template <class T,class Parameter,template <class, class, class> class Unit,class Root>
  class KGenLinearHierarchyWithParameter<KTypelist<T, KNullType>, Parameter, Unit, Root>
    : public Unit<T, Parameter, Root>
  {
  };
}

#endif /* KTYPELIST_DEF */
