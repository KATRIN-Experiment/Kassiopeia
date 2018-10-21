#ifndef KTYPEMANIPULATION_DEF
#define KTYPEMANIPULATION_DEF

#include <typeinfo>

// this fixes compile-time errors on GCC when using the C++11 standard
#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define __typeof__(x) decltype(x)
#endif

namespace KEMField
{
  template<bool Cond, class T = void>
  struct EnableIf {};

  template<class T>
  struct EnableIf<true, T>
  {
    typedef T type;
  };

  template<typename T>
  struct Type2Type
  {
    typedef T OriginalType;
  };

  template <int i>
  struct Int2Type
  {
    enum {type = i};
  };

  template<typename D, typename B>
  class IsDerivedFrom
  {
    class No {};
    class Yes { No no[2]; };

    static Yes Test(B*);
    static No Test(...);

  public:
    enum { Is = sizeof(Test(static_cast<D*>(0))) == sizeof(Yes) };
  };

  template<typename T>
  class IsNamed
  {
    class No {};
    class Yes { No no[2]; };

    template <typename C>
    static Yes Test(__typeof__(&C::Name));
    template <typename C>
    static No Test(...);

  public:
    enum { Is = sizeof(Test<T>(0)) == sizeof(Yes) };
  };

}

#endif /* KTYPEMANIPULATION_DEF */
