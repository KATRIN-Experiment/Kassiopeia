#ifndef KTYPEMANIPULATION_DEF
#define KTYPEMANIPULATION_DEF

#include <typeinfo>

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
    static Yes Test(typeof(&C::Name));
    template <typename C>
    static No Test(...);

  public:
    enum { Is = sizeof(Test<T>(0)) == sizeof(Yes) };
  };

}

#endif /* KTYPEMANIPULATION_DEF */
