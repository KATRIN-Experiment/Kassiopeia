#ifndef KTRANSITIVESTREAMER_DEF
#define KTRANSITIVESTREAMER_DEF

#include <deque>

#include "KFundamentalTypes.hh"

namespace KEMField
{

/**
 * @class KTransitiveStreamer
 *
 * @brief A class for streaming from one object to another.
 *
 * @author T.J. Corona
 */

  template <typename Type, class Derived>
  class KTransitiveStreamerType : public std::deque<Type>
  {
  public:
    friend inline Derived& operator>>(KTransitiveStreamerType<Type,Derived>& t, Type &x)
    {
      x = t.front();
      t.pop_front();
      return t.Self();
    }

    friend inline Derived& operator<<(KTransitiveStreamerType<Type,Derived>& t, const Type &x)
    {
      t.push_back(x);
      return t.Self();
    }

    virtual ~KTransitiveStreamerType() {}

  protected:
    virtual Derived& Self() = 0;
  };

  template <class Derived>
  class KTransitiveStreamer :
    public KGenScatterHierarchyWithParameter<KEMField::FundamentalTypes,
					     Derived,
					     KTransitiveStreamerType>
  {
  public:
    KTransitiveStreamer() {}
    ~KTransitiveStreamer() {}

  protected:
    Derived& Self() { return *(static_cast<Derived*>(this)); }


    template <class Object>
    friend inline Derived& operator>>(const Object& object,Derived& streamer) { return streamer << object; }

    template <class Object>
    friend inline Derived& operator<<(Object& object,Derived& streamer) { return streamer >> object; }
  };
}

#endif /* KTransitiveStreamer_DEF */
