#ifndef KRESIDUALVECTOR_DEF
#define KRESIDUALVECTOR_DEF

#include "KSimpleVector.hh"

namespace KEMField
{
  template <typename ValueType>
  class KResidualVector : public KSimpleVector<ValueType>
  {
  public:
    KResidualVector() : KSimpleVector<ValueType>() {}
    virtual ~KResidualVector() {}

    static std::string Name() { return std::string("ResidualVector"); }

    void SetIteration(unsigned long long iteration) { fIteration = iteration; }
    unsigned long long GetIteration() const { return fIteration; }

    template <typename Stream>
    friend Stream& operator>>(Stream& s,KResidualVector& v)
    {
      s.PreStreamInAction(v);
      unsigned long long iteration;
      s >> iteration;
      v.SetIteration(iteration);
      unsigned int dimension;
      s >> dimension;
      v.resize(dimension);
      for (unsigned int i=0;i<dimension;i++)
	s >> v[i];
      s.PostStreamInAction(v);
      return s;
    }

    template <typename Stream>
    friend Stream& operator<<(Stream& s,const KResidualVector& v)
    {
      s.PreStreamOutAction(v);
      s << v.GetIteration();
      s << v.Dimension();
      for (unsigned int i=0;i<v.Dimension();i++)
	s << v(i);
      s.PostStreamOutAction(v);
      return s;
    }

  private:
    unsigned long long fIteration;
  };
}

#endif /* KRESIDUALVECTOR_DEF */
