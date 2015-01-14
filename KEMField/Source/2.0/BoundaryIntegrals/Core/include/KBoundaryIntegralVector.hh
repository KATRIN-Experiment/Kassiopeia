#ifndef KBOUNDARYINTEGRALVECTOR_DEF
#define KBOUNDARYINTEGRALVECTOR_DEF

#include "KVector.hh"
#include "KSurfaceContainer.hh"
#include "KSurface.hh"

namespace KEMField
{
  template <typename Integrator, bool enableCaching=false>
  class KBoundaryIntegralVector : public KVector<typename Integrator::ValueType>
  {
  public:
    typedef typename Integrator::Basis::ValueType ValueType;

    KBoundaryIntegralVector(KSurfaceContainer& c,Integrator& integrator);

    virtual ~KBoundaryIntegralVector() {}

    unsigned int Dimension() const { return fDimension; }

    virtual const ValueType& operator()(unsigned int i) const;

  private:
    // We disable this method by making it private.
    virtual ValueType& operator[](unsigned int )
    { static ValueType dummy; return dummy; }

  private:
    KSurfaceContainer& fContainer;
    const unsigned int fDimension;
    Integrator& fIntegrator;

    mutable std::vector<bool> fValueIsCached;
    mutable std::vector<ValueType> fCachedValue;
  };

  template <typename Integrator, bool enableCaching>
  KBoundaryIntegralVector<Integrator,enableCaching>::
  KBoundaryIntegralVector(KSurfaceContainer& c,Integrator& integrator) :
      KVector<ValueType>(),
      fContainer(c),
      fDimension(c.size()*Integrator::Basis::Dimension),
      fIntegrator(integrator)
  {
    if (enableCaching)
    {
      fValueIsCached.resize(c.size()*Integrator::Basis::Dimension,false);
      fCachedValue.resize(c.size()*Integrator::Basis::Dimension);
    }
  }

  template <typename Integrator, bool enableCaching>
  const typename Integrator::Basis::ValueType& KBoundaryIntegralVector<Integrator,enableCaching>::operator()(unsigned int i) const
    {
      if (enableCaching)
      {
	if (fValueIsCached[i])
	  return fCachedValue[i];
      }

      static ValueType value;

      value = fIntegrator.BoundaryValue(fContainer.at(i/Integrator::Basis::Dimension),i%Integrator::Basis::Dimension);

      if (enableCaching)
      {
	fValueIsCached[i] = true;
	fCachedValue[i] = value;
      }

      return value;
    }

}

#endif /* KBOUNDARYINTEGRALVECTOR_DEF */
