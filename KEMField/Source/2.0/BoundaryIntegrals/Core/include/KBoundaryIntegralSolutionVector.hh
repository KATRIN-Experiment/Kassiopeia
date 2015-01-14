#ifndef KBOUNDARYINTEGRALSOLUTIONVECTOR_DEF
#define KBOUNDARYINTEGRALSOLUTIONVECTOR_DEF

#include "KVector.hh"
#include "KSurfaceContainer.hh"
#include "KSurface.hh"

#include "KSurfaceVisitors.hh"

namespace KEMField
{
  template <typename Integrator, bool enableCaching=false>
  class KBoundaryIntegralSolutionVector :
    public KVector<typename Integrator::ValueType>
  {
  public:
    typedef typename Integrator::Basis::ValueType ValueType;

    KBoundaryIntegralSolutionVector(KSurfaceContainer& c,Integrator& integrator);
    virtual ~KBoundaryIntegralSolutionVector() {}

    unsigned int Dimension() const { return fDimension; }

    virtual const ValueType& operator()(unsigned int i) const;
    virtual ValueType& operator[](unsigned int i);

  private:
    KSurfaceContainer& fContainer;
    const unsigned int fDimension;
    Integrator& fIntegrator;

    mutable std::vector<bool> fValueIsCached;
    mutable std::vector<ValueType*> fCachedValue;
  };

  template <typename Integrator, bool enableCaching>
  KBoundaryIntegralSolutionVector<Integrator,enableCaching>::KBoundaryIntegralSolutionVector(KSurfaceContainer& c,Integrator& integrator)
    : KVector<ValueType>(),
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
  const typename Integrator::Basis::ValueType& KBoundaryIntegralSolutionVector<Integrator,enableCaching>::operator()(unsigned int i) const
  {
    if (enableCaching)
    {
      if (fValueIsCached[i])
	return *fCachedValue[i];
    }

    ValueType* value = &(fIntegrator.BasisValue(fContainer.at(i/Integrator::Basis::Dimension),i%Integrator::Basis::Dimension));

    if (enableCaching)
    {
      fValueIsCached[i] = true;
      fCachedValue[i] = value;
    }

    return *value;
  }

  template <typename Integrator, bool enableCaching>
  typename Integrator::Basis::ValueType& KBoundaryIntegralSolutionVector<Integrator,enableCaching>::operator[](unsigned int i)
  {
    if (enableCaching)
    {
      if (fValueIsCached[i])
	return *fCachedValue[i];
    }

    ValueType* value = &(fIntegrator.BasisValue(fContainer.at(i/Integrator::Basis::Dimension),i%Integrator::Basis::Dimension));

    if (enableCaching)
    {
      fValueIsCached[i] = true;
      fCachedValue[i] = value;
    }

    return *value;
  }
}

#endif /* KBOUNDARYINTEGRALSOLUTIONVECTOR_DEF */
