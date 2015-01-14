#ifndef KBOUNDARYINTEGRALMATRIX_DEF
#define KBOUNDARYINTEGRALMATRIX_DEF

#include "KDataDisplay.hh"

#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KSquareMatrix.hh"

#define KEM_USE_CACHING true

namespace KEMField
{
  template <class Integrator, bool enableCaching=false>
  class KBoundaryIntegralMatrix :
    public KSquareMatrix<typename Integrator::Basis::ValueType>
  {
  public:
    typedef typename Integrator::Basis::ValueType ValueType;

    KBoundaryIntegralMatrix(KSurfaceContainer& c,Integrator& integrator);

    ~KBoundaryIntegralMatrix() {}

    unsigned int Dimension() const { return fDimension; }

    virtual const ValueType& operator()(unsigned int i,unsigned int j) const
    {
      if (enableCaching)
      {
	if (fValueIsCached[i*fDimension+j])
	  return fCachedValue[i*fDimension+j];
      }

      static ValueType value;

      value = fIntegrator.BoundaryIntegral(fContainer.at(j/Integrator::Basis::Dimension),j%Integrator::Basis::Dimension,fContainer.at(i/Integrator::Basis::Dimension),i%Integrator::Basis::Dimension);

      if (enableCaching)
      {
	fValueIsCached[i*fDimension+j] = true;
	fCachedValue[i*fDimension+j] = value;
      }

      return value;
    }

  protected:
    KSurfaceContainer& fContainer;
    const unsigned int fDimension;
    Integrator& fIntegrator;

    mutable std::vector<bool> fValueIsCached;
    mutable std::vector<ValueType> fCachedValue;
  };

  template <class Integrator, bool enableCaching>
  KBoundaryIntegralMatrix<Integrator,enableCaching>::
  KBoundaryIntegralMatrix(KSurfaceContainer& c,Integrator& integrator) :
    KSquareMatrix<ValueType>(),
    fContainer(c),
    fDimension(c.size()*Integrator::Basis::Dimension),
    fIntegrator(integrator)
  {
    if (enableCaching)
    {
      unsigned int basisDim2 = Integrator::Basis::Dimension;
      basisDim2*=basisDim2;

      fValueIsCached.resize(c.size()*c.size()*basisDim2,false);
      fCachedValue.resize(c.size()*c.size()*basisDim2);
    }
  }

}

#endif /* KVBOUNDARYINTEGRALMATRIX_DEF */
