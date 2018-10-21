#ifndef KZHCOEFFICIENTGENERATORRING_H
#define KZHCOEFFICIENTGENERATORRING_H

#include "KZHCoefficientGeneratorElement.hh"
#include "KSurface.hh"

namespace KEMField
{
  template <>
  class KZHCoefficientGenerator<KRing> : public KZHCoefficientGeneratorElement
  {
  public:
    typedef KRing ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement(), fNullDistance(0.) {}
    virtual ~KZHCoefficientGenerator() {}

    void SetElement(const KRing* r) { fRing = r; }

    const KEMCoordinateSystem& GetCoordinateSystem() const
    { return gGlobalCoordinateSystem; }

    double Prefactor() const;

    void ComputeCentralCoefficients(double,
    				    double,
    				    std::vector<double>&) const;
    void ComputeRemoteCoefficients(double,
    				   double,
    				   std::vector<double>&) const;

    double ComputeRho(double,bool) const;

    void GetExtrema(double&,double&) const;

    void SetNullDistance(double d) { fNullDistance = d; }

  protected:
    const KRing* fRing;

    double fNullDistance;
  };
}

#endif /* KZHCOEFFICIENTGENERATORRING */
