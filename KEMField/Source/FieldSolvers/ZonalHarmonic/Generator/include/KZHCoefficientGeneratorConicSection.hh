#ifndef KZHCOEFFICIENTGENERATORCONICSECTION_H
#define KZHCOEFFICIENTGENERATORCONICSECTION_H

#include "KZHCoefficientGeneratorElement.hh"
#include "KSurface.hh"

#include "KZHLegendreCoefficients.hh"

namespace KEMField
{
  template <>
  class KZHCoefficientGenerator<KConicSection> :
    public KZHCoefficientGeneratorElement
  {
  public:
    typedef KConicSection ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    virtual ~KZHCoefficientGenerator() {}

    void SetElement(const KConicSection* c) { fConicSection = c; }

    const KEMCoordinateSystem& GetCoordinateSystem() const
    { return gGlobalCoordinateSystem; }

    double Prefactor() const;

    void ComputeCentralCoefficients(double,
    				    double,
    				    std::vector<double>&) const;
    void ComputeRemoteCoefficients(double,
    				   double,
    				   std::vector<double>&) const;

    void ComputeCoefficients(double z0,
			     double rho_const,
			     std::vector<double>& coeff,
			     bool isCen) const;

    double ComputeRho(double,bool) const;

    void GetExtrema(double&,double&) const;

  protected:
    const KConicSection* fConicSection;

  };
}

#endif /* KZHCOEFFICIENTGENERATORCONICSECTION */
