#ifndef KZHCOEFFICIENTGENERATORCOIL_H
#define KZHCOEFFICIENTGENERATORCOIL_H

#include "KZHCoefficientGeneratorElement.hh"
#include "KCoil.hh"

namespace KEMField
{
  template <>
  class KZHCoefficientGenerator<KCoil> : public KZHCoefficientGeneratorElement
  {
  public:
    typedef KCoil ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    virtual ~KZHCoefficientGenerator() {}

    void SetElement(const KCoil* c) { fCoil = c; }

    const KEMCoordinateSystem& GetCoordinateSystem() const
    { return fCoil->GetCoordinateSystem(); }

    double Prefactor() const { return fCoil->GetCurrent(); }

    void ComputeCentralCoefficients(double,
    				    double,
    				    std::vector<double>&) const;
    void ComputeRemoteCoefficients(double,
    				   double,
    				   std::vector<double>&) const;

    double ComputeRho(double,bool) const;

    void GetExtrema(double&,double&) const;

  protected:
    const KCoil* fCoil;
  };
}

#endif /* KCOIL */
