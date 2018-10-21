#ifndef KZONALHARMONICSOLENOID_H
#define KZONALHARMONICSOLENOID_H

#include "KZHCoefficientGeneratorElement.hh"
#include "KSolenoid.hh"

namespace KEMField
{
  template <>
  class KZHCoefficientGenerator<KSolenoid> :
    public KZHCoefficientGeneratorElement
  {
  public:
    typedef KSolenoid ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    virtual ~KZHCoefficientGenerator() {}

    void SetElement(const KSolenoid* s) { fSolenoid = s; }

    const KEMCoordinateSystem& GetCoordinateSystem() const
    { return fSolenoid->GetCoordinateSystem(); }

    double Prefactor() const { return fSolenoid->GetCurrent(); }

    void ComputeCentralCoefficients(double,
    				    double,
    				    std::vector<double>&) const;
    void ComputeRemoteCoefficients(double,
    				   double,
    				   std::vector<double>&) const;

    double ComputeRho(double,bool) const;

    void GetExtrema(double&,double&) const;

  protected:
    const KSolenoid* fSolenoid;
  };

}

#endif /* KSOLENOID */
