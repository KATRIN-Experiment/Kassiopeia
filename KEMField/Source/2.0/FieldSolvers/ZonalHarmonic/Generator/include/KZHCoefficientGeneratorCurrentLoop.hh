#ifndef KZHCOEFFICIENTGENERATORCURRENTLOOP_H
#define KZHCOEFFICIENTGENERATORCURRENTLOOP_H

#include "KZHCoefficientGeneratorElement.hh"
#include "KCurrentLoop.hh"

namespace KEMField
{
  template <>
  class KZHCoefficientGenerator<KCurrentLoop> : public KZHCoefficientGeneratorElement
  {
  public:
    typedef KCurrentLoop ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    virtual ~KZHCoefficientGenerator() {}

    void SetElement(const KCurrentLoop* c) { fCurrentLoop = c; }

    const KEMCoordinateSystem& GetCoordinateSystem() const
    { return fCurrentLoop->GetCoordinateSystem(); }

    double Prefactor() const { return fCurrentLoop->GetCurrent(); }

    void ComputeCentralCoefficients(double,
    				    double,
    				    std::vector<double>&) const;
    void ComputeRemoteCoefficients(double,
    				   double,
    				   std::vector<double>&) const;

    double ComputeRho(double,bool) const;

    void GetExtrema(double&,double&) const;

  protected:
    const KCurrentLoop* fCurrentLoop;
  };

}

#endif /* KZHCOEFFICIENTGENERATORCURRENTLOOP */
