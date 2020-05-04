#ifndef KZHCOEFFICIENTGENERATORCOIL_H
#define KZHCOEFFICIENTGENERATORCOIL_H

#include "KCoil.hh"
#include "KZHCoefficientGeneratorElement.hh"

namespace KEMField
{
template<> class KZHCoefficientGenerator<KCoil> : public KZHCoefficientGeneratorElement
{
  public:
    typedef KCoil ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    ~KZHCoefficientGenerator() override {}

    void SetElement(const KCoil* c)
    {
        fCoil = c;
    }

    const KEMCoordinateSystem& GetCoordinateSystem() const override
    {
        return fCoil->GetCoordinateSystem();
    }

    double Prefactor() const override
    {
        return fCoil->GetCurrent();
    }

    void ComputeCentralCoefficients(double, double, std::vector<double>&) const override;
    void ComputeRemoteCoefficients(double, double, std::vector<double>&) const override;

    double ComputeRho(double, bool) const override;

    void GetExtrema(double&, double&) const override;

  protected:
    const KCoil* fCoil;
};
}  // namespace KEMField

#endif /* KCOIL */
