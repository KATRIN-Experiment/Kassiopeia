#ifndef KZONALHARMONICSOLENOID_H
#define KZONALHARMONICSOLENOID_H

#include "KSolenoid.hh"
#include "KZHCoefficientGeneratorElement.hh"

namespace KEMField
{
template<> class KZHCoefficientGenerator<KSolenoid> : public KZHCoefficientGeneratorElement
{
  public:
    typedef KSolenoid ElementType;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    ~KZHCoefficientGenerator() override {}

    void SetElement(const KSolenoid* s)
    {
        fSolenoid = s;
    }

    const KEMCoordinateSystem& GetCoordinateSystem() const override
    {
        return fSolenoid->GetCoordinateSystem();
    }

    double Prefactor() const override
    {
        return fSolenoid->GetCurrent();
    }

    void ComputeCentralCoefficients(double, double, std::vector<double>&) const override;
    void ComputeRemoteCoefficients(double, double, std::vector<double>&) const override;

    double ComputeRho(double, bool) const override;

    void GetExtrema(double&, double&) const override;

  protected:
    const KSolenoid* fSolenoid;
};

}  // namespace KEMField

#endif /* KSOLENOID */
