#ifndef KZHCOEFFICIENTGENERATORCONICSECTION_H
#define KZHCOEFFICIENTGENERATORCONICSECTION_H

#include "KSurface.hh"
#include "KZHCoefficientGeneratorElement.hh"
#include "KZHLegendreCoefficients.hh"

namespace KEMField
{
template<> class KZHCoefficientGenerator<KConicSection> : public KZHCoefficientGeneratorElement
{
  public:
    using ElementType = KConicSection;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    ~KZHCoefficientGenerator() override = default;

    void SetElement(const KConicSection* c)
    {
        fConicSection = c;
    }

    const KEMCoordinateSystem& GetCoordinateSystem() const override
    {
        return gGlobalCoordinateSystem;
    }

    double Prefactor() const override;

    void ComputeCentralCoefficients(double, double, std::vector<double>&) const override;
    void ComputeRemoteCoefficients(double, double, std::vector<double>&) const override;

    void ComputeCoefficients(double z0, double rho_const, std::vector<double>& coeff, bool isCen) const;

    double ComputeRho(double, bool) const override;

    void GetExtrema(double&, double&) const override;

  protected:
    const KConicSection* fConicSection;
};
}  // namespace KEMField

#endif /* KZHCOEFFICIENTGENERATORCONICSECTION */
