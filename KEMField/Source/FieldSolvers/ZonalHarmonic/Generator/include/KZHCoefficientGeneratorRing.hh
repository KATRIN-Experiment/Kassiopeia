#ifndef KZHCOEFFICIENTGENERATORRING_H
#define KZHCOEFFICIENTGENERATORRING_H

#include "KSurface.hh"
#include "KZHCoefficientGeneratorElement.hh"

namespace KEMField
{
template<> class KZHCoefficientGenerator<KRing> : public KZHCoefficientGeneratorElement
{
  public:
    using ElementType = KRing;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement(), fNullDistance(0.) {}
    ~KZHCoefficientGenerator() override = default;

    void SetElement(const KRing* r)
    {
        fRing = r;
    }

    const KEMCoordinateSystem& GetCoordinateSystem() const override
    {
        return gGlobalCoordinateSystem;
    }

    double Prefactor() const override;

    void ComputeCentralCoefficients(double, double, std::vector<double>&) const override;
    void ComputeRemoteCoefficients(double, double, std::vector<double>&) const override;

    double ComputeRho(double, bool) const override;

    void GetExtrema(double&, double&) const override;

    void SetNullDistance(double d)
    {
        fNullDistance = d;
    }

  protected:
    const KRing* fRing;

    double fNullDistance;
};
}  // namespace KEMField

#endif /* KZHCOEFFICIENTGENERATORRING */
