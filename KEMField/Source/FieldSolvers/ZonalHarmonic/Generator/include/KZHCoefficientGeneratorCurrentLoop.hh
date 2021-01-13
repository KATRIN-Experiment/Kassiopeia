#ifndef KZHCOEFFICIENTGENERATORCURRENTLOOP_H
#define KZHCOEFFICIENTGENERATORCURRENTLOOP_H

#include "KCurrentLoop.hh"
#include "KZHCoefficientGeneratorElement.hh"

namespace KEMField
{
template<> class KZHCoefficientGenerator<KCurrentLoop> : public KZHCoefficientGeneratorElement
{
  public:
    using ElementType = KCurrentLoop;

    KZHCoefficientGenerator() : KZHCoefficientGeneratorElement() {}
    ~KZHCoefficientGenerator() override = default;

    void SetElement(const KCurrentLoop* c)
    {
        fCurrentLoop = c;
    }

    const KEMCoordinateSystem& GetCoordinateSystem() const override
    {
        return fCurrentLoop->GetCoordinateSystem();
    }

    double Prefactor() const override
    {
        return fCurrentLoop->GetCurrent();
    }

    void ComputeCentralCoefficients(double, double, std::vector<double>&) const override;
    void ComputeRemoteCoefficients(double, double, std::vector<double>&) const override;

    double ComputeRho(double, bool) const override;

    void GetExtrema(double&, double&) const override;

  protected:
    const KCurrentLoop* fCurrentLoop;
};

}  // namespace KEMField

#endif /* KZHCOEFFICIENTGENERATORCURRENTLOOP */
