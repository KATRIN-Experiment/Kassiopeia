#ifndef KELECTROSTATICRWGRECTANGLEINTEGRATOR_DEF
#define KELECTROSTATICRWGRECTANGLEINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSolidAngle.hh"
#include "KSurface.hh"
#include "KSymmetryGroup.hh"

#include <cmath>

namespace KEMField
{
class KElectrostaticRWGRectangleIntegrator : public KElectrostaticElementIntegrator<KRectangle>
{
  public:
    typedef KRectangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticRWGRectangleIntegrator() {}
    ~KElectrostaticRWGRectangleIntegrator() override {}

    double Potential(const KRectangle* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KRectangle* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KRectangle* source,
                                                              const KPosition& P) const override;

    double Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KSymmetryGroup<KRectangle>* source,
                                                              const KPosition& P) const override;

  private:
    double LogArgTaylor(const double sMin, const double dist) const;

    double IqLPotential(const double* data, const double* P, const unsigned short countCross,
                        const unsigned short lineIndex, const double dist) const;

    KThreeVector IqLField(const double* data, const double* P, const unsigned short countCross,
                          const unsigned short lineIndex, const double dist) const;

    std::pair<KThreeVector, double> IqLFieldAndPotential(const double* data, const double* P,
                                                         const unsigned short countCross,
                                                         const unsigned short lineIndex, const double dist) const;

    KSolidAngle solidAngle;

    const double fMinDistanceToSideLine = 1.E-14;
    const double fDistanceCorrectionN3 = 1.E-7; /* step in N3 direction if field point is on edge */
    const double fLogArgQuotient =
        1.E-6; /* limit of quotient dist/sM for Taylor expansion (if field point is on line) */
    const double fToleranceLambda = 1.E-15; /* tolerance for determining if field point is on vertex */
};

}  // namespace KEMField

#endif /* KELECTROSTATICRWGRECTANGLEINTEGRATOR_DEF */
