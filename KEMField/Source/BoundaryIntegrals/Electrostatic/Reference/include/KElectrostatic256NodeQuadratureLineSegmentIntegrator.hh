#ifndef KELECTROSTATIC256NODEQUADRATURELINESEGMENTINTEGRATOR_DEF
#define KELECTROSTATIC256NODEQUADRATURELINESEGMENTINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSurface.hh"

#include <iostream>

namespace KEMField
{
class KElectrostatic256NodeQuadratureLineSegmentIntegrator : public KElectrostaticElementIntegrator<KLineSegment>
{
  public:
    typedef KLineSegment Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostatic256NodeQuadratureLineSegmentIntegrator() {}
    ~KElectrostatic256NodeQuadratureLineSegmentIntegrator() override {}

    double Potential(const KLineSegment* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KLineSegment* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KLineSegment* source,
                                                              const KPosition& P) const override;

    double Potential(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KSymmetryGroup<KLineSegment>* source,
                                                              const KPosition& P) const override;
};


static const double g256NodeQuadx32[16] = {0.048307665687738316,
                                           0.144471961582796493,
                                           0.239287362252137075,
                                           0.331868602282127650,
                                           0.421351276130635345,
                                           0.506899908932229390,
                                           0.587715757240762329,
                                           0.663044266930215201,
                                           0.732182118740289680,
                                           0.794483795967942407,
                                           0.849367613732569970,
                                           0.896321155766052124,
                                           0.934906075937739689,
                                           0.964762255587506431,
                                           0.985611511545268335,
                                           0.997263861849481564};
static const double g256NodeQuadw32[16] = {0.09654008851472780056,
                                           0.09563872007927485942,
                                           0.09384439908080456564,
                                           0.09117387869576388471,
                                           0.08765209300440381114,
                                           0.08331192422694675522,
                                           0.07819389578707030647,
                                           0.07234579410884850625,
                                           0.06582222277636184684,
                                           0.05868409347853554714,
                                           0.05099805926237617619,
                                           0.04283589802222680057,
                                           0.03427386291302143313,
                                           0.02539206530926205956,
                                           0.01627439473090567065,
                                           0.00701861000947009660};
}  // namespace KEMField

#endif /* KELECTROSTATIC256NODEQUADRATURELINESEGMENTINTEGRATOR_DEF */
