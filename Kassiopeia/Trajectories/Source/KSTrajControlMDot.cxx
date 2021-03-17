#include "KSTrajControlMDot.h"

#include "KSTrajectoriesMessage.h"

#include <cmath>

using KGeoBag::KThreeMatrix;
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSTrajControlMDot::KSTrajControlMDot() : fFraction(1. / 16.) {}
KSTrajControlMDot::KSTrajControlMDot(const KSTrajControlMDot& aCopy) : KSComponent(aCopy), fFraction(aCopy.fFraction) {}
KSTrajControlMDot* KSTrajControlMDot::Clone() const
{
    return new KSTrajControlMDot(*this);
}
KSTrajControlMDot::~KSTrajControlMDot() = default;

void KSTrajControlMDot::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    KThreeVector e3 = aParticle.GetMagneticField() / aParticle.GetMagneticField().Magnitude();  // = b
    KThreeVector E1(e3.Z() - e3.Y(), e3.X() - e3.Z(), e3.Y() - e3.X());
    KThreeVector e1 = E1 / E1.Magnitude();
    KThreeVector e2 = e3.Cross(e1);

    KThreeVector a = -1 * e1 * sin(aParticle.GetSpinAngle()) + e2 * cos(aParticle.GetSpinAngle());
    KThreeVector c = e1 * cos(aParticle.GetSpinAngle()) + e2 * sin(aParticle.GetSpinAngle());

    //std::cout << "e1: " << e1 << "\t\te2: " << e2 << "\t\te3: " << e3 << "\n\ta: " << a << "\t\tc: " << c << "\n";

    KThreeVector GradBMagnitude =
        aParticle.GetMagneticGradient() * aParticle.GetMagneticField() / aParticle.GetMagneticField().Magnitude();
    KThreeMatrix GradBDirection = aParticle.GetMagneticGradient() / aParticle.GetMagneticField().Magnitude() -
                                  KThreeMatrix::OuterProduct(aParticle.GetMagneticField(), GradBMagnitude) /
                                      aParticle.GetMagneticField().Magnitude() /
                                      aParticle.GetMagneticField().Magnitude();
    KThreeMatrix GradE1(GradBDirection[6] - GradBDirection[3],
                        GradBDirection[7] - GradBDirection[4],
                        GradBDirection[8] - GradBDirection[5],
                        GradBDirection[0] - GradBDirection[6],
                        GradBDirection[1] - GradBDirection[7],
                        GradBDirection[2] - GradBDirection[8],
                        GradBDirection[3] - GradBDirection[0],
                        GradBDirection[4] - GradBDirection[1],
                        GradBDirection[5] - GradBDirection[2]);
    KThreeVector GradE1Magnitude = GradE1 * E1 / E1.Magnitude();
    KThreeMatrix Grade1 =
        GradE1 / E1.Magnitude() - KThreeMatrix::OuterProduct(E1, GradE1Magnitude) / E1.Magnitude() / E1.Magnitude();
    KThreeVector A = Grade1 * e2;

    //std::cout << "GradBM: " << GradBMagnitude << "\t\tGradBD: " << GradBDirection << "\t\tGradE1: " << GradE1 << "\n\tGradE1M: " << GradE1Magnitude << "\t\tGrade1: " << Grade1 << "\n";

    double tMDot = std::sqrt(1 - aParticle.GetAlignedSpin() * aParticle.GetAlignedSpin()) *
                   aParticle.GetVelocity().Dot(GradBDirection * c);
    double tPhiDot =
        aParticle.GetGyromagneticRatio() * aParticle.GetMagneticField().Magnitude() - aParticle.GetVelocity().Dot(A) -
        aParticle.GetAlignedSpin() / std::sqrt(1 - aParticle.GetAlignedSpin() * aParticle.GetAlignedSpin()) *
            aParticle.GetVelocity().Dot(GradBDirection * a);

    double MDotControl = std::fabs(fFraction / tMDot);
    double PhiDotControl = std::fabs(fFraction / tPhiDot);

    aValue = (MDotControl < PhiDotControl) ? MDotControl : PhiDotControl;

    //std::cout << "m: " << aParticle.GetAlignedSpin() << "\t\tphi: " << aParticle.GetSpinAngle() << "\t\tControl: " << aValue << "\n";

    return;
}
void KSTrajControlMDot::Check(const KSTrajAdiabaticSpinParticle&, const KSTrajAdiabaticSpinParticle&,
                              const KSTrajAdiabaticSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}


}  // namespace Kassiopeia
