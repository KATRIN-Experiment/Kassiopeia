#include "KSGenValueAngleCosine.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSGenValueAngleCosine::KSGenValueAngleCosine() : fAngleMin(0.), fAngleMax(0.), fMode(EDistributionMode::Classic), fDirection(EDirection::Forward) {}
KSGenValueAngleCosine::KSGenValueAngleCosine(const KSGenValueAngleCosine& aCopy) :
    KSComponent(aCopy),
    fAngleMin(aCopy.fAngleMin),
    fAngleMax(aCopy.fAngleMax),
    fMode(EDistributionMode::Classic)
{}
KSGenValueAngleCosine* KSGenValueAngleCosine::Clone() const
{
    return new KSGenValueAngleCosine(*this);
}
KSGenValueAngleCosine::~KSGenValueAngleCosine() = default;

void KSGenValueAngleCosine::DiceValue(std::vector<double>& aDicedValues)
{
    genmsg_assert(fAngleMax, >= 0);
    genmsg_assert(fAngleMin, >= 0);
    genmsg_assert(fAngleMax, <= 90);
    genmsg_assert(fAngleMin, <= 90);

    double tAngle = 0.;
    if (fMode == EDistributionMode::MolecularFlow) {
        double tsinThetaSquaredMin = pow(sin((katrin::KConst::Pi() / 180.) * fAngleMin), 2);
        double tsinThetaSquaredMax = pow(sin((katrin::KConst::Pi() / 180.) * fAngleMax), 2);

        //Random generation follows Eq. 12 from J. Greenwood, Vacuum, 67 (2002), pp. 217-222
        double tsinThetaSquared = KRandom::GetInstance().Uniform(tsinThetaSquaredMin, tsinThetaSquaredMax);
        tAngle = asin(sqrt(tsinThetaSquared));
    }
    else if (fMode == EDistributionMode::Classic) {
        double tSinThetaMin = sin( (katrin::KConst::Pi() / 180.) * fAngleMin );
        double tSinThetaMax = sin( (katrin::KConst::Pi() / 180.) * fAngleMax );

        double tSinTheta = KRandom::GetInstance().Uniform( tSinThetaMin, tSinThetaMax );
        tAngle = acos( sqrt( 1. - tSinTheta*tSinTheta ) );
    }
    
    if (fDirection == EDirection::Backward) {
        tAngle = katrin::KConst::Pi() - tAngle;
    }

    aDicedValues.push_back((180.0 / katrin::KConst::Pi()) * tAngle);

    return;
}

}  // namespace Kassiopeia
