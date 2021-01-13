#include "KSIntSurfaceUCN.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include "KConst.h"

using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSIntSurfaceUCN::KSIntSurfaceUCN() :
    fEta(0.),
    fAlpha(0.),
    fRealOpticalPotential(0.),
    fCorrelationLength(0.),
    fTanThetaIn(0.),
    fExpThetaCoef(0.)
{}
KSIntSurfaceUCN::KSIntSurfaceUCN(const KSIntSurfaceUCN& aCopy) :
    KSComponent(aCopy),
    fEta(aCopy.fEta),
    fAlpha(aCopy.fAlpha),
    fRealOpticalPotential(aCopy.fRealOpticalPotential),
    fCorrelationLength(aCopy.fCorrelationLength),
    fTanThetaIn(aCopy.fTanThetaIn),
    fExpThetaCoef(aCopy.fExpThetaCoef)
{}
KSIntSurfaceUCN* KSIntSurfaceUCN::Clone() const
{
    return new KSIntSurfaceUCN(*this);
}
KSIntSurfaceUCN::~KSIntSurfaceUCN() = default;

void KSIntSurfaceUCN::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                         KSParticleQueue& aQueue)
{
    KThreeVector tNormal;
    if (anInitialParticle.GetCurrentSurface() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSurface()->Normal(anInitialParticle.GetPosition());
    }
    else if (anInitialParticle.GetCurrentSide() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSide()->Normal(anInitialParticle.GetPosition());
    }
    else {
        intmsg(eError) << "surface UCN interaction named <" << GetName()
                       << "> was given a particle with neither a surface nor a side set" << eom;
        return;
    }
    KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
    KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot(tNormal) * tNormal;
    double Eperp = tInitialNormalMomentum * tInitialNormalMomentum / 2 / anInitialParticle.GetMass();

    double pLoss = 2 * fEta * sqrt(Eperp / (fRealOpticalPotential * katrin::KConst::Q() - Eperp));
    if (std::isnan(pLoss)) {
        pLoss = 1.;
    }
    double tChoice = KRandom::GetInstance().Uniform(0., 1.);
    if (tChoice < pLoss) {
        ExecuteTransmission(anInitialParticle, aFinalParticle, aQueue);
    }
    else {
        ExecuteReflection(anInitialParticle, aFinalParticle, aQueue);
    }
    return;
}
void KSIntSurfaceUCN::ExecuteReflection(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                        KSParticleQueue&)
{
    KThreeVector tNormal;
    if (anInitialParticle.GetCurrentSurface() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSurface()->Normal(anInitialParticle.GetPosition());
    }
    else if (anInitialParticle.GetCurrentSide() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSide()->Normal(anInitialParticle.GetPosition());
    }
    else {
        intmsg(eError) << "surface UCN interaction named <" << GetName()
                       << "> was given a particle with neither a surface nor a side set" << eom;
        return;
    }
    KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
    KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot(tNormal) * tNormal;
    KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;
    KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross(tInitialNormalMomentum.Unit());

    KThreeVector tSpin = anInitialParticle.GetSpin();
    double tAlignedSpin = anInitialParticle.GetAlignedSpin();
    double tSpinAngle = anInitialParticle.GetSpinAngle();

    double tChoice = KRandom::GetInstance().Uniform(0., 1.);
    if (tChoice < 2 * fAlpha)  // there's a 50-50 chance of getting the old spin after measurement, hence the 2*
    {
        bool done = false;
        while (!done) {
            double tx = KRandom::GetInstance().Uniform(-1., 1.);
            double ty = KRandom::GetInstance().Uniform(-1., 1.);
            double tz = KRandom::GetInstance().Uniform(-1., 1.);
            if (tx * tx + ty * ty + tz * tz < 1.) {
                tSpin = KThreeVector(tx, ty, tz);
                tSpin = tSpin / tSpin.Magnitude();
                tAlignedSpin =
                    tSpin.Dot(anInitialParticle.GetMagneticField()) / anInitialParticle.GetMagneticField().Magnitude();
                tSpinAngle = KRandom::GetInstance().Uniform(0., 180.);
                done = true;
            }
        }
    }

    // calculate the new reflection direction

    double k = tInitialMomentum.Magnitude() / katrin::KConst::Hbar();
    double cosThetaIn = tInitialNormalMomentum.Magnitude() / tInitialMomentum.Magnitude();
    double sinThetaIn = tInitialTangentMomentum.Magnitude() / tInitialMomentum.Magnitude();
    fTanThetaIn = sinThetaIn / cosThetaIn;
    double thetaIn = acos(cosThetaIn);
    fExpThetaCoef = (fCorrelationLength * k) * (fCorrelationLength * k) * cosThetaIn * cosThetaIn / 2;
    double expPhiCoef = (fCorrelationLength * k) * (fCorrelationLength * k) * sinThetaIn * sinThetaIn / 2;

    double phiOut = KRandom::GetInstance().Gauss(0., sqrt(0.5 / expPhiCoef));

    double deltaTheta = 0.;
    double tValueMin = ValueFunction(-thetaIn);
    double tValueMax = ValueFunction(katrin::KConst::Pi() / 2 - thetaIn);
    double tValue = KRandom::GetInstance().Uniform(tValueMin, tValueMax);
    fSolver.Solve(katrin::KMathBracketingSolver::eBrent,
                  this,
                  &KSIntSurfaceUCN::ValueFunction,
                  tValue,
                  -thetaIn,
                  katrin::KConst::Pi() / 2 - thetaIn,
                  deltaTheta);
    double thetaOut = thetaIn + deltaTheta;

    KThreeVector tFinalMomentum =
        -tInitialMomentum.Magnitude() * tInitialNormalMomentum.Unit() * cos(thetaOut) +
        tInitialMomentum.Magnitude() * tInitialTangentMomentum.Unit() * sin(thetaOut) * cos(phiOut) +
        tInitialMomentum.Magnitude() * tInitialOrthogonalMomentum.Unit() * sin(thetaOut) * sin(phiOut);

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tFinalMomentum);

    // spin changes need to happen aftet SetMomentum to make Spin0 correct

    aFinalParticle.SetInitialSpin(tSpin);
    aFinalParticle.SetAlignedSpin(tAlignedSpin);
    aFinalParticle.SetSpinAngle(tSpinAngle);

    return;
}
void KSIntSurfaceUCN::ExecuteTransmission(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                          KSParticleQueue&)
{
    aFinalParticle = anInitialParticle;
    return;
}

double KSIntSurfaceUCN::ValueFunction(const double& aValue) const
{
    return fTanThetaIn * exp(-fExpThetaCoef * aValue * aValue) / 2 / sqrt(katrin::KConst::Pi() * fExpThetaCoef) +
           (1 + erf(aValue * sqrt(fExpThetaCoef))) / 2;
}

}  // namespace Kassiopeia
