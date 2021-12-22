#include "KESSSurfaceInteraction.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

using katrin::KThreeVector;

#include <cmath>

namespace Kassiopeia
{

KESSSurfaceInteraction::KESSSurfaceInteraction() :
    fElectronDirection(eEnteringSilicon),
    fElectronAffinity(-4.05),
    fSurfaceOrientation(eNormalPointingAway)
{}

KESSSurfaceInteraction::KESSSurfaceInteraction(const KESSSurfaceInteraction& aCopy) :
    KSComponent(aCopy),
    fElectronDirection(aCopy.fElectronDirection),
    fElectronAffinity(aCopy.fElectronAffinity),
    fSurfaceOrientation(aCopy.fSurfaceOrientation)
{}

KESSSurfaceInteraction* KESSSurfaceInteraction::Clone() const
{
    return new KESSSurfaceInteraction(*this);
}

KESSSurfaceInteraction::~KESSSurfaceInteraction() = default;

double KESSSurfaceInteraction::CalculateTransmissionProbability(const double aKineticEnergy,
                                                                const double aCosIncidentAngle)
{
    double tChi = aKineticEnergy * aCosIncidentAngle * aCosIncidentAngle;

    if (tChi < fElectronAffinity) {
        intmsg_debug("kess surface interaction forbids transmission [chi =  "
                     << tChi << ", energy = " << aKineticEnergy
                     << ", angle = " << std::acos(aCosIncidentAngle) * 180. / katrin::KConst::Pi() << "]" << eom);

        return 0.;
    }
    else {
        double tRoot = sqrt(1. - fElectronAffinity / tChi);
        double tNumerator = 4. * tRoot;
        double tDenominator2 = 1 + 2 * tRoot + tRoot * tRoot;
        double tProbability = tNumerator / tDenominator2;

        intmsg_debug("kess surface interaction transmission probability of "
                     << tProbability << " [chi =  " << tChi << ", energy = " << aKineticEnergy
                     << ", angle = " << std::acos(aCosIncidentAngle) * 180. / katrin::KConst::Pi() << "]" << eom);

        return tProbability;
    }
}

void KESSSurfaceInteraction::ExecuteTransmission(const KSParticle& anInitialParticle, KSParticle& aFinalParticle)
{

    double tEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tMomentum = anInitialParticle.GetMomentum();

    KThreeVector tNormal;
    KSSurface* tSurface = anInitialParticle.GetCurrentSurface();
    if (tSurface != nullptr) {
        tNormal = tSurface->Normal(anInitialParticle.GetPosition());
    }
    else {
        KSSide* tSide = anInitialParticle.GetCurrentSide();
        if (tSide != nullptr) {
            tNormal = tSide->Normal(anInitialParticle.GetPosition());
        }
        else {
            intmsg(eError) << "KSIntSurfaceDiffuse: particle has neither a surface, nor a side. Stopping!" << eom;
        }
    }


    double tCosAngle = tNormal.Dot(tMomentum.Unit());
    double tChi = tEnergy * tCosAngle * tCosAngle;

    double tTheta = acos(sqrt((tChi - fElectronAffinity) / (tEnergy - fElectronAffinity)));
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());


    KThreeVector tInitialDirection = tMomentum.Unit();
    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);

    tMomentum =
        tMomentum.Magnitude() * (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                                 cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tMomentum);
    aFinalParticle.SetKineticEnergy_eV(tEnergy + fElectronAffinity);

    return;
}

void KESSSurfaceInteraction::ExecuteReflection(const KSParticle& anInitialParticle, KSParticle& aFinalParticle)
{
    KThreeVector tNormal;
    KSSurface* tSurface = anInitialParticle.GetCurrentSurface();
    if (tSurface != nullptr) {
        tNormal = tSurface->Normal(anInitialParticle.GetPosition());
    }
    else {
        KSSide* tSide = anInitialParticle.GetCurrentSide();
        if (tSide != nullptr) {
            tNormal = tSide->Normal(anInitialParticle.GetPosition());
        }
        else {
            intmsg(eError) << "KSIntSurfaceDiffuse: particle has neither a surface, nor a side. Stopping!" << eom;
        }
    }


    KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
    KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot(tNormal) * tNormal;
    KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tInitialTangentMomentum - tInitialNormalMomentum);

    return;
}

void KESSSurfaceInteraction::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                KSParticleQueue& /*aQueue*/)
{
    double tKineticEnergy = anInitialParticle.GetKineticEnergy_eV();

    KThreeVector tNormal;
    KSSurface* tSurface = anInitialParticle.GetCurrentSurface();
    if (tSurface != nullptr) {
        tNormal = tSurface->Normal(anInitialParticle.GetPosition());
    }
    else {
        KSSide* tSide = anInitialParticle.GetCurrentSide();
        if (tSide != nullptr) {
            tNormal = tSide->Normal(anInitialParticle.GetPosition());
        }
        else {
            intmsg(eError) << "KSIntSurfaceDiffuse: particle has neither a surface, nor a side. Stopping!" << eom;
        }
    }

    double tCosIncidentAngle = tNormal.Unit().Dot(anInitialParticle.GetMomentum().Unit());

    if (tCosIncidentAngle > 0.) {
        if (fSurfaceOrientation == eNormalPointingAway) {
            fElectronDirection = eExitingSilicon;
        }
        else {
            fElectronDirection = eEnteringSilicon;
        }
    }
    if (tCosIncidentAngle <= 0.) {
        if (fSurfaceOrientation == eNormalPointingAway) {
            fElectronDirection = eEnteringSilicon;
        }
        else {
            fElectronDirection = eExitingSilicon;
        }
    }

    if (fElectronDirection == eEnteringSilicon) {
        fElectronAffinity = -4.05;
    }
    else {
        fElectronAffinity = 4.05;
    }

    double transmissionProbability = CalculateTransmissionProbability(tKineticEnergy, tCosIncidentAngle);

    double random = KRandom::GetInstance().Uniform(0., 1.0);

    if (random < transmissionProbability) {
        ExecuteTransmission(anInitialParticle, aFinalParticle);
    }
    else {

        ExecuteReflection(anInitialParticle, aFinalParticle);
    }
}

}  // namespace Kassiopeia
