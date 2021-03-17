#include "KSIntSurfaceMultiplication.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include "KConst.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSIntSurfaceMultiplication::KSIntSurfaceMultiplication() :
    KSComponent(),
    fPerformSideCheck(false),
    fSideSignIsNegative(false),
    fSideName(std::string("both")),
    fEnergyLossFraction(0.),
    fEnergyRequiredPerParticle(std::numeric_limits<double>::max())
{}

KSIntSurfaceMultiplication::KSIntSurfaceMultiplication(const KSIntSurfaceMultiplication& aCopy) :
    KSComponent(aCopy),
    fPerformSideCheck(aCopy.fPerformSideCheck),
    fSideSignIsNegative(aCopy.fSideSignIsNegative),
    fSideName(aCopy.fSideName),
    fEnergyLossFraction(aCopy.fEnergyLossFraction),
    fEnergyRequiredPerParticle(aCopy.fEnergyRequiredPerParticle)
{}

KSIntSurfaceMultiplication* KSIntSurfaceMultiplication::Clone() const
{
    return new KSIntSurfaceMultiplication(*this);
}

KSIntSurfaceMultiplication::~KSIntSurfaceMultiplication() = default;

void KSIntSurfaceMultiplication::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                    KSParticleQueue& aQueue)
{
    //determine the amount of energy we have to work with
    double tKineticEnergy = anInitialParticle.GetKineticEnergy();
    tKineticEnergy *= (1.0 - fEnergyLossFraction);

    //prevent kinetic energy from going negative
    if (tKineticEnergy < 0.0) {
        intmsg(eError) << "surface diffuse interaction named <" << GetName()
                       << "> tried to give a particle a negative kinetic energy." << eom;
        return;
    }

    //now determine the number of particles we will generate
    double tMean = tKineticEnergy / fEnergyRequiredPerParticle;
    unsigned int tNParticles = KRandom::GetInstance().Poisson(tMean);

    std::vector<double> tChildEnergy;
    if (tNParticles > 1) {
        //randomly partition energy, not completely equally distributed
        std::vector<double> tRandomSample;

        tRandomSample.push_back(0);
        for (unsigned int i = 0; i < tNParticles - 1; i++) {
            tRandomSample.push_back(KRandom::GetInstance().Uniform(0.0, 1.0));
        }
        //order from min to max
        std::sort(tRandomSample.begin(), tRandomSample.end());

        for (unsigned int i = 0; i < tNParticles; i++) {
            double e_val = 0;
            if (i + 1 < tNParticles) {
                e_val = (tRandomSample[i + 1] - tRandomSample[i]) * tKineticEnergy;
            }
            else {
                e_val = (1.0 - tRandomSample[i]) * tKineticEnergy;
            }

            tChildEnergy.push_back(e_val);
        }
    }
    else {
        if (tNParticles == 1) {
            tChildEnergy.push_back(tKineticEnergy);
        };
    }

    //figure out the basis directions for the particle ejections
    //we eject them with a diffuse 'Lambertian' distribution
    KThreeVector tNormal;
    if (anInitialParticle.GetCurrentSurface() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSurface()->Normal(anInitialParticle.GetPosition());
    }
    else if (anInitialParticle.GetCurrentSide() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSide()->Normal(anInitialParticle.GetPosition());
    }
    else {
        intmsg(eError) << "surface diffuse interaction named <" << GetName()
                       << "> was given a particle with neither a surface nor a side set" << eom;
        return;
    }

    KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
    KThreeVector momDirection = tInitialMomentum.Unit();

    double dot_prod = tInitialMomentum.Dot(tNormal);
    KThreeVector tInitialNormalMomentum = dot_prod * tNormal;

    tInitialNormalMomentum = -1.0 * tInitialNormalMomentum;  //reverse direction for reflection
    KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;

    tInitialNormalMomentum = tInitialNormalMomentum.Unit();
    tInitialTangentMomentum = tInitialTangentMomentum.Unit();
    KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross(tInitialNormalMomentum.Unit());

    bool execute_interaction = true;
    if (fPerformSideCheck) {

        if (fSideSignIsNegative && dot_prod > 0) {
            execute_interaction = false;
        }
        if (!fSideSignIsNegative && dot_prod < 0) {
            execute_interaction = false;
        }
    }

    if (execute_interaction)  //only execute interaction if the specified side of this surface is active
    {
        //now generate the ejected particles
        for (unsigned int i = 0; i < tNParticles; i++) {
            auto* tParticle = new KSParticle(anInitialParticle);

            //dice direction
            double tAzimuthalAngle = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
            double tSinTheta =
                KRandom::GetInstance().Uniform(0., 0.5);  //this is not a true lambertian (has cut-off angle)
            double tCosTheta = std::sqrt((1.0 - tSinTheta) * (1.0 + tSinTheta));

            KThreeVector tDirection;
            tDirection = tCosTheta * tInitialNormalMomentum;
            tDirection += tSinTheta * std::cos(tAzimuthalAngle) * tInitialTangentMomentum.Unit();
            tDirection += tSinTheta * std::sin(tAzimuthalAngle) * tInitialOrthogonalMomentum.Unit();

            if (tDirection.Dot(momDirection) > 0) {
                tDirection = -1.0 * tDirection;
            }

            tParticle->SetMomentum(tDirection);
            tParticle->SetKineticEnergy(tChildEnergy[i]);
            tParticle->SetCurrentSurface(nullptr);
            aQueue.push_back(tParticle);
        }
    }

    //kill parent
    aFinalParticle = anInitialParticle;
    aFinalParticle.SetActive(false);
    aFinalParticle.AddLabel(GetName());
    aFinalParticle.SetMomentum(-1.0 * tInitialMomentum);
    aFinalParticle.SetKineticEnergy(0);

    return;
}

}  // namespace Kassiopeia
