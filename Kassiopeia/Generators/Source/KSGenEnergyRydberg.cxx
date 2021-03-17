//
// Created by trost on 14.03.16.
//

#include "KSGenEnergyRydberg.h"

#include "KConst.h"
#include "KSGeneratorsMessage.h"

//#include "KSParticleFactory.h"
#include "KRandom.h"

using katrin::KRandom;

namespace Kassiopeia
{

KSGenEnergyRydberg::KSGenEnergyRydberg() : fDepositedEnergy(0.), fIonizationEnergy(katrin::KConst::ERyd_eV()) {}
KSGenEnergyRydberg::KSGenEnergyRydberg(const KSGenEnergyRydberg& aCopy) :
    KSComponent(aCopy),
    fDepositedEnergy(aCopy.fDepositedEnergy),
    fIonizationEnergy(aCopy.fIonizationEnergy)
{}
KSGenEnergyRydberg* KSGenEnergyRydberg::Clone() const
{
    return new KSGenEnergyRydberg(*this);
}
KSGenEnergyRydberg::~KSGenEnergyRydberg() = default;

void KSGenEnergyRydberg::InitializeComponent() {}

void KSGenEnergyRydberg::DeinitializeComponent() {}

void KSGenEnergyRydberg::Dice(KSParticleQueue* aPrimaries)
{
    KSParticleIt tParticleIt;

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        int n = (*tParticleIt)->GetMainQuantumNumber();
        double tEnergy = fDepositedEnergy - fIonizationEnergy + fIonizationEnergy / n / n;

        (*tParticleIt)->SetKineticEnergy_eV(tEnergy);
        (*tParticleIt)->SetLabel(GetName());
    }


    return;
}

}  // namespace Kassiopeia
