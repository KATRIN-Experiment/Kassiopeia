#include "KSGenEnergyRadonEvent.h"

#include "KRandom.h"
#include "KSGenConversion.h"
#include "KSGenRelaxation.h"
#include "KSGenShakeOff.h"
#include "KSGeneratorsMessage.h"
#include "KSParticleFactory.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenEnergyRadonEvent::KSGenEnergyRadonEvent() :
    fForceConversion(false),
    fForceShakeOff(false),
    fDoConversion(true),
    fDoShakeOff(true),
    fDoAuger(true),
    fIsotope(219),
    fMyRelaxation(nullptr),
    fMyShakeOff(nullptr),
    fMyConversion(nullptr)
{}
KSGenEnergyRadonEvent::KSGenEnergyRadonEvent(const KSGenEnergyRadonEvent& aCopy) :
    KSComponent(aCopy),
    fForceConversion(aCopy.fForceConversion),
    fForceShakeOff(aCopy.fForceShakeOff),
    fDoConversion(aCopy.fDoConversion),
    fDoShakeOff(aCopy.fDoShakeOff),
    fDoAuger(aCopy.fDoAuger),
    fIsotope(aCopy.fIsotope),
    fMyRelaxation(aCopy.fMyRelaxation),
    fMyShakeOff(aCopy.fMyShakeOff),
    fMyConversion(aCopy.fMyConversion)
{}
KSGenEnergyRadonEvent* KSGenEnergyRadonEvent::Clone() const
{
    return new KSGenEnergyRadonEvent(*this);
}
KSGenEnergyRadonEvent::~KSGenEnergyRadonEvent() = default;

void KSGenEnergyRadonEvent::Dice(KSParticleQueue* aPrimaries)
{
    KSParticle* tParticle;
    KSParticleQueue tParticles;
    KSParticleIt tParticleIt;

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {

        //**********
        //shake offs
        //**********

        std::vector<int> shakeOffVacancy;
        std::vector<double> shakeOffElectronEnergy;

        if (fDoShakeOff == true) {
            fMyShakeOff->SetForceCreation(fForceShakeOff);
            fMyShakeOff->CreateSO(shakeOffVacancy, shakeOffElectronEnergy);

            for (unsigned int i = 0; i < shakeOffElectronEnergy.size(); i++) {
                genmsg_debug("KSGenEnergyRadonEvent:: Execute" << ret);
                genmsg_debug("Vacancy at: " << shakeOffVacancy.at(i) << ret);
                genmsg_debug("Shake off electron energy: " << shakeOffElectronEnergy.at(i) << eom);
                fMyRelaxation->GetVacancies()->push_back(shakeOffVacancy.at(i));
                tParticle = new KSParticle(**tParticleIt);
                tParticle->SetKineticEnergy_eV(shakeOffElectronEnergy.at(i));
                tParticle->SetLabel("radon_shakeoff");
                tParticles.push_back(tParticle);
            }
        }
        else {
            genmsg_debug("KSGenGeneratorCompositeRadonEvent::Execute" << ret);
            genmsg_debug("Shake-off electron generation not activated!" << eom);
        }

        //******
        //augers after shakeoff
        //******

        if (fDoAuger == true) {
            if (shakeOffElectronEnergy.size() != 0) {
                fMyRelaxation->ClearAugerEnergies();
                fMyRelaxation->Relax();

                for (unsigned int i = 0; i < fMyRelaxation->GetAugerEnergies().size(); i++) {

                    genmsg_debug("KSGenGeneratorCompositeRadonEvent::Execute" << ret);
                    genmsg_debug("Auger energy: " << fMyRelaxation->GetAugerEnergies().at(i) << eom);
                    tParticle = new KSParticle(**tParticleIt);
                    tParticle->SetKineticEnergy_eV(fMyRelaxation->GetAugerEnergies().at(i));
                    tParticle->SetLabel("polonium_auger");
                    tParticles.push_back(tParticle);
                }
                fMyRelaxation->ClearVacancies();
            }
            else {
                genmsg_debug("KSGenGeneratorCompositeRadonEvent::Execute" << ret);
                genmsg_debug("There is no vacancy, therefore no auger electron can be produced !" << eom);
            }
        }
        else {
            genmsg_debug("KSGenGeneratorCompositeRadonEvent::Execute" << ret);
            genmsg_debug("Auger electron production not activated!" << eom);
        }

        //***********
        //conversions
        //***********

        std::vector<int> conversionVacancy;
        std::vector<double> conversionElectronEnergy;

        if (fDoConversion == true) {
            fMyConversion->SetForceCreation(fForceConversion);
            fMyConversion->CreateCE(conversionVacancy, conversionElectronEnergy);

            for (unsigned int i = 0; i < conversionElectronEnergy.size(); i++) {
                genmsg_debug("KSGenEnergyKryptonEvent:: Execute" << ret);
                genmsg_debug("Vacancy at: " << conversionVacancy.at(i) << ret);
                genmsg_debug("   Conversion electron energy: " << conversionElectronEnergy.at(i) << eom);
                fMyRelaxation->GetVacancies()->push_back(conversionVacancy.at(i));
                tParticle = new KSParticle(**tParticleIt);
                tParticle->SetKineticEnergy_eV(conversionElectronEnergy.at(i));
                tParticle->SetLabel("polonium_conversion");
                tParticles.push_back(tParticle);
            }
        }
        else {
            genmsg_debug("KSGenGeneratorCompositeRadonEvent::Execute" << ret);
            genmsg_debug("Conversion electron generation not activated!" << eom);
        }

        //******
        //augers after conversion
        //******

        if (fDoAuger == true) {
            if (conversionElectronEnergy.size() != 0) {
                fMyRelaxation->ClearAugerEnergies();
                fMyRelaxation->Relax();
                for (unsigned int i = 0; i < fMyRelaxation->GetAugerEnergies().size(); i++) {

                    genmsg_debug("Auger energy: " << fMyRelaxation->GetAugerEnergies().at(i) << eom);
                    tParticle = new KSParticle(**tParticleIt);
                    tParticle->SetKineticEnergy_eV(fMyRelaxation->GetAugerEnergies().at(i));
                    tParticle->SetLabel("polonium_auger");
                    tParticles.push_back(tParticle);
                }
                fMyRelaxation->ClearVacancies();
            }
            else {
                genmsg_debug("There is no vacancy, therefore no auger electron can be produced !" << eom);
            }
        }
        else {
            genmsg_debug("Auger electron production not activated!" << eom);
        }

        //********************
        //shell reorganization
        //********************

        if (shakeOffElectronEnergy.size() == 0 && conversionElectronEnergy.size() == 0) {
            double shellElectronEnergy;
            double surplusEnergy;
            shellElectronEnergy = KRandom::GetInstance().Uniform() * 230.;
            surplusEnergy = 230. - shellElectronEnergy;
            genmsg_debug("shell reorganization electron energies: " << surplusEnergy << " and " << shellElectronEnergy
                                                                    << eom);

            //First Paricle
            tParticle = new KSParticle(**tParticleIt);
            tParticle->SetKineticEnergy_eV(shellElectronEnergy);
            tParticle->SetLabel("radon_shell_reorganisation");
            tParticles.push_back(tParticle);

            //Second Particle
            tParticle = new KSParticle(**tParticleIt);
            tParticle->SetKineticEnergy_eV(surplusEnergy);
            tParticle->SetLabel("radon_shell_reorganisation");
            tParticles.push_back(tParticle);
        }
        else {
            genmsg_debug("shake-off or conversion electron was produced, therefore no shell reorganization can occur"
                         << eom);
        }
    }

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = *tParticleIt;
        delete tParticle;
    }

    aPrimaries->assign(tParticles.begin(), tParticles.end());

    return;
}

void KSGenEnergyRadonEvent::SetForceShakeOff(bool aSetting)
{
    fForceShakeOff = aSetting;
}
void KSGenEnergyRadonEvent::SetForceConversion(bool aSetting)
{
    fForceConversion = aSetting;
}
void KSGenEnergyRadonEvent::SetDoShakeOff(bool aSetting)
{
    fDoShakeOff = aSetting;
}
void KSGenEnergyRadonEvent::SetDoConversion(bool aSetting)
{
    fDoConversion = aSetting;
}
void KSGenEnergyRadonEvent::SetDoAuger(bool aSetting)
{
    fDoAuger = aSetting;
}
void KSGenEnergyRadonEvent::SetIsotope(int anIsotope)
{
    fIsotope = anIsotope;
}

void KSGenEnergyRadonEvent::InitializeComponent()
{
    genmsg_debug("reading shake-off, conversion and relaxation data for radon..." << eom);

    fMyShakeOff = new KSGenShakeOff();

    fMyConversion = new KSGenConversion();
    fMyConversion->Initialize(fIsotope);

    fMyRelaxation = new KSGenRelaxation();
    fMyRelaxation->Initialize(fIsotope);

    genmsg_debug("...all data read, instances of shake-off, conversion and relaxation class created" << eom);

    return;
}
void KSGenEnergyRadonEvent::DeinitializeComponent()
{
    delete fMyShakeOff;
    delete fMyConversion;
    delete fMyRelaxation;
    return;
}

}  // namespace Kassiopeia
