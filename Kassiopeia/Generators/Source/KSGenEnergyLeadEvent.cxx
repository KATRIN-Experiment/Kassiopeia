#include "KSGenEnergyLeadEvent.h"

#include "KRandom.h"
#include "KSGenConversion.h"
#include "KSGenRelaxation.h"
#include "KSGenShakeOff.h"
#include "KSGeneratorsMessage.h"
#include "KSParticleFactory.h"

using katrin::KRandom;

namespace Kassiopeia
{

KSGenEnergyLeadEvent::KSGenEnergyLeadEvent() :
    fForceConversion(false),
    fDoConversion(true),
    fDoAuger(true),
    fIsotope(210),
    fZDaughter(83),
    fFermiMax17(0.),
    fFermiMax63(0.),
    fnmax(1000),
    fBismuthRelaxation(nullptr),
    fBismuthConversion(nullptr)
{}
KSGenEnergyLeadEvent::KSGenEnergyLeadEvent(const KSGenEnergyLeadEvent& aCopy) :
    KSComponent(aCopy),
    fForceConversion(aCopy.fForceConversion),
    fDoConversion(aCopy.fDoConversion),
    fDoAuger(aCopy.fDoAuger),
    fIsotope(aCopy.fIsotope),
    fZDaughter(aCopy.fZDaughter),
    fFermiMax17(aCopy.fFermiMax17),
    fFermiMax63(aCopy.fFermiMax63),
    fnmax(aCopy.fnmax),
    fBismuthRelaxation(aCopy.fBismuthRelaxation),
    fBismuthConversion(aCopy.fBismuthConversion)
{}
KSGenEnergyLeadEvent* KSGenEnergyLeadEvent::Clone() const
{
    return new KSGenEnergyLeadEvent(*this);
}
KSGenEnergyLeadEvent::~KSGenEnergyLeadEvent() = default;

void KSGenEnergyLeadEvent::Dice(KSParticleQueue* aPrimaries)
{
    KSParticle* tParticle;
    KSParticleQueue tParticles;
    KSParticleIt tParticleIt;

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {

        if (KRandom::GetInstance().Uniform() > 0.84) {
            tParticle = new KSParticle(**tParticleIt);
            tParticle->SetKineticEnergy_eV(GenBetaEnergy(63.5e3, 0., fFermiMax63, fZDaughter));
            tParticle->SetLabel("lead_beta_decay_63");
            tParticles.push_back(tParticle);

            continue;
        }

        tParticle = new KSParticle(**tParticleIt);
        tParticle->SetKineticEnergy_eV(GenBetaEnergy(17.e3, 0., fFermiMax17, fZDaughter));
        tParticle->SetLabel("lead_beta_decay_17");
        tParticles.push_back(tParticle);

        //***********
        //conversions
        //***********

        std::vector<int> conversionVacancy;
        std::vector<double> conversionElectronEnergy;

        if (fDoConversion == true) {
            fBismuthConversion->SetForceCreation(fForceConversion);
            fBismuthConversion->CreateCE(conversionVacancy, conversionElectronEnergy);

            for (unsigned int i = 0; i < conversionElectronEnergy.size(); i++) {
                genmsg_debug("KSGenEnergyLeadEvent:: Execute" << ret);
                genmsg_debug("Vacancy at: " << conversionVacancy.at(i) << ret);
                genmsg_debug("   Conversion electron energy: " << conversionElectronEnergy.at(i) << eom);
                fBismuthRelaxation->GetVacancies()->push_back(conversionVacancy.at(i));
                tParticle = new KSParticle(**tParticleIt);
                tParticle->SetKineticEnergy_eV(conversionElectronEnergy.at(i));
                tParticle->SetLabel("bismuth_conversion");
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
                fBismuthRelaxation->ClearAugerEnergies();
                fBismuthRelaxation->Relax();
                for (unsigned int i = 0; i < fBismuthRelaxation->GetAugerEnergies().size(); i++) {

                    genmsg_debug("Auger energy: " << fBismuthRelaxation->GetAugerEnergies().at(i) << eom);
                    tParticle = new KSParticle(**tParticleIt);
                    tParticle->SetKineticEnergy_eV(fBismuthRelaxation->GetAugerEnergies().at(i));
                    tParticle->SetLabel("bismuth_auger");
                    tParticles.push_back(tParticle);
                }
                fBismuthRelaxation->ClearVacancies();
            }
            else {
                genmsg_debug("There is no vacancy, therefore no auger electron can be produced !" << eom);
            }
        }
        else {
            genmsg_debug("Auger electron production not activated!" << eom);
        }
    }

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = *tParticleIt;
        delete tParticle;
    }

    aPrimaries->assign(tParticles.begin(), tParticles.end());

    return;
}

double KSGenEnergyLeadEvent::Fermi(double E, double mnu, double E0, double Z)
{
    // This subr. computes the Fermi beta energy spectrum shape,
    // with neutrino mass
    //   E: electron kinetic energy in eV
    //  mnu: neutrino mass
    double beta, E1, p1, p2, E2, FC, x, Fermiret;
    E2 = katrin::KConst::M_el_eV() + E;                                          // electron total energy
    p2 = sqrt(E2 * E2 - katrin::KConst::M_el_eV() * katrin::KConst::M_el_eV());  // electron momentum
    beta = p2 / E2;
    E1 = E0 - E;  // neutrino total energy
    if (E1 >= mnu)
        p1 = sqrt(fabs(E1 * E1 - mnu * mnu));  // neutrino momentum
    else
        p1 = 0.;
    x = 2. * katrin::KConst::Pi() * Z * katrin::KConst::Alpha() / beta;
    FC = x / (1. - exp(-x));  // Coulomb correction factor
    Fermiret = p2 * E2 * p1 * E1 * FC;
    return Fermiret;
}

double KSGenEnergyLeadEvent::GetFermiMax(double E0, double mnu, double Z)
{
    double Fermimax = 0;

    for (int i = 0; i < fnmax; i++) {
        double E = (E0 / double(fnmax)) * i;
        double F = Fermi(E, mnu, E0, Z);
        if (F > Fermimax) {
            Fermimax = F;
        }
    }
    //Fermimax*=1.05;
    return Fermimax;
}

double KSGenEnergyLeadEvent::GenBetaEnergy(double E0, double mnu, double Fermimax, double Z)
{
    // Generation of E:
    double E = 0;
    double F = 0;
    do {
        E = E0 * KRandom::GetInstance().Uniform();
        F = Fermi(E, mnu, E0, Z);
    } while ((Fermimax * KRandom::GetInstance().Uniform()) > F);
    return E;
}

void KSGenEnergyLeadEvent::SetForceConversion(bool aSetting)
{
    fForceConversion = aSetting;
}
void KSGenEnergyLeadEvent::SetDoConversion(bool aSetting)
{
    fDoConversion = aSetting;
}
void KSGenEnergyLeadEvent::SetDoAuger(bool aSetting)
{
    fDoAuger = aSetting;
}


void KSGenEnergyLeadEvent::InitializeComponent()
{
    genmsg_debug("reading shake-off, conversion and relaxation data for radon..." << eom);

    fBismuthConversion = new KSGenConversion();
    fBismuthConversion->Initialize(fIsotope);

    fBismuthRelaxation = new KSGenRelaxation();
    fBismuthRelaxation->Initialize(fIsotope);

    fFermiMax17 = GetFermiMax(17e3, 0., fZDaughter);
    fFermiMax63 = GetFermiMax(63.5e3, 0., fZDaughter);

    genmsg_debug("...all data read, instances of shake-off, conversion and relaxation class created" << eom);

    return;
}
void KSGenEnergyLeadEvent::DeinitializeComponent()
{
    delete fBismuthConversion;
    delete fBismuthRelaxation;
    return;
}

}  // namespace Kassiopeia
