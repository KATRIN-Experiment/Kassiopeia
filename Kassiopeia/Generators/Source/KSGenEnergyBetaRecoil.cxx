//
// Created by wdconinc on 13.02.20.
//

#include "KSGenEnergyBetaRecoil.h"

#include "KSGeneratorsMessage.h"

//#include "KSParticleFactory.h"
#include "KRandom.h"

using namespace katrin;

namespace Kassiopeia
{

KSGenEnergyBetaRecoil::KSGenEnergyBetaRecoil() : fNMax(1000), fEMax(0.), fPMax(0.), fMinEnergy(0.), fMaxEnergy(-1.) {}
KSGenEnergyBetaRecoil::KSGenEnergyBetaRecoil(const KSGenEnergyBetaRecoil& aCopy) :
    KSComponent(),
    fNMax(aCopy.fNMax),
    fEMax(aCopy.fEMax),
    fPMax(aCopy.fPMax),
    fMinEnergy(aCopy.fMinEnergy),
    fMaxEnergy(aCopy.fMaxEnergy)
{}
KSGenEnergyBetaRecoil* KSGenEnergyBetaRecoil::Clone() const
{
    return new KSGenEnergyBetaRecoil(*this);
}
KSGenEnergyBetaRecoil::~KSGenEnergyBetaRecoil() {}

void KSGenEnergyBetaRecoil::Dice(KSParticleQueue* aPrimaries)
{
    KSParticleIt tParticleIt;

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        double tEnergy;
        do {
            tEnergy = GenRecoilEnergy();
        } while ((tEnergy < fMinEnergy) || (tEnergy > fMaxEnergy));

        (*tParticleIt)->SetKineticEnergy_eV(tEnergy);
        (*tParticleIt)->SetLabel(GetName());
    }

    return;
}

double KSGenEnergyBetaRecoil::g(double E)
{
    double a = -0.103;
    double g = g1(E) + a * g2(E);
    return g;
}

double KSGenEnergyBetaRecoil::g1(double E)
{
    static const double Delta = KConst::M_neut_eV() - KConst::M_prot_eV();
    static const double x = KConst::M_el_eV() / Delta;
    static const double x2 = x * x;
    static const double y = 2.0 * KConst::M_neut_eV() / (Delta * Delta);
    double s = 1.0 - E * y;
    double g0 = pow(1.0 - x2 / s, 2.0) * sqrt(1 - s);
    double g1 = g0 * (4 * (1 + x2 / s) - 4 / 3 * (s - x2) / s * (1 - s));
    return g1;
}

double KSGenEnergyBetaRecoil::g2(double E)
{
    static const double Delta = KConst::M_neut_eV() - KConst::M_prot_eV();
    static const double x = KConst::M_el_eV() / Delta;
    static const double x2 = x * x;
    static const double y = 2.0 * KConst::M_neut_eV() / (Delta * Delta);
    double s = 1.0 - E * y;
    double g0 = pow(1.0 - x2 / s, 2.0) * sqrt(1 - s);
    double g2 = g0 * (4 * (1 + x2 / s - 2 * s) - 4 / 3 * (s - x2) / s * (1 - s));
    return g2;
}

double KSGenEnergyBetaRecoil::GetRecoilEnergyMax()
{
    static const double Delta = KConst::M_neut_eV() - KConst::M_prot_eV();
    static const double EMax = (Delta * Delta - KConst::M_el_eV() * KConst::M_el_eV()) / (2 * KConst::M_neut_eV());
    return EMax;
}

double KSGenEnergyBetaRecoil::GetRecoilEnergyProbabilityMax(double Emax)
{
    double Pmax = 0;

    for (int i = 0; i < fNMax; i++) {
        double E = (Emax / double(fNMax)) * i;
        double P = g(E);
        if (P > Pmax) {
            Pmax = P;
        }
    }
    Pmax *= 1.05;
    return Pmax;
}
double KSGenEnergyBetaRecoil::GenRecoilEnergy()
{
    // Generation of E:
    double E = 0;
    double w = 0;

    do {

        E = KRandom::GetInstance().Uniform(fMinEnergy, fMaxEnergy);
        w = g(E);

    } while ((fPMax * KRandom::GetInstance().Uniform()) > w);

    return E;
}


void KSGenEnergyBetaRecoil::InitializeComponent()
{
    fEMax = GetRecoilEnergyMax();
    fPMax = GetRecoilEnergyProbabilityMax(fEMax);

    if (fMaxEnergy == -1.)
        fMaxEnergy = fEMax;

    return;
}

void KSGenEnergyBetaRecoil::DeinitializeComponent()
{
    return;
}

}  // namespace Kassiopeia
