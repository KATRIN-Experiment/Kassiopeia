/**
 * @file KSIntCalculatorTritium.cxx
 *
 * @date 01.12.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#include "KSIntCalculatorTritium.h"

#include "KRandom.h"
#include "KThreeVector.hh"

using namespace std;
using namespace katrin;
using namespace KGeoBag;

constexpr double T2H2Ratio = 6.032099 / 2.015650;

namespace Kassiopeia
{

/////////////////////////////////
/////       Elastic         /////
/////////////////////////////////

void KSIntCalculatorTritiumElastic::CalculateCrossSection(const double anEnergie, double& aCrossSection)
{
    //        See: Liu, Phys. Rev. A35 (1987) 591,
    //        Trajmar, Phys Reports 97 (1983) 221.

    const double e[14] = {0., 1.5, 5., 7., 10., 15., 20., 30., 60., 100., 150., 200., 300., 400.};
    const double s[14] = {9.6, 13., 15., 12., 10., 7., 5.6, 3.3, 1.1, 0.9, 0.5, 0.36, 0.23, 0.15};

    const double emass = 1. / (katrin::KConst::Alpha() * katrin::KConst::Alpha());
    const double a02 = katrin::KConst::BohrRadiusSquared();

    double gam, T;
    T = anEnergie / (2 * katrin::KConst::ERyd_eV());
    if (anEnergie >= 400.) {
        gam = (emass + T) / emass;
        aCrossSection = gam * gam * katrin::KConst::Pi() / (2. * T) * (4.2106 - 1. / T) * a02;
    }
    else {
        for (unsigned int i = 0; i <= 12; i++) {
            if (anEnergie >= e[i] && anEnergie < e[i + 1])
                aCrossSection = 1.e-20 * (s[i] + (s[i + 1] - s[i]) * (anEnergie - e[i]) / (e[i + 1] - e[i]));
        }
    }

    return;
}

void KSIntCalculatorTritiumElastic::CalculateEloss(const double anEnergie, const double aTheta, double& anEloss)
{
    double H2molmass = 69.e6;
    double emass = 1. / (katrin::KConst::Alpha() * katrin::KConst::Alpha());
    double cosTheta = cos(aTheta);

    anEloss = 2. * emass / (T2H2Ratio * H2molmass) * (1. - cosTheta) * anEnergie;

    //check if electron won energy by elastic scattering on a molecule;
    //this keeps electron energies around the gas temperature
    if (anEnergie < 1.) {
        double rndNr = sqrt(-2. * log(KRandom::GetInstance().Uniform()));
        double rndAngle = 2. * katrin::KConst::Pi() * KRandom::GetInstance().Uniform();

        //generation of molecule velocity by maxwell-boltzmann distribution
        double Gx = rndNr * cos(rndAngle);
        double Gy = rndNr * sin(rndAngle);
        double Gz = sqrt(-2. * log(KRandom::GetInstance().Uniform())) *
                    cos(2. * katrin::KConst::Pi() * KRandom::GetInstance().Uniform());

        //thermal velocity of gas molecules
        double T = 300.;  //gas temperature
        double sigmaT = sqrt(katrin::KConst::kB() * T / (2. * katrin::KConst::M_prot_kg()));
        KThreeVector MolVelocity(sigmaT * Gx, sigmaT * Gy, sigmaT * Gz);

        //new electron velocity vector and energy:

        //assume electron velocity along z
        KThreeVector ElVelocity(0., 0., sqrt(2. * anEnergie * katrin::KConst::Q() / katrin::KConst::M_el_kg()));
        //relative velocity electron-molecule
        KThreeVector RelativeVelocity = ElVelocity - MolVelocity;
        //transformation into CMS
        KThreeVector CMSVelocity =
            (katrin::KConst::M_el_kg() / (katrin::KConst::M_el_kg() + katrin::KConst::M_prot_kg()) * ElVelocity +
             2. * katrin::KConst::M_prot_kg() * MolVelocity /
                 (katrin::KConst::M_el_kg() + katrin::KConst::M_prot_kg()));

        //generation of random direction
        KThreeVector Random(KRandom::GetInstance().Uniform(),
                            KRandom::GetInstance().Uniform(),
                            KRandom::GetInstance().Uniform());

        //new electron velocity
        ElVelocity = katrin::KConst::M_prot_kg() / (katrin::KConst::M_prot_kg() + katrin::KConst::M_el_kg()) *
                         RelativeVelocity.Magnitude() * Random.Unit() +
                     CMSVelocity;

        anEloss = anEnergie - katrin::KConst::M_el_kg() / (2. * katrin::KConst::Q()) * ElVelocity.Magnitude() *
                                  ElVelocity.Magnitude();
    }
    return;
}

} /* namespace Kassiopeia */
