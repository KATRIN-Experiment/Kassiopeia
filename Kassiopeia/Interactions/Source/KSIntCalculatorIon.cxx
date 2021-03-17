#include "KSIntCalculatorIon.h"

#include "KRandom.h"
#include "KSIntCalculatorHydrogen.h"
#include "KSParticleFactory.h"
using katrin::KRandom;

#include "KConst.h"

using KGeoBag::KThreeVector;

namespace Kassiopeia
{
KSIntCalculatorIon::KSIntCalculatorIon() : fGas("H_2"), E_Binding(katrin::KConst::BindingEnergy_H2()) {}
KSIntCalculatorIon::KSIntCalculatorIon(const KSIntCalculatorIon& aCopy) : KSComponent(aCopy), fGas(aCopy.fGas) {}
KSIntCalculatorIon* KSIntCalculatorIon::Clone() const
{
    return new KSIntCalculatorIon(*this);
}
KSIntCalculatorIon::~KSIntCalculatorIon() = default;

//
// Total cross sections for ionization
//
void KSIntCalculatorIon::CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection)
{
    int aParticleID = aParticle.GetPID();
    double aEnergy = aParticle.GetKineticEnergy_eV() / 1000;  //put in keV

    //Ionization of H_2
    if (fGas.compare("H_2") == 0) {

        E_Binding = katrin::KConst::BindingEnergy_H2();

        //H+,D+,T+
        if (aParticleID == 2212 || aParticleID == 99041 || aParticleID == 99071) {
            aCrossSection = Hplus_H2_crossSection(aEnergy);
        }

        //H2+,D2+,T2+
        else if (aParticleID == 99012 || aParticleID == 99042 || aParticleID == 99072) {
            aCrossSection = H2plus_H2_crossSection(aEnergy);
        }

        //H3+,D3+,T3+
        else if (aParticleID == 99013 || aParticleID == 99043 || aParticleID == 99073) {
            aCrossSection = H3plus_H2_crossSection(aEnergy);
        }

        else {
            aCrossSection = 0.;
            return;
        }
    }

    //Ionization of H2O
    else if (fGas.compare("H2O") == 0) {

        E_Binding = katrin::KConst::BindingEnergy_H2O();

        //H+
        if (aParticleID == 2212) {
            aCrossSection = KSIntCalculatorIon::Hplus_H2O_crossSection(aEnergy);
        }
    }

    //else if (fGas.compare("He") == 0) {

    //E_Binding = katrin::KConst::BindingEnergy_He;

    //Add cross sections for He...

    // }

    else {
        aCrossSection = 0.;
        return;
    }

    return;
}

//
//Total ionization cross sections
//
//TATSUO TABATA, TOSHIZO SHIRAI
//ANALYTIC CROSS SECTIONS FOR COLLISIONS OF H+, H2+, H3+, H, H2, AND H− WITH HYDROGEN MOLECULES
//Atomic Data and Nuclear Data Tables, Volume 76, Issue 1, 2000, Pages 1-25, ISSN 0092-640X
//http://dx.doi.org/10.1006/adnd.2000.0835

// H^+ + H_2 -> e^-
//analytic expression #9 in TABATA SHIRAI (page 8-9)
double KSIntCalculatorIon::Hplus_H2_crossSection(double aEnergy)
{
    const double E_threshold = 2.0e-2;  //keV
    const double a1 = 1.864e-4;
    const double a2 = 1.216;
    const double a3 = 5.31e1;
    const double a4 = 8.97e-1;
    //const double E_min = 7.50e-2;
    const double E_max = 1.00e+2;
    double E1 = E_1(aEnergy, E_threshold);
    double value = 0;
    if ((aEnergy > E_threshold) && (aEnergy < E_max)) {
        value = sigma1(E1, a1, a2, a3, a4);
    }
    return value;
}

// H_2^+ + H_2 -> e^-
//analytic expression #17 in TABATA SHIRAI (page 8-9)
double KSIntCalculatorIon::H2plus_H2_crossSection(double aEnergy)
{
    const double E_threshold = 3.0e-2;
    const double a1 = 1.086e-3;
    const double a2 = 1.153;
    const double a3 = 1.24e+1;
    const double a4 = -4.44e-1;
    const double a5 = 5.96e+1;
    const double a6 = 1.0;
    //const double E_min = 3.16e-2;
    const double E_max = 1.00e+2;
    double E1 = E_1(aEnergy, E_threshold);
    double value = 0;
    if ((aEnergy > E_threshold) && (aEnergy < E_max)) {
        value = sigma6(E1, a1, a2, a3, a4, a5, a6);
    }
    return value;
}

// H_3^+ + H_2 -> e^-
//analytic expression #24 in TABATA SHIRAI (page 8-9)
double KSIntCalculatorIon::H3plus_H2_crossSection(double aEnergy)
{
    const double E_threshold = 3.6e-2;
    const double a1 = 2.63e-3;
    const double a2 = 9.31e-1;
    const double a3 = 4.05e-1;
    const double a4 = 1.0;
    const double a5 = 1.26e+2;
    const double a6 = 2.13e+2;
    //const double E_min = 7.50e-2;
    const double E_max = 1.00e+2;
    double E1 = E_1(aEnergy, E_threshold);
    double value = 0;
    if ((aEnergy > E_threshold) && (aEnergy < E_max)) {
        value = sigma2(E1, a1, a2, a3, a4, a5, a6);
    }
    return value;
}

//
//Functions used in total cross sections from TABATA SHIRAI (page 3)
//
double KSIntCalculatorIon::f1(double x, double c1, double c2)
{
    double ERyd_keV = katrin::KConst::ERyd_eV() / 1000;
    double sigma0 = 1e-20;  //m^2
    double value = sigma0 * c1 * pow((x / ERyd_keV), c2);
    return value;
}

double KSIntCalculatorIon::f2(double x, double c1, double c2, double c3, double c4)
{
    double value = f1(x, c1, c2) / (1 + pow((x / c3), c2 + c4));
    return value;
}

double KSIntCalculatorIon::f3(double x, double c1, double c2, double c3, double c4, double c5, double c6)
{
    double value = f1(x, c1, c2) / (1 + pow((x / c3), c2 + c4) + pow((x / c5), c2 + c6));
    return value;
}

double KSIntCalculatorIon::sigma1(double E1, double a1, double a2, double a3, double a4)
{
    double value = f2(E1, a1, a2, a3, a4);
    return value;
}

double KSIntCalculatorIon::sigma2(double E1, double a1, double a2, double a3, double a4, double a5, double a6)
{
    double value = f2(E1, a1, a2, a3, a4) + a5 * f2(E1 / a6, a1, a2, a3, a4);
    return value;
}

double KSIntCalculatorIon::sigma6(double E1, double a1, double a2, double a3, double a4, double a5, double a6)
{
    double value = f3(E1, a1, a2, a3, a4, a5, a6);
    return value;
}

double KSIntCalculatorIon::sigma10(double E1, double a1, double a2, double a3, double a4, double a5, double a6,
                                   double a7, double a8)
{
    double value = f3(E1, a1, a2, a3, a4, a5, a6) + a7 * f3(E1 / a8, a1, a2, a3, a4, a5, a6);
    return value;
}

//E is given in keV
double KSIntCalculatorIon::E_1(double E, double E_threshold)
{
    double value = E - E_threshold;
    return value;
}

//
// Total ionization cross section for H+ on H2O
//
//Taken from M. E. Rudd, T. V. Goffe, R. D. DuBois, and L. H. Toburen,
//Cross sections for ionization of water vapor by 7–4000-keV protons
//Phys. Rev. A 31, 492 – Published 1 January 1985
//https://doi.org/10.1103/PhysRevA.31.492
//
double KSIntCalculatorIon::Hplus_H2O_crossSection(double aEnergy)
{
    const double A = 2.98;
    const double B = 4.42;
    const double C = 1.48;
    const double D = 0.75;
    const double E_min = 100e-3;  //From Geant4, see https://doi.org/10.1118/1.3476457
    const double E_max = 5000;
    double value = 0;
    if ((aEnergy > E_min) && (aEnergy < E_max)) {
        value = sigmatot(aEnergy, A, B, C, D);
    }
    return value;
}
//
//Functions used in total cross sections for H2O from Rudd, 1985
//
double KSIntCalculatorIon::sigmatot(double aEnergy, double A, double B, double C, double D)
{
    double T = aEnergy / 1836;
    double R = katrin::KConst::ERyd_eV() / 1000;  //put in keV
    double x = T / R;
    double value = 1 / (1 / (sigmalow(x, C, D)) + 1 / (sigmahigh(x, A, B)));
    return value;
}
double KSIntCalculatorIon::sigmalow(double x, double C, double D)
{
    double value = 4 * katrin::KConst::Pi() * katrin::KConst::BohrRadiusSquared() * (C * pow(x, D));
    return value;
}

double KSIntCalculatorIon::sigmahigh(double x, double A, double B)
{
    double value = 4 * katrin::KConst::Pi() * katrin::KConst::BohrRadiusSquared() * (A * log(1 + x) + B) / x;
    return value;
}


//
// Kinematics of the ionization
//

void KSIntCalculatorIon::ExecuteInteraction(const KSParticle& anIncomingIon, KSParticle& anOutgoingIon,
                                            KSParticleQueue& aSecondaries)
{
    // incoming primary ion
    double tIncomingIonEnergy = anIncomingIon.GetKineticEnergy_eV();
    KThreeVector tIncomingIonPosition = anIncomingIon.GetPosition();
    KThreeVector tIncomingIonMomentum = anIncomingIon.GetMomentum();
    double tIncomingIonMass = anIncomingIon.GetMass();

    // outgoing secondary electron
    //Create electron
    KSParticle* tSecondaryElectron = KSParticleFactory::GetInstance().StringCreate("e-");
    //Set position (same as initial particle)
    tSecondaryElectron->SetPosition(tIncomingIonPosition);
    //Set momentum (same as initial particle)
    tSecondaryElectron->SetMomentum(tIncomingIonMomentum);
    //Set energy
    double tSecondaryElectronEnergy = 0;
    CalculateSecondaryElectronEnergy(tIncomingIonMass, tIncomingIonEnergy, tSecondaryElectronEnergy);
    tSecondaryElectron->SetKineticEnergy_eV(tSecondaryElectronEnergy);

    //Set angle (isotropic). Should be improved with distribution from the literature
    //double tTheta = acos( KRandom::GetInstance().Uniform( -1., 1. ) )*180/katrin::KConst::Pi();
    //double tPhi = KRandom::GetInstance().Uniform( 0., 2. * katrin::KConst::Pi() )*180/katrin::KConst::Pi();
    //tSecondaryElectron->SetPolarAngleToZ( tTheta );
    //tSecondaryElectron->SetAzimuthalAngleToX( tPhi );

    //Set angle of secondary electron used differential cross section
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());  //radians
    double tTheta = 0;
    CalculateSecondaryElectronAngle(tTheta);       //use diff. cross section
    tTheta = tTheta * katrin::KConst::Pi() / 180;  //convert to radians
    KThreeVector tSecondaryMomentum = tSecondaryElectron->GetMomentum();

    //Correctly apply the angle in the incoming ion reference frame (code copied from KSIntCalculatorHydrogen.cxx)
    KThreeVector tOrthogonalOne = tIncomingIonMomentum.Orthogonal();
    KThreeVector tOrthogonalTwo = tIncomingIonMomentum.Cross(tOrthogonalOne);
    tSecondaryMomentum = tSecondaryMomentum.Magnitude() *
                         (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                          cos(tTheta) * tIncomingIonMomentum.Unit());
    tSecondaryElectron->SetMomentum(tSecondaryMomentum);


    tSecondaryElectron->SetLabel(GetName());
    aSecondaries.push_back(tSecondaryElectron);

    fStepAngularChange = tTheta * 180. / katrin::KConst::Pi();

    // outgoing primary ion
    anOutgoingIon = anIncomingIon;
    //Energy loss only from secondary electron and binding energy
    anOutgoingIon.SetKineticEnergy_eV(tIncomingIonEnergy - tSecondaryElectronEnergy - E_Binding);

    //Assume no deflection (this should be improved)
    //tTheta = acos( KRandom::GetInstance().Uniform( -1., 1. ) )*180/katrin::KConst::Pi();
    //tPhi = KRandom::GetInstance().Uniform( 0., 2. * katrin::KConst::Pi() )*180/katrin::KConst::Pi();
    //anOutgoingIon.SetPolarAngleToZ( tTheta );
    //anOutgoingIon.SetAzimuthalAngleToX( tPhi );

    // outgoing secondary ion (H2+). Available by not yet in use...
    //Isotropic distribution
    /*KSParticle* tSecondaryIon = KSParticleFactory::GetInstance().StringCreate( "H_2^+" );
    tSecondaryIon->SetPosition( tIncomingIonPosition );
    tSecondaryIon->SetPolarAngleToZ( tTheta );
    tSecondaryIon->SetAzimuthalAngleToX( tPhi );
    tSecondaryIon->SetKineticEnergy_eV( 0 );
    tSecondaryIon->SetLabel( GetName() );
    aSecondaries.push_back( tSecondaryIon );*/

    return;
}

//
// Calculate the secondary (ionization) electron energy
//
void KSIntCalculatorIon::CalculateSecondaryElectronEnergy(const double anIncomingIonMass,
                                                          const double anIncomingIonEnergy,
                                                          double& aSecondaryElectronEnergy)
{
    double tSecondaryElectronEnergy;
    double tCrossSection = 0;
    double I = E_Binding;

    //Diff. cross section for maximum possible secondary electron energy
    double aSecondaryElectronMaxEnergy =
        anIncomingIonEnergy - I;  //extremely unlikely/impossible for electron to have this high of an energy
    double sigma_max;
    CalculateEnergyDifferentialCrossSection(anIncomingIonMass,
                                            anIncomingIonEnergy,
                                            aSecondaryElectronMaxEnergy,
                                            sigma_max);

    //Diff. cross section for minimum possible secondary electron energy
    double aSecondaryElectronMinEnergy = 0;
    double sigma_min;
    CalculateEnergyDifferentialCrossSection(anIncomingIonMass,
                                            anIncomingIonEnergy,
                                            aSecondaryElectronMinEnergy,
                                            sigma_min);

    //Rejection sampling
    while (true) {
        //Randomly select a possible electron energy
        tSecondaryElectronEnergy =
            KRandom::GetInstance().Uniform(aSecondaryElectronMinEnergy, aSecondaryElectronMaxEnergy, false, true);
        //Get the diff. cross section for this electron energy
        CalculateEnergyDifferentialCrossSection(anIncomingIonMass,
                                                anIncomingIonEnergy,
                                                tSecondaryElectronEnergy,
                                                tCrossSection);
        //Randomly select a diff. cross section
        double tRandom = KRandom::GetInstance().Uniform(0., sigma_min, false, true);
        //Trying to optimize the random sampling
        //See https://am207.github.io/2017/wiki/rejectionsampling.html
        /*double tRandom = KRandom::GetInstance().Uniform( 0.,
							 2*( (sigma_max-sigma_min)*tSecondaryElectronEnergy/(anIncomingIonEnergy-I)+sigma_min),
							 false,true
							 );*/

        //If the random diff. cross section is less than the actual diff. cross section (i.e. it lies within the distribution), use the electron energy
        if (tRandom < tCrossSection)
            break;
    }

    aSecondaryElectronEnergy = tSecondaryElectronEnergy;
}

//
//Differential cross section for secondary electron energy
//
void KSIntCalculatorIon::CalculateEnergyDifferentialCrossSection(const double anIncomingIonMass,
                                                                 const double anIncomingIonEnergy,
                                                                 const double aSecondaryElectronEnergy,
                                                                 double& aCrossSection)
{
    //
    // Ionization of H_2 and H2O by H^+
    // H^+ on H_2 data was applied to other hydrogen ions as well
    //
    //Taken from
    //M. E. Rudd et al., Electron production in proton collisions with atoms and molecules: energy distributions, Rev. Mod. Phys. 64, 441, Published 1 April 1992
    //https://doi.org/10.1103/RevModPhys.64.441
    //See also M. E. Rudd, Differential cross sections for secondary electron production by proton impact, Phys. Rev. A 38, 6129, 1 December 1988
    //https://doi.org/10.1103/PhysRevA.38.6129
    //

    aCrossSection = 0;

    //for H2
    std::vector<std::vector<double>> shell_list = {{15.43, 2}};  //list of I and N from Table 1a in Rudd 1992

    //for H2O
    if (fGas.compare("H2O") == 0) {
        shell_list = {{12.61, 2},
                      {14.73, 2},
                      {18.55, 2},
                      {32.2, 2},
                      {539.7, 2}};  //list of I and N from Table 1b in Rudd 1992
    }

    //iterate over shells
    for (int s = 0; s < (int) shell_list.size(); s++) {

        //for H2
        double A_1 = 0.96;
        double B_1 = 2.6;
        double C_1 = 0.38;
        double D_1 = 0.23;
        double E_1 = 2.2;
        double A_2 = 1.04;
        double B_2 = 5.9;
        double C_2 = 1.15;
        double D_2 = 0.20;
        double alpha = 0.87;

        //for H2O
        if (fGas.compare("H2O") == 0) {
            if (s < 3) {  //for outer shells (I < 2 * 12.61)
                A_1 = 0.97;
                B_1 = 82.0;
                C_1 = 0.40;
                D_1 = -0.30;
                E_1 = 0.38;
                A_2 = 1.04;
                B_2 = 17.3;
                C_2 = 0.76;
                D_2 = 0.04;
                alpha = 0.64;
            }
            else {  //for inner shells (I > 2 * 12.61)
                A_1 = 1.25;
                B_1 = 0.50;
                C_1 = 1.00;
                D_1 = 1.00;
                E_1 = 3.0;
                A_2 = 1.10;
                B_2 = 1.30;
                C_2 = 1.00;
                D_2 = 0.0;
                alpha = 0.66;
            }
        }
        double I = shell_list[s][0];  //Binding energy of target atom
        double N = shell_list[s][1];  //Occupation number of electrons in atomic subshell of target
        double S =
            4 * katrin::KConst::Pi() * katrin::KConst::BohrRadiusSquared() * N * pow(katrin::KConst::ERyd_eV() / I, 2);
        double lambda =
            anIncomingIonMass / katrin::KConst::M_el_kg();  //ratio of incoming projectile mass and electron mass
        double T = anIncomingIonEnergy / lambda;
        double v = sqrt(T / I);  //reduced initial velocity
        double w = aSecondaryElectronEnergy / I;
        aCrossSection += (S / I) * (F_1(v, A_1, B_1, C_1, D_1, E_1) + F_2(v, A_2, B_2, C_2, D_2) * w) * pow(1 + w, -3) *
                         pow(1 + exp(alpha * (w - w_c(v, I)) / v), -1);
    }

    // For comparing with plots in Rudd 1992 paper
    //double lambda = anIncomingIonMass/katrin::KConst::M_el_kg(); //ratio of incoming projectile mass and electron mass
    //double T = anIncomingIonEnergy/lambda;
    //double I_1 = shell_list[0][0];
    //double E = aSecondaryElectronEnergy + I_1;
    //double Y = T/(4*katrin::KConst::Pi()*katrin::KConst::BohrRadiusSquared()) * pow(E/katrin::KConst::ERyd_eV(),2) * aCrossSection;
    //aCrossSection = Y;
}

//
//Functions used in Rudd 1988, Rudd 1992
//
double KSIntCalculatorIon::F_1(double v, double A_1, double B_1, double C_1, double D_1, double E_1)
{
    double value = L_1(v, C_1, D_1, E_1) + H_1(v, A_1, B_1);
    return value;
}
double KSIntCalculatorIon::F_2(double v, double A_2, double B_2, double C_2, double D_2)
{
    double value = L_2(v, C_2, D_2) * H_2(v, A_2, B_2) / (L_2(v, C_2, D_2) + H_2(v, A_2, B_2));
    return value;
}
double KSIntCalculatorIon::H_1(double v, double A_1, double B_1)
{
    double value = A_1 * log(1 + pow(v, 2)) / (pow(v, 2) + B_1 / pow(v, 2));
    return value;
}
double KSIntCalculatorIon::H_2(double v, double A_2, double B_2)
{
    double value = A_2 / pow(v, 2) + B_2 / pow(v, 4);
    return value;
}
double KSIntCalculatorIon::L_1(double v, double C_1, double D_1, double E_1)
{
    double value = C_1 * pow(v, D_1) / (1 + E_1 * pow(v, D_1 + 4));
    return value;
}
double KSIntCalculatorIon::L_2(double v, double C_2, double D_2)
{
    double value = C_2 * pow(v, D_2);
    return value;
}
double KSIntCalculatorIon::w_c(double v, double I)
{
    double w_2 = katrin::KConst::ERyd_eV() / (4 * I);
    double value = 4 * pow(v, 2) - 2 * v - w_2;
    return value;
}

//
// Calculate the secondary (ionization) electron angle
//
void KSIntCalculatorIon::CalculateSecondaryElectronAngle(double& aSecondaryElectronAngle)
{
    double tSecondaryElectronAngle;
    double tCrossSection = 0;

    //Diff. cross section for maximum possible secondary electron angle
    double aSecondaryElectronMaxAngle = 180;
    double sigma_max;
    CalculateAngleDifferentialCrossSection(aSecondaryElectronMaxAngle, sigma_max);

    //Diff. cross section for minimum possible secondary electron energy
    double aSecondaryElectronMinAngle = 0;
    double sigma_min;
    CalculateAngleDifferentialCrossSection(aSecondaryElectronMinAngle, sigma_min);

    //Rejection sampling
    while (true) {
        //Randomly select a possible electron angle
        tSecondaryElectronAngle =
            KRandom::GetInstance().Uniform(aSecondaryElectronMinAngle, aSecondaryElectronMaxAngle, false, true);
        //Get the diff. cross section for this electron angle
        CalculateAngleDifferentialCrossSection(tSecondaryElectronAngle, tCrossSection);
        //Randomly select a diff. cross section
        double tRandom = KRandom::GetInstance().Uniform(0., sigma_min, false, true);

        //If the random diff. cross section is less than the actual diff. cross section (i.e. it lies within the distribution), use the electron angle
        if (tRandom < tCrossSection)
            break;
    }

    aSecondaryElectronAngle = tSecondaryElectronAngle;
}


//
//Differential cross section for secondary electron angle
//
void KSIntCalculatorIon::CalculateAngleDifferentialCrossSection(const double aSecondaryElectronAngle,
                                                                double& aCrossSection)
{
    aCrossSection = 0;

    //
    //for H^+ on H2
    //
    //Fit from 20-keV datapoints from M. W. Gealy, G. W. Kerby, III, Y.-Y. Hsu, and M. E. Rudd
    //Energy and angular distributions of electrons from ion impact on atomic and molecular hydrogen. I. 20–114-keV H+ + H2
    //Phys. Rev. A 51, 2247 – Published 1 March 1995
    //https://doi.org/10.1103/PhysRevA.51.2247

    double p0 = 1.64084e-22;
    double p1 = 2.92559e-20;
    double p2 = 3.88393e+09;
    double p3 = 0.747347;
    double p4 = 2.80982e+06;

    aCrossSection = p0 + (p1 - p0) / pow(1 + pow(aSecondaryElectronAngle / p2, p3), p4);  //units of m^2

    //for H2O
    if (fGas.compare("H2O") == 0) {

        //
        //for H^+ on H2O
        //
        //Fit from 15-keV ion, 10 eV electron datapoints from M. A. Bolorizadeh and M. E. Rudd
        //Angular and energy dependence of cross sections for ejection of electrons from water vapor. II. 15–150-keV proton impact
        //Phys. Rev. A 33, 888 – Published 1 February 1986
        //https://doi.org/10.1103/PhysRevA.33.888

        double p0 = 8.02532e-23;
        double p1 = 5.81735e-22;
        double p2 = 1.51111e+02;
        double p3 = 2.16586e+00;
        double p4 = 2.71416e+01;

        aCrossSection = p0 + (p1 - p0) / pow(1 + pow(aSecondaryElectronAngle / p2, p3), p4);  //units of m^2
    }
}
}  // namespace Kassiopeia
