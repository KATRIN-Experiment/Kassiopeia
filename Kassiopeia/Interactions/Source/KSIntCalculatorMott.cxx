#include "KSIntCalculatorMott.h"
#include <vector>
#include <algorithm>

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"

#include "KThreeVector.hh"
using katrin::KThreeVector;

namespace Kassiopeia
{

KSIntCalculatorMott::KSIntCalculatorMott() : fThetaMin(0.), fThetaMax(0.), fNucleus("") {}
KSIntCalculatorMott::KSIntCalculatorMott(const KSIntCalculatorMott& aCopy) :
    KSComponent(aCopy),
    fThetaMin(aCopy.fThetaMin),
    fThetaMax(aCopy.fThetaMax),
    fNucleus(aCopy.fNucleus)
{}
KSIntCalculatorMott* KSIntCalculatorMott::Clone() const
{
    return new KSIntCalculatorMott(*this);
}
KSIntCalculatorMott::~KSIntCalculatorMott() = default;

void KSIntCalculatorMott::CalculateCrossSection(const KSParticle& anInitialParticle, double& aCrossSection)
{

    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();

    double fCrossSection = (MTCS(fThetaMax, tInitialKineticEnergy) - MTCS(fThetaMin, tInitialKineticEnergy));

    aCrossSection = fCrossSection;
    return;
}
void KSIntCalculatorMott::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                 KSParticleQueue& /*aSecondaries*/)
{
    double tTime = anInitialParticle.GetTime();
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tPosition = anInitialParticle.GetPosition();
    KThreeVector tMomentum = anInitialParticle.GetMomentum();

    double tTheta = GetTheta(tInitialKineticEnergy);
    double tLostKineticEnergy = GetEnergyLoss(tInitialKineticEnergy, tTheta);


    double tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
    tMomentum.SetPolarAngle(tTheta * katrin::KConst::Pi() / 180);
    tMomentum.SetAzimuthalAngle(tPhi);

    KThreeVector tOrthogonalOne = tMomentum.Orthogonal();
    KThreeVector tOrthogonalTwo = tMomentum.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tMomentum.Magnitude() *
        (sin(tTheta* katrin::KConst::Pi() / 180) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta* katrin::KConst::Pi() / 180) * tMomentum.Unit());


    aFinalParticle.SetTime(tTime);
    aFinalParticle.SetPosition(tPosition);
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.SetLabel(GetName());

    fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;
    fStepAngularChange = tTheta;

    return;
}

double KSIntCalculatorMott::GetTheta(const double& anEnergy){

    double tTheta;
    double resolution;
    double resolution_factor = 10; // arbitrarily chosen for o(1ms) computation time for He; computation time increases linearly with resolution factor; resolution factors under 1 yields incorrect results


    resolution = ceil((fThetaMax - fThetaMin) * resolution_factor);


    // Initializing array for possible theta values
    std::vector<double> theta_arr;
    for (int i = 0; i <= int(resolution); ++i) {
        theta_arr.push_back(fThetaMin + i * (fThetaMax - fThetaMin) / resolution);
    }

    // Calculating rescaling factor for rejection sampling
    std::vector<double> k_values;
    for (double theta : theta_arr) {
        k_values.push_back(MDCS(theta, anEnergy) / Normalized_RDCS(theta));
    }

    // Find the maximum value of k
    double k = *std::max_element(k_values.begin(), k_values.end());



    // Rejection sampling taking initial sample from Rutherford Diff. X-Sec
    while (true) {
        double sample = Normalized_RTCSInverse(KRandom::GetInstance().Uniform(0.0, 1.0, false, true));
        double uniform_sample = KRandom::GetInstance().Uniform(0.0, 1.0, false, true);
        if (MDCS(sample, anEnergy) / (k * Normalized_RDCS(sample)) > uniform_sample) {
            tTheta = sample;
            break;
        }
    }

    return tTheta;

}

double KSIntCalculatorMott::GetEnergyLoss(const double& anEnergy, const double& aTheta){
    double M, me, p, anELoss;

    double amu = 0.0;

    if (fNucleus == "He") {
        amu = 4.0026; // mass in amu from W.J. Huang et al 2021 Chinese Phys. C 45 030002
    } else if (fNucleus == "Ne") {
        amu = 19.9924; // mass in amu from W.J. Huang et al 2021 Chinese Phys. C 45 030002
    } 

    M = amu * katrin::KConst::AtomicMassUnit_eV(); //  eV/c^2
    me = katrin::KConst::M_el_eV(); // electron mass eV/c^2 

    p = sqrt(anEnergy * (anEnergy + 2 * me * pow(katrin::KConst::C(), 2))) / katrin::KConst::C();

    anELoss = 2 * pow(p, 2) * M / (pow(me, 2) + pow(M, 2) + 2 * M * sqrt( pow((p/katrin::KConst::C()), 2) + pow(me, 2))) * (1 - cos(aTheta * katrin::KConst::Pi() / 180));

    return anELoss;
}


double KSIntCalculatorMott::Beta(double const E0) {
    return sqrt( pow(E0, 2) + 2 * E0 * katrin::KConst::M_el_eV()) / (E0 + katrin::KConst::M_el_eV());
}

std::vector<double> KSIntCalculatorMott::RMott_coeffs(double const E0) {

    std::vector<double> a(6, 0.0); // Initialize a with 6 zeros, the last entry is not related to coefficient calculation it is Z for the nucleus of choice
    std::vector<std::vector<double>> b(5, std::vector<double>(6));

    if (fNucleus == "He") {
        a[5] = 2; // Charge Z
        b = {{ 1.0       ,  3.76476e-8, -3.05313e-7, -3.27422e-7,  2.44235e-6,  4.08754e-6},
             { 2.35767e-2,  3.24642e-2, -6.37269e-4, -7.69160e-4,  5.28004e-3,  9.45642e-3},
             {-2.73743e-1, -7.40767e-1, -4.98195e-1,  1.74337e-3, -1.25798e-2, -2.24046e-2},
             {-7.79128e-4, -4.14495e-4, -1.62657e-3, -1.37286e-3,  1.04319e-2,  1.83488e-2},
             { 2.02855e-4,  1.94598e-6,  4.30102e-4,  4.32180e-4, -3.31526e-3, -5.81788e-3}};
    } else if (fNucleus == "Ne") {
        a[5] = 10; // Charge Z
        b = {{ 9.99997e-1, -1.87404e-7,  3.10276e-5,  5.20000e-5,  2.98132e-4, -5.19259e-4},
             { 1.20783e-1,  1.66407e-1,  1.06608e-2,  6.48772e-3, -1.53031e-3, -7.59354e-2},
             {-3.15222e-1, -8.28793e-1, -6.05740e-1, -1.47812e-1,  1.15760   ,  1.58565   },
             {-2.89055e-2, -9.08096e-4,  1.21467e-1, -1.77575e-1, -1.29110   , -1.90333   },
             { 7.65342e-3, -8.85417e-4, -3.89092e-2, -5.44040e-2,  3.93087e-1,  5.79439e-1}};
    } 

    for (int j = 0; j < 5; ++j) {
        a[j] = 0.0;
        for (int k = 0; k < 6; ++k) {
            a[j] += b[j][k] * pow(Beta(E0) - 0.7181287, k);
        }
    }

    return a;
}

double KSIntCalculatorMott::MDCS(double theta, const double E0) {

    double r_e = 2.8179403227 * pow(10, -15); // classical electron radius

    std::vector<double> a = RMott_coeffs(E0);

    double Z = a[5];

    return  pow(Z, 2) * pow(r_e, 2) * ( (1 - pow(Beta(E0), 2))/(pow(Beta(E0), 4)) ) * ((a[0] * pow(1 - cos(theta * katrin::KConst::Pi() / 180), -2)) + (a[1] * pow(sqrt(1 - cos(theta * katrin::KConst::Pi() / 180)), -3)) + (a[2] * pow((1 - cos(theta * katrin::KConst::Pi() / 180)), (-1))) + (a[3] * pow(sqrt(1 - cos(theta * katrin::KConst::Pi() / 180)), -1)) + a[4] );
}

double KSIntCalculatorMott::MTCS(double theta, const double E0) {

    double r_e = 2.8179403227 * pow(10, -15); // classical electron radius

    std::vector<double> a = RMott_coeffs(E0);

    double Z = a[5]; 


    return 2 * katrin::KConst::Pi() * pow(Z, 2) * pow(r_e, 2) * 
                    ((1 - pow(Beta(E0), 2)) / pow(Beta(E0), 4)) *
                    ((a[0] * pow((1 - cos(theta * katrin::KConst::Pi() / 180)), -1) / (-1)) + 
                     (a[1] * pow((1 - cos(theta * katrin::KConst::Pi() / 180)), (1.0 / 2) - 1) / ((1.0 / 2) - 1)) + 
                     (a[2] * log(1 - cos(theta * katrin::KConst::Pi() / 180))) + 
                     (a[3] * pow((1 - cos(theta * katrin::KConst::Pi() / 180)), (3.0 / 2) - 1) / ((3.0 / 2) - 1)) + 
                     (a[4] * -1 * cos(theta * katrin::KConst::Pi() / 180)))
            - 2 * katrin::KConst::Pi() * pow(Z, 2) * pow(r_e, 2) * 
                    ((1 - pow(Beta(E0), 2)) / pow(Beta(E0), 4)) *
                    ((a[0] * pow((1 - cos(fThetaMin * katrin::KConst::Pi() / 180)), -1) / (-1)) + 
                     (a[1] * pow((1 - cos(fThetaMin * katrin::KConst::Pi() / 180)), (1.0 / 2) - 1) / ((1.0 / 2) - 1)) + 
                     (a[2] * log(1 - cos(fThetaMin * katrin::KConst::Pi() / 180))) + 
                     (a[3] * pow((1 - cos(fThetaMin * katrin::KConst::Pi() / 180)), (3.0 / 2) - 1) / ((3.0 / 2) - 1)) + 
                     (a[4] * -1 * cos(fThetaMin * katrin::KConst::Pi() / 180)));
}

double KSIntCalculatorMott::Normalized_RDCS(double theta) {

    double C = 1 / ((1 / (1 - cos(fThetaMin* katrin::KConst::Pi() / 180))) - (1 / (1 - cos(fThetaMax* katrin::KConst::Pi() / 180))));

    return C / pow((1 - cos(theta* katrin::KConst::Pi() / 180)), 2);
}

double KSIntCalculatorMott::Normalized_RTCSInverse(double x) {

    double C = 1 / ((1 / (1 - cos(fThetaMin * katrin::KConst::Pi() / 180))) - (1 / (1 - cos(fThetaMax * katrin::KConst::Pi() / 180))));

    return acos( 1 - (1 / ((1 / (1 - cos(fThetaMin * katrin::KConst::Pi() / 180))) - x / C)) ) * 180 / katrin::KConst::Pi();
}

}  // namespace Kassiopeia
