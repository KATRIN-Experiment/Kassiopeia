#include "KSIntCalculatorMott.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"

using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSIntCalculatorMott::KSIntCalculatorMott() : fThetaMin(0.), fThetaMax(0.), fMDCS(nullptr), fMTCS(nullptr) {}
KSIntCalculatorMott::KSIntCalculatorMott(const KSIntCalculatorMott& aCopy) :
    KSComponent(aCopy),
    fThetaMin(aCopy.fThetaMin),
    fThetaMax(aCopy.fThetaMax),
    fMDCS(nullptr),
    fMTCS(nullptr)
{}
KSIntCalculatorMott* KSIntCalculatorMott::Clone() const
{
    return new KSIntCalculatorMott(*this);
}
KSIntCalculatorMott::~KSIntCalculatorMott() = default;

void KSIntCalculatorMott::CalculateCrossSection(const KSParticle& anInitialParticle, double& aCrossSection)
{

    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    InitializeMDCS(tInitialKineticEnergy);

    double fCrossSection = (fMTCS->Eval(fThetaMax) - fMTCS->Eval(fThetaMin)) * pow(10, -30);

    DeinitializeMDCS();

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
    tMomentum.SetPolarAngle(tTheta);
    tMomentum.SetAzimuthalAngle(tPhi);

    aFinalParticle.SetTime(tTime);
    aFinalParticle.SetPosition(tPosition);
    aFinalParticle.SetMomentum(tMomentum);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.SetLabel(GetName());

    fStepAngularChange = tTheta * 180. / katrin::KConst::Pi();

    return;
}

double KSIntCalculatorMott::GetTheta(const double& anEnergy){

    double tTheta;

    InitializeMDCS(anEnergy);

    tTheta = fMDCS->GetRandom(fThetaMin, fThetaMax);

    DeinitializeMDCS();

    return tTheta;

}

double KSIntCalculatorMott::GetEnergyLoss(const double& anEnergy, const double& aTheta){
    double M, me, p, anELoss;

    M = 4.0026 * katrin::KConst::AtomicMassUnit_eV(); //  eV/c^2
    me = katrin::KConst::M_el_eV(); // electron mass eV/c^2 

    p = sqrt(anEnergy * (anEnergy + 2 * me * pow(katrin::KConst::C(), 2))) / katrin::KConst::C();

    anELoss = 2 * pow(p, 2) * M / (pow(me, 2) + pow(M, 2) + 2 * M * sqrt( pow((p/katrin::KConst::C()), 2) + pow(me, 2))) * (1 - cos(aTheta));

    return anELoss;
}

void KSIntCalculatorMott::InitializeMDCS(const double E0)
{ 

    double beta = TMath::Sqrt( pow(E0, 2) + 2 * E0 * katrin::KConst::M_el_eV()) / (E0 + katrin::KConst::M_el_eV());

    /* Constants given in publication in header file. These are the constants for scattering off of Helium */
    static double b[5][6] = {{  1.                   ,  3.76476 * pow(10, -8), -3.05313 * pow(10, -7), -3.27422 * pow(10, -7),  2.44235 * pow(10, -6),  4.08754 * pow(10, -6)}, 
                             {  2.35767 * pow(10, -2),  3.24642 * pow(10, -2), -6.37269 * pow(10, -4), -7.69160 * pow(10, -4),  5.28004 * pow(10, -3),  9.45642 * pow(10, -3)}, 
                             { -2.73743 * pow(10, -1), -7.40767 * pow(10, -1), -4.98195 * pow(10, -1),  1.74337 * pow(10, -3), -1.25798 * pow(10, -2), -2.24046 * pow(10, -2)}, 
                             { -7.79128 * pow(10, -4), -4.14495 * pow(10, -4), -1.62657 * pow(10, -3), -1.37286 * pow(10, -3),  1.04319 * pow(10, -2),  1.83488 * pow(10, -2)}, 
                             {  2.02855 * pow(10, -4),  1.94598 * pow(10, -6),  4.30102 * pow(10, -4),  4.32180 * pow(10, -4), -3.31526 * pow(10, -3), -5.81788 * pow(10, -3)}};

    double a[6] = {0, 0, 0, 0, 0, 0};

    for (int j = 0; j < 5; j++){
        for (int k = 0; k < 6; k++){
            a[j] += b[j][k] * pow((beta - 0.7181287), k);
        }
    }
    
    /* ROOT TF1 takes in analytical formulas as strings. This is the conversion into strings */
    std::string Rmott;
    std::ostringstream RmottStream;

    RmottStream << std::fixed;
    RmottStream << std::setprecision(12);

    RmottStream << a[0] << " + " << a[1] << " * pow( ( 1 - cos(x) ) , 0.5) + " << a[2] << " * ( 1 - cos(x) ) + " << a[3] << " * pow( ( 1 - cos(x) ) , 1.5) + " << a[4] << " * pow( ( 1 - cos(x) ) , 2)";
    Rmott = RmottStream.str();



    std::string RutherfordDCS;
    std::ostringstream RutherfordDCSStream;

    RutherfordDCSStream << std::fixed;
    RutherfordDCSStream << std::setprecision(12);

    RutherfordDCSStream << "pow((2 * 2.8179403227 ), 2) * ( (1 - pow(" << beta << ", 2) ) / ( pow(" << beta << ", 4) )) * (1 / pow( (1 - cos(x)), 2))";
    RutherfordDCS = RutherfordDCSStream.str();



    std::string MottDCS = "(" + RutherfordDCS + ") * (" + Rmott + ")";


    std::string MottTCS;
    std::ostringstream MottTCSStream;
    std::ostringstream MottTCSStreamThetaIndepPrefactor;

    MottTCSStream << std::fixed;
    MottTCSStream << std::setprecision(12);
    
    MottTCSStreamThetaIndepPrefactor << std::fixed;
    MottTCSStreamThetaIndepPrefactor << std::setprecision(12);

    MottTCSStreamThetaIndepPrefactor << "pow((2 * 2.8179403227 ), 2) * ( (1 - pow(" << beta << ", 2) ) / ( pow(" << beta << ", 4) )) * ";

    MottTCSStream << "(2 * " << katrin::KConst::Pi() << " * (-4 * sqrt(2) * ( (" << a[1] << ") + (-1) * (" << a[2]
                    << ") * sqrt(( 1 - cos(x) )) * log( sin(x / 2) )) * pow(sin(x / 2), 4) + 4 * sqrt(2) * (2 * ("
                    << a[3] << ") + (" << a[4] << ") * sqrt( (1 - cos(x))) ) * pow(sin(x / 2), 6) - 2 * (" << a[0]
                    << ") * pow(sin( x / 2 ), 3) ) ) / ( pow((-1 + cos(x)), 2) * sin( x / 2 ))"; 

    MottTCS = MottTCSStreamThetaIndepPrefactor.str() + MottTCSStream.str();

    fMDCS = new TF1("function", MottDCS.c_str(), fThetaMin, fThetaMax); 
    fMTCS = new TF1("function", MottTCS.c_str(), fThetaMin, fThetaMax);

    return;
}

void KSIntCalculatorMott::DeinitializeMDCS()
{
    delete fMDCS;
    fMDCS = nullptr;
    delete fMTCS;
    fMTCS = nullptr;
    return;
}

}  // namespace Kassiopeia
