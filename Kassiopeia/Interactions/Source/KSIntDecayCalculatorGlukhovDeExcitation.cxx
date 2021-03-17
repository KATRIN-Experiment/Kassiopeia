#include "KSIntDecayCalculatorGlukhovDeExcitation.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSIntDecayCalculatorGlukhovDeExcitation::KSIntDecayCalculatorGlukhovDeExcitation() :
    fTargetPID(0),
    fminPID(0),
    fmaxPID(0)
{}

KSIntDecayCalculatorGlukhovDeExcitation::KSIntDecayCalculatorGlukhovDeExcitation(
    const KSIntDecayCalculatorGlukhovDeExcitation& aCopy) :
    KSComponent(aCopy),
    fTargetPID(aCopy.fTargetPID),
    fminPID(aCopy.fminPID),
    fmaxPID(aCopy.fmaxPID)
{}

KSIntDecayCalculatorGlukhovDeExcitation* KSIntDecayCalculatorGlukhovDeExcitation::Clone() const
{
    return new KSIntDecayCalculatorGlukhovDeExcitation(*this);
}

KSIntDecayCalculatorGlukhovDeExcitation::~KSIntDecayCalculatorGlukhovDeExcitation() = default;

const double KSIntDecayCalculatorGlukhovDeExcitation::p_coefficients[3][4] = {{5.8664e8, -0.3634, -16.704, 51.07},
                                                                              {5.4448e9, -0.03953, -1.5171, 5.115},
                                                                              {1.9153e9, -0.11334, -3.1140, 12.913}};
const double KSIntDecayCalculatorGlukhovDeExcitation::T_a_tilde = 0.31578;

const double KSIntDecayCalculatorGlukhovDeExcitation::b_dex[3][3][3] = {
    {{17.287, 1.1094, -0.4820}, {-24.886, 13.505, -5.6676}, {11.780, -10.330, 4.1794}},

    {{1.8683, 0.1071, -0.0463}, {-2.7344, 1.5025, -0.6243}, {1.2669, -1.1082, 0.4502}},

    {{5.3091, 0.3091, -0.1331}, {-6.2159, 1.3430, -0.2841}, {3.0276, -2.0534, 0.7185}}};

void KSIntDecayCalculatorGlukhovDeExcitation::CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime)
{
    long long tPID = aParticle.GetPID();
    if ((tPID == fTargetPID && fTargetPID != 0) || ((tPID >= fminPID) && (tPID <= fmaxPID))) {
        int n = aParticle.GetMainQuantumNumber();
        // int l = aParticle.GetSecondQuantumNumber();

        aLifeTime = 1. / (CalculateSpontaneousDecayRate(n, 1) * CalculateRelativeDeExcitationRate(n, 1, fTemperature));
    }
    else {
        aLifeTime = std::numeric_limits<double>::max();
    }
    return;
}

void KSIntDecayCalculatorGlukhovDeExcitation::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                                 KSParticle& aFinalParticle,
                                                                 KSParticleQueue& /*aSecondaries*/)
{
    aFinalParticle.SetTime(anInitialParticle.GetTime());
    aFinalParticle.SetPosition(anInitialParticle.GetPosition());
    aFinalParticle.SetMomentum(anInitialParticle.GetMomentum());

    if ((anInitialParticle.GetPID() == fTargetPID && fTargetPID != 0) ||
        ((anInitialParticle.GetPID() >= fminPID) && (anInitialParticle.GetPID() <= fmaxPID))) {
        int n = anInitialParticle.GetMainQuantumNumber();
        int l = anInitialParticle.GetSecondQuantumNumber();

        aFinalParticle.SetLabel(GetName());
        if (n > 9)
            aFinalParticle.SetMainQuantumNumber(n - 1);

        if (l == 0)
            aFinalParticle.SetSecondQuantumNumber(l + 1);
        else if (l == n - 1)
            aFinalParticle.SetSecondQuantumNumber(l - 1);
        else if (KRandom::GetInstance().Bool(0.5))
            aFinalParticle.SetSecondQuantumNumber(l - 1);
        else
            aFinalParticle.SetSecondQuantumNumber(l + 1);

        fStepNDecays = 1;
        fStepEnergyLoss = 0.;
    }
}

double KSIntDecayCalculatorGlukhovDeExcitation::CalculateSpontaneousDecayRate(int n, int l)
{
    if (l > 2 || l < 0 || n < 0)
        intmsg(eError) << GetName() << ": Only l states 0,1,2 available for Rydberg decay" << eom;

    return p_coefficients[l][0] / (n * n * n) *
           (1. + p_coefficients[l][1] / n + p_coefficients[l][2] / (n * n) + p_coefficients[l][3] / (n * n * n));
}

double KSIntDecayCalculatorGlukhovDeExcitation::tau(double T)
{
    return std::pow(100.0 / T, 1. / 3.);
}

double KSIntDecayCalculatorGlukhovDeExcitation::a_dex(int l, int i, double T)
{
    double tTau = tau(T);
    return b_dex[l][i][0] + b_dex[l][i][1] * tTau + b_dex[l][i][2] * tTau * tTau;
}

double KSIntDecayCalculatorGlukhovDeExcitation::x(int n, double T)
{
    return 100. / (n * std::pow(T, 1. / 3.));
}

double KSIntDecayCalculatorGlukhovDeExcitation::CalculateRelativeDeExcitationRate(int n, int l, double T)
{
    if (l > 2 || l < 0 || n < 0)
        intmsg(eError) << "KSIntDecayCalculatorGlukhovDeExcitation: Only l states 0,1,2 available for Rydberg decay"
                       << eom;

    return (a_dex(l, 0, T) + a_dex(l, 1, T) * x(n, T) + a_dex(l, 2, T) * x(n, T) * x(n, T))

           / (n * n * (std::exp(T_a_tilde * 1000000 / (n * n * n * T)) - 1));
}


}  // namespace Kassiopeia
