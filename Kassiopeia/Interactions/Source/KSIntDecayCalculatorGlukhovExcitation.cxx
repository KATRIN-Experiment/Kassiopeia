#include "KSIntDecayCalculatorGlukhovExcitation.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSIntDecayCalculatorGlukhovExcitation::KSIntDecayCalculatorGlukhovExcitation() : fTargetPID(0), fminPID(0), fmaxPID(0)
{}

KSIntDecayCalculatorGlukhovExcitation::KSIntDecayCalculatorGlukhovExcitation(
    const KSIntDecayCalculatorGlukhovExcitation& aCopy) :
    KSComponent(),
    fTargetPID(aCopy.fTargetPID),
    fminPID(aCopy.fminPID),
    fmaxPID(aCopy.fmaxPID)
{}

KSIntDecayCalculatorGlukhovExcitation* KSIntDecayCalculatorGlukhovExcitation::Clone() const
{
    return new KSIntDecayCalculatorGlukhovExcitation(*this);
}

KSIntDecayCalculatorGlukhovExcitation::~KSIntDecayCalculatorGlukhovExcitation() {}

const double KSIntDecayCalculatorGlukhovExcitation::p_coefficients[3][4] = {{5.8664e8, -0.3634, -16.704, 51.07},
                                                                            {5.4448e9, -0.03953, -1.5171, 5.115},
                                                                            {1.9153e9, -0.11334, -3.1140, 12.913}};

const double KSIntDecayCalculatorGlukhovExcitation::T_a_tilde = 0.31578;

const double KSIntDecayCalculatorGlukhovExcitation::b_ex[3][3][3] = {
    {{14.557, 0.7418, -0.0437}, {-8.8432, 5.3981, -2.6255}, {10.9688, -19.720, 9.0861}},

    {{1.5675, 0.0791, -0.0045}, {-1.0648, 0.7484, -0.3610}, {0.9096, -1.5654, 0.6850}},

    {{4.4430, 0.2415, -0.0212}, {-2.4783, 1.2168, -0.5731}, {2.9479, -5.1645, 2.2992}}};


void KSIntDecayCalculatorGlukhovExcitation::CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime)
{
    long long tPID = aParticle.GetPID();
    if ((tPID == fTargetPID && fTargetPID != 0) || ((tPID >= fminPID) && (tPID <= fmaxPID))) {
        int n = aParticle.GetMainQuantumNumber();
        // int l = aParticle.GetSecondQuantumNumber();

        aLifeTime = 1. / (CalculateSpontaneousDecayRate(n, 1) * CalculateRelativeExcitationRate(n, 1, fTemperature));
    }
    else {
        aLifeTime = std::numeric_limits<double>::max();
    }
    return;
}

void KSIntDecayCalculatorGlukhovExcitation::ExecuteInteraction(const KSParticle& anInitialParticle,
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
        aFinalParticle.SetMainQuantumNumber(n + 1);

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

    return;
}

double KSIntDecayCalculatorGlukhovExcitation::CalculateSpontaneousDecayRate(int n, int l)
{
    if (l > 2 || l < 0 || n < 0)
        intmsg(eError) << GetName() << ": Only l states 0,1,2 available for Rydberg decay" << eom;

    return p_coefficients[l][0] / (n * n * n) *
           (1. + p_coefficients[l][1] / n + p_coefficients[l][2] / (n * n) + p_coefficients[l][3] / (n * n * n));
}

double KSIntDecayCalculatorGlukhovExcitation::tau(double T)
{
    return std::pow(100.0 / T, 1. / 3.);
}

double KSIntDecayCalculatorGlukhovExcitation::a_ex(int l, int i, double T)
{
    double tTau = tau(T);
    return b_ex[l][i][0] + b_ex[l][i][1] * tTau + b_ex[l][i][2] * tTau * tTau;
}

double KSIntDecayCalculatorGlukhovExcitation::x(int n, double T)
{
    return 100. / (n * std::pow(T, 1. / 3.));
}

double KSIntDecayCalculatorGlukhovExcitation::CalculateRelativeExcitationRate(int n, int l, double T)
{
    if (l > 2 || l < 0 || n < 0)
        intmsg(eError) << "KSIntDecayCalculatorGlukhovExcitation: Only l states 0,1,2 available for Rydberg decay"
                       << eom;

    return (a_ex(l, 0, T) + a_ex(l, 1, T) * x(n, T) + a_ex(l, 2, T) * x(n, T) * x(n, T))

           / (n * n * (std::exp(T_a_tilde * 1000000 / (n * n * n * T)) - 1));
}


}  // namespace Kassiopeia
