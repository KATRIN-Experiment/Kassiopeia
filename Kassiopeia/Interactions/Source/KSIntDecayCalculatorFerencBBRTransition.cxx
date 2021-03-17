//
// Created by trost on 27.05.15.
//
#include "KSIntDecayCalculatorFerencBBRTransition.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSIntDecayCalculatorFerencBBRTransition::KSIntDecayCalculatorFerencBBRTransition() :
    fTargetPID(0),
    fminPID(0),
    fmaxPID(0),
    fLastn(-1),
    fLastl(-1),
    fLastLifetime(std::numeric_limits<double>::max())
{
    fCalculator = new RydbergCalculator();
}

KSIntDecayCalculatorFerencBBRTransition::KSIntDecayCalculatorFerencBBRTransition(
    const KSIntDecayCalculatorFerencBBRTransition& aCopy) :
    KSComponent(aCopy),
    fTargetPID(aCopy.fTargetPID),
    fminPID(aCopy.fminPID),
    fmaxPID(aCopy.fmaxPID),
    fLastn(aCopy.fLastn),
    fLastl(aCopy.fLastl),
    fLastLifetime(aCopy.fLastLifetime)
{
    fCalculator = new RydbergCalculator();
    for (int n = 0; n <= 150; n++) {
        for (int l = 0; l < n; l++) {
            low_n_lifetimes[n][l] = aCopy.low_n_lifetimes[n][l];
        }
    }
}


void KSIntDecayCalculatorFerencBBRTransition::InitializeComponent()
{

    for (int n = 0; n < 150; n++) {
        for (int l = 0; l < n + 1; l++) {
            int npmax = 10 * (n + 1);
            if ((n + 1) > 80)
                npmax = 5 * (n + 1);

            low_n_lifetimes[n][l] = 1. / (fCalculator->PBBRdecay(fTemperature, n + 1, l) +
                                          fCalculator->PBBRexcitation(fTemperature, n + 1, l, npmax));
        }
    }
}

KSIntDecayCalculatorFerencBBRTransition* KSIntDecayCalculatorFerencBBRTransition::Clone() const
{
    return new KSIntDecayCalculatorFerencBBRTransition(*this);
}

KSIntDecayCalculatorFerencBBRTransition::~KSIntDecayCalculatorFerencBBRTransition()
{
    delete fCalculator;
}

void KSIntDecayCalculatorFerencBBRTransition::CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime)
{
    long long tPID = aParticle.GetPID();
    if ((tPID == fTargetPID && fTargetPID != 0) || ((tPID >= fminPID) && (tPID <= fmaxPID))) {
        int n = aParticle.GetMainQuantumNumber();
        int l = aParticle.GetSecondQuantumNumber();

        if ((n == fLastn) && (l == fLastl)) {
            aLifeTime = fLastLifetime;
            return;
        }

        if (n <= 150) {
            fLastLifetime = low_n_lifetimes[n - 1][l];
            aLifeTime = fLastLifetime;
            fLastn = n;
            fLastl = l;
            return;
        }


        int npmax = 10 * n;
        if (n > 80)
            npmax = 5 * n;

        if (npmax > 8000)
            npmax = 8000;

        fLastLifetime =
            1. / (fCalculator->PBBRdecay(fTemperature, n, l) + fCalculator->PBBRexcitation(fTemperature, n, l, npmax));
        fLastn = n;
        fLastl = l;

        aLifeTime = fLastLifetime;
    }
    else {
        aLifeTime = std::numeric_limits<double>::max();
    }
    return;
}

void KSIntDecayCalculatorFerencBBRTransition::ExecuteInteraction(const KSParticle& anInitialParticle,
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
        int np = n;
        int lp = l;
        double tRate;

        fCalculator->BBRTransitionGenerator(fTemperature, n, l, tRate, np, lp);

        aFinalParticle.SetMainQuantumNumber(np);
        aFinalParticle.SetSecondQuantumNumber(lp);

        if (np == 1) {
            aFinalParticle.SetActive(false);
            aFinalParticle.SetLabel(GetName());
        }

        fStepNDecays = 1;
        fStepEnergyLoss = 0.;
    }

    return;
}

}  // namespace Kassiopeia
