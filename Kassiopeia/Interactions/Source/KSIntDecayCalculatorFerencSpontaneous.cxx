//
// Created by trost on 27.05.15.
//

#include "KSIntDecayCalculatorFerencSpontaneous.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSInteractionsMessage.h"
#include "RydbergFerenc.h"

using katrin::KRandom;


namespace Kassiopeia
{

KSIntDecayCalculatorFerencSpontaneous::KSIntDecayCalculatorFerencSpontaneous() :
    fTargetPID(0),
    fminPID(0),
    fmaxPID(0),
    fLastn(-1),
    fLastl(-1),
    fLastLifetime(std::numeric_limits<double>::max())
{
    fCalculator = new RydbergCalculator();
}

KSIntDecayCalculatorFerencSpontaneous::KSIntDecayCalculatorFerencSpontaneous(
    const KSIntDecayCalculatorFerencSpontaneous& aCopy) :
    KSComponent(aCopy),
    fTargetPID(aCopy.fTargetPID),
    fminPID(aCopy.fminPID),
    fmaxPID(aCopy.fmaxPID),
    fLastn(aCopy.fLastn),
    fLastl(aCopy.fLastl),
    fLastLifetime(aCopy.fLastLifetime)
{
    fCalculator = new RydbergCalculator();
    for (int n = 0; n < 150; n++) {
        for (int l = 0; l < n; l++) {
            low_n_lifetimes[n][l] = aCopy.low_n_lifetimes[n][l];
        }
    }
}

KSIntDecayCalculatorFerencSpontaneous* KSIntDecayCalculatorFerencSpontaneous::Clone() const
{
    return new KSIntDecayCalculatorFerencSpontaneous(*this);
}

void KSIntDecayCalculatorFerencSpontaneous::InitializeComponent()
{

    for (int n = 0; n < 150; n++) {
        for (int l = 0; l < n + 1; l++) {
            low_n_lifetimes[n][l] = 1. / (fCalculator->Pspsum(n + 1, l));
        }
    }
}

KSIntDecayCalculatorFerencSpontaneous::~KSIntDecayCalculatorFerencSpontaneous()
{
    delete fCalculator;
}

void KSIntDecayCalculatorFerencSpontaneous::CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime)
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

        fLastLifetime = 1. / (fCalculator->Pspsum(n, l));
        fLastn = n;
        fLastl = l;

        aLifeTime = 1. / (fCalculator->Pspsum(n, l));
    }
    else {
        aLifeTime = std::numeric_limits<double>::max();
    }
    return;
}

void KSIntDecayCalculatorFerencSpontaneous::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                               KSParticle& aFinalParticle,
                                                               KSParticleQueue& /*aSecondaries*/)
{
    aFinalParticle.SetTime(anInitialParticle.GetTime());
    aFinalParticle.SetPosition(anInitialParticle.GetPosition());
    aFinalParticle.SetMomentum(anInitialParticle.GetMomentum());

    long long tPID = anInitialParticle.GetPID();

    if ((tPID == fTargetPID && fTargetPID != 0) || ((tPID >= fminPID) && (tPID <= fmaxPID))) {
        int n = anInitialParticle.GetMainQuantumNumber();
        int l = anInitialParticle.GetSecondQuantumNumber();
        int np = n;
        int lp = l;
        double tRate;

        fCalculator->SpontaneousEmissionGenerator(n, l, tRate, np, lp);

        aFinalParticle.SetMainQuantumNumber(np);
        aFinalParticle.SetSecondQuantumNumber(lp);

        if (np == 1) {
            aFinalParticle.SetLabel(GetName());
            aFinalParticle.SetActive(false);
        }

        fStepNDecays = 1;
        fStepEnergyLoss = 0.;
    }

    return;
}

}  // namespace Kassiopeia
