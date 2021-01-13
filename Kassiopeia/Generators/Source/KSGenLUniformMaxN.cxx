//
// Created by Nikolaus Trost on 07.05.15.
//

#include "KSGenLUniformMaxN.h"

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{
KSGenLUniformMaxN::KSGenLUniformMaxN() = default;

KSGenLUniformMaxN::KSGenLUniformMaxN(const KSGenLUniformMaxN& /*aCopy*/) : KSComponent() {}

KSGenLUniformMaxN* KSGenLUniformMaxN::Clone() const
{
    return new KSGenLUniformMaxN(*this);
}

KSGenLUniformMaxN::~KSGenLUniformMaxN() = default;

void KSGenLUniformMaxN::InitializeComponent() {}

void KSGenLUniformMaxN::DeinitializeComponent() {}

void KSGenLUniformMaxN::Dice(KSParticleQueue* aPrimaries)
{

    for (auto& aPrimarie : *aPrimaries) {
        aPrimarie->SetSecondQuantumNumber(KRandom::GetInstance().Uniform(0, aPrimarie->GetMainQuantumNumber() - 1));
    }
}
}  // namespace Kassiopeia