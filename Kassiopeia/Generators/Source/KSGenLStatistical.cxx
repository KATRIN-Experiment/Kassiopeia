//
// Created by Nikolaus Trost on 08.05.15.
//

#include "KSGenLStatistical.h"

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{
KSGenLStatistical::KSGenLStatistical() {}

KSGenLStatistical::KSGenLStatistical(const KSGenLStatistical& /*aCopy*/) : KSComponent() {}

KSGenLStatistical* KSGenLStatistical::Clone() const
{
    return new KSGenLStatistical(*this);
}

KSGenLStatistical::~KSGenLStatistical() {}

void KSGenLStatistical::InitializeComponent() {}

void KSGenLStatistical::DeinitializeComponent() {}

void KSGenLStatistical::Dice(KSParticleQueue* aPrimaries)
{

    for (auto p = aPrimaries->begin(); p != aPrimaries->end(); ++p) {
        int n = (*p)->GetMainQuantumNumber();
        int l = std::floor(std::sqrt(KRandom::GetInstance().Uniform(1, n * n)) - 1);

        (*p)->SetSecondQuantumNumber(l);
    }
}
}  // namespace Kassiopeia