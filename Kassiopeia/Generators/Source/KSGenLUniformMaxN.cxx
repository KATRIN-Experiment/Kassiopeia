//
// Created by Nikolaus Trost on 07.05.15.
//

#include "KSGenLUniformMaxN.h"
#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia {
    KSGenLUniformMaxN::KSGenLUniformMaxN() { }

    KSGenLUniformMaxN::KSGenLUniformMaxN(const KSGenLUniformMaxN& /*aCopy*/) : KSComponent() { }

    KSGenLUniformMaxN *KSGenLUniformMaxN::Clone() const {
        return new KSGenLUniformMaxN(*this);
    }

    KSGenLUniformMaxN::~KSGenLUniformMaxN() { }

    void KSGenLUniformMaxN::InitializeComponent() { }

    void KSGenLUniformMaxN::DeinitializeComponent() { }

    void KSGenLUniformMaxN::Dice(KSParticleQueue* aPrimaries) {

        for(KSParticleIt p = aPrimaries->begin(); p != aPrimaries->end(); ++p) {
            (*p)->SetSecondQuantumNumber(KRandom::GetInstance().Uniform(0,(*p)->GetMainQuantumNumber()-1) );
        }

    }
}