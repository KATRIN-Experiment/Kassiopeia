//
// Created by Nikolaus Trost on 07.05.15.
//

#ifndef KASPER_KSGENVALUELUNIFORMMAXN_H
#define KASPER_KSGENVALUELUNIFORMMAXN_H

#include "KSGeneratorsMessage.h"
#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
    /**
     * \brief Dices angular momentum of particles.
     */
    class KSGenLUniformMaxN :
            public KSComponentTemplate<KSGenLUniformMaxN, KSGenCreator>
    {
    public:
        KSGenLUniformMaxN();
        KSGenLUniformMaxN(const KSGenLUniformMaxN&);
        KSGenLUniformMaxN* Clone() const;
        virtual ~KSGenLUniformMaxN();

    public:
        /**
         * \brief Dices the quantized angular Momentum L of all particles of
         * the KSParticleQueue equally distributed between 0 and the particles n-1
         *
         *
         * \param aPrimaries
         */
        virtual void Dice(KSParticleQueue* aPrimaries);


    public:
        void InitializeComponent();
        void DeinitializeComponent();
    };
}


#endif //KASPER_KSGENVALUELUNIFORMMAXN_H
