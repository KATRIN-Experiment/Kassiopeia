//
// Created by Nikolaus Trost on 08.05.15.
//

#ifndef KASPER_KSGENLSTATISTICAL_H
#define KASPER_KSGENLSTATISTICAL_H

#include "KSGeneratorsMessage.h"
#include "KSGenCreator.h"

namespace Kassiopeia
{
    /**
     * \brief Dices angular momentum of particles.
     */
    class KSGenLStatistical :
            public KSComponentTemplate<KSGenLStatistical, KSGenCreator>
    {
    public:
        KSGenLStatistical();
        KSGenLStatistical(const KSGenLStatistical&);
        KSGenLStatistical* Clone() const;
        virtual ~KSGenLStatistical();

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


#endif //KASPER_KSGENLSTATISTICAL_H
