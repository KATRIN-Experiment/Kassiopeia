//
// Created by Nikolaus Trost on 08.05.15.
//

#ifndef KASPER_KSGENLSTATISTICAL_H
#define KASPER_KSGENLSTATISTICAL_H

#include "KSGenCreator.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{
/**
     * \brief Dices angular momentum of particles.
     */
class KSGenLStatistical : public KSComponentTemplate<KSGenLStatistical, KSGenCreator>
{
  public:
    KSGenLStatistical();
    KSGenLStatistical(const KSGenLStatistical&);
    KSGenLStatistical* Clone() const override;
    ~KSGenLStatistical() override;

  public:
    /**
         * \brief Dices the quantized angular Momentum L of all particles of
         * the KSParticleQueue equally distributed between 0 and the particles n-1
         *
         *
         * \param aPrimaries
         */
    void Dice(KSParticleQueue* aPrimaries) override;


  public:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia


#endif  //KASPER_KSGENLSTATISTICAL_H
