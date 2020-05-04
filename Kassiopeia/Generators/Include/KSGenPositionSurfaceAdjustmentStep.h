/*
 * KSGenPositionSurfaceAdjustmentStep.h
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#ifndef _KSGenPositionSurfaceAdjustmentStep_H_
#define _KSGenPositionSurfaceAdjustmentStep_H_

#include "KField.h"
#include "KSGenCreator.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{
/**
    * \brief Adjusts the position of a particle by a linear step in the direction of the momentum
    */
class KSGenPositionSurfaceAdjustmentStep : public KSComponentTemplate<KSGenPositionSurfaceAdjustmentStep, KSGenCreator>
{
  public:
    KSGenPositionSurfaceAdjustmentStep();
    KSGenPositionSurfaceAdjustmentStep(const KSGenPositionSurfaceAdjustmentStep&);
    KSGenPositionSurfaceAdjustmentStep* Clone() const override;
    ~KSGenPositionSurfaceAdjustmentStep() override;

  public:
    /**
        * \brief Actually a deterministic function
        *
        * \param aPrimaries
        */
    void Dice(KSParticleQueue* aPrimaries) override;


  private:
    K_SET_GET(double, Length)


  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia

#endif /* _KSGenPositionSurfaceAdjustmentStep_H_ */
