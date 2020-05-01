/*
 * KSGenPositionSurfaceAdjustmentStep.h
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#include "KSGenPositionSurfaceAdjustmentStep.h"


namespace Kassiopeia
{
KSGenPositionSurfaceAdjustmentStep::KSGenPositionSurfaceAdjustmentStep() : fLength(0.0) {}

KSGenPositionSurfaceAdjustmentStep::KSGenPositionSurfaceAdjustmentStep(
    const KSGenPositionSurfaceAdjustmentStep& aCopy) :
    KSComponent(),
    fLength(aCopy.fLength)
{}

KSGenPositionSurfaceAdjustmentStep* KSGenPositionSurfaceAdjustmentStep::Clone() const
{
    return new KSGenPositionSurfaceAdjustmentStep(*this);
}

KSGenPositionSurfaceAdjustmentStep::~KSGenPositionSurfaceAdjustmentStep() {}

void KSGenPositionSurfaceAdjustmentStep::Dice(KSParticleQueue* aPrimaries)
{
    for (auto tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); ++tParticleIt) {
        genmsg_debug("Position before: <" << (*tParticleIt)->GetPosition() << ">" << eom);
        KThreeVector tPosition = (*tParticleIt)->GetPosition() + fLength * (*tParticleIt)->GetMomentum().Unit();
        genmsg_debug("KSGenPositionSurfaceAdjustmentStep: <" << GetName() << "> set position <" << tPosition << ">"
                                                             << eom);
        genmsg_debug("Distance between points: " << ((*tParticleIt)->GetPosition() - tPosition).Magnitude() << eom);
        genmsg_debug("Radius smaller by: " << (*tParticleIt)->GetPosition().Perp() - tPosition.Perp() << eom);
        (*tParticleIt)->SetPosition(tPosition);
    }
}

void KSGenPositionSurfaceAdjustmentStep::InitializeComponent()
{
    if (fLength < 0.)
        genmsg(eWarning) << "Adjustment Length smaller than 0." << eom;
}
void KSGenPositionSurfaceAdjustmentStep::DeinitializeComponent() {}

}  // namespace Kassiopeia
