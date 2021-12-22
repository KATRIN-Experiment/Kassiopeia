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
    KSComponent(aCopy),
    fLength(aCopy.fLength)
{}

KSGenPositionSurfaceAdjustmentStep* KSGenPositionSurfaceAdjustmentStep::Clone() const
{
    return new KSGenPositionSurfaceAdjustmentStep(*this);
}

KSGenPositionSurfaceAdjustmentStep::~KSGenPositionSurfaceAdjustmentStep() = default;

void KSGenPositionSurfaceAdjustmentStep::Dice(KSParticleQueue* aPrimaries)
{
    for (auto& aPrimarie : *aPrimaries) {
        genmsg_debug("Position before: <" << aPrimarie->GetPosition() << ">" << eom);
        katrin::KThreeVector tPosition = aPrimarie->GetPosition() + fLength * aPrimarie->GetMomentum().Unit();
        genmsg_debug("KSGenPositionSurfaceAdjustmentStep: <" << GetName() << "> set position <" << tPosition << ">"
                                                             << eom);
        genmsg_debug("Distance between points: " << (aPrimarie->GetPosition() - tPosition).Magnitude() << eom);
        genmsg_debug("Radius smaller by: " << (aPrimarie->GetPosition().Perp() - tPosition.Perp()) << eom);
        aPrimarie->SetPosition(tPosition);
    }
}

void KSGenPositionSurfaceAdjustmentStep::InitializeComponent()
{
    if (fLength < 0.)
        genmsg(eWarning) << "Adjustment Length smaller than 0." << eom;
}
void KSGenPositionSurfaceAdjustmentStep::DeinitializeComponent() {}

}  // namespace Kassiopeia
