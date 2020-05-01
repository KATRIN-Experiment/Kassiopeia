#include "KFMElectrostaticRegionSizeEstimator.hh"

namespace KEMField
{


KFMElectrostaticRegionSizeEstimator::KFMElectrostaticRegionSizeEstimator() : fElementContainer(nullptr)
{
    fEstimator = new KFMBoundaryCalculator<3>();
}

KFMElectrostaticRegionSizeEstimator::~KFMElectrostaticRegionSizeEstimator()
{
    delete fEstimator;
}

void KFMElectrostaticRegionSizeEstimator::ComputeEstimate()
{
    unsigned int n_elem = fElementContainer->GetNElements();
    for (unsigned int i = 0; i < n_elem; i++) {
        fEstimator->AddBall(fElementContainer->GetBoundingBall(i));
    }
}

KFMCube<3> KFMElectrostaticRegionSizeEstimator::GetCubeEstimate() const
{
    return fEstimator->GetMinimalBoundingCube();
}

KFMBall<3> KFMElectrostaticRegionSizeEstimator::GetBallEstimate() const
{
    return fEstimator->GetMinimalBoundingBall();
}

KFMBox<3> KFMElectrostaticRegionSizeEstimator::GetBoxEstimate() const
{
    return fEstimator->GetMinimalBoundingBox();
}


}  // namespace KEMField
