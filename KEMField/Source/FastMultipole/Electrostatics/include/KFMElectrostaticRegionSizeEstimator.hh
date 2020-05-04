#ifndef KFMElectrostaticRegionSizeEstimator_HH__
#define KFMElectrostaticRegionSizeEstimator_HH__

#include "KFMBoundaryCalculator.hh"
#include "KFMCube.hh"
#include "KFMElectrostaticElementContainerBase.hh"

#include <cstddef>

namespace KEMField
{

/*
*
*@file KFMElectrostaticRegionSizeEstimator.hh
*@class KFMElectrostaticRegionSizeEstimator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep  4 18:16:42 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticRegionSizeEstimator
{
  public:
    KFMElectrostaticRegionSizeEstimator();
    virtual ~KFMElectrostaticRegionSizeEstimator();

    void SetElectrostaticElementContainer(const KFMElectrostaticElementContainerBase<3, 1>* container)
    {
        fElementContainer = container;
    }

    void ComputeEstimate();

    KFMCube<3> GetCubeEstimate() const;
    KFMBall<3> GetBallEstimate() const;
    KFMBox<3> GetBoxEstimate() const;

  private:
    const KFMElectrostaticElementContainerBase<3, 1>* fElementContainer;

    KFMBoundaryCalculator<3>* fEstimator;
};

}  // namespace KEMField

#endif /* KFMElectrostaticRegionSizeEstimator_H__ */
