#ifndef KFMElectrostaticElementContainerFlyweight_HH__
#define KFMElectrostaticElementContainerFlyweight_HH__


#include "KFMBall.hh"
#include "KFMBasisData.hh"
#include "KFMBoundingBallContainer.hh"
#include "KFMElectrostaticBasisDataContainer.hh"
#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainerBase.hh"
#include "KFMElementAspectRatioExtractor.hh"
#include "KFMObjectContainer.hh"
#include "KFMPointCloud.hh"
#include "KFMPointCloudContainer.hh"
#include "KSortedSurfaceContainer.hh"
#include "KSurfaceContainer.hh"
#include "KVector.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticElementContainerFlyweight.hh
*@class KFMElectrostaticElementContainerFlyweight
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr 10 13:26:03 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticElementContainerFlyweight : public KFMElectrostaticElementContainerBase<3, 1>
{
  public:
    KFMElectrostaticElementContainerFlyweight(const KSurfaceContainer& container);
    KFMElectrostaticElementContainerFlyweight(const KSortedSurfaceContainer& container);
    ~KFMElectrostaticElementContainerFlyweight() override = default;
    ;

    unsigned int GetNElements() const override;

    void AddElectrostaticElement(const KFMElectrostaticElement<3, 1>& /*elem*/) override;

    KFMElectrostaticElement<3, 1> GetElectrostaticElement(unsigned int id) override;

    KFMPointCloud<3>* GetPointCloud(unsigned int id) override;
    KFMBall<3>* GetBoundingBall(unsigned int id) override;
    KFMBasisData<1>* GetBasisData(unsigned int id) override;

    const KFMPointCloud<3>* GetPointCloud(unsigned int id) const override;
    const KFMBall<3>* GetBoundingBall(unsigned int id) const override;
    const KFMBasisData<1>* GetBasisData(unsigned int id) const override;
    double GetAspectRatio(unsigned int id) const override;

    KFMObjectContainer<KFMPointCloud<3>>* GetPointCloudContainer() override;
    KFMObjectContainer<KFMBall<3>>* GetBoundingBallContainer() override;
    KFMObjectContainer<KFMBasisData<1>>* GetChargeDensityContainer() override;

    const KFMObjectContainer<KFMPointCloud<3>>* GetPointCloudContainer() const override;
    const KFMObjectContainer<KFMBall<3>>* GetBoundingBallContainer() const override;
    const KFMObjectContainer<KFMBasisData<1>>* GetChargeDensityContainer() const override;


  private:
    const KSurfaceContainer* fSurfaceContainer;
    const KSortedSurfaceContainer* fSortedSurfaceContainer;
    bool fContainerIsSorted;
    KFMPointCloudContainer fPointCloudContainer;
    KFMBoundingBallContainer fBoundingBallContainer;
    KFMObjectContainer<KFMBasisData<1>> fBasisContainer;

    std::vector<double> fAspectRatio;
};

}  // namespace KEMField


#endif /* KFMElectrostaticElementContainerFlyweight_H__ */
