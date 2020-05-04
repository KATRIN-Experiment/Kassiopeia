#ifndef KFMElectrostaticElementContainer_HH__
#define KFMElectrostaticElementContainer_HH__

#include "KFMBall.hh"
#include "KFMBasisData.hh"
#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainerBase.hh"
#include "KFMObjectContainer.hh"
#include "KFMPointCloud.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticElementContainer.hh
*@class KFMElectrostaticElementContainer
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug 28 19:01:39 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int SpatialDimension, unsigned int BasisDimension>
class KFMElectrostaticElementContainer : public KFMElectrostaticElementContainerBase<SpatialDimension, BasisDimension>
{
  public:
    KFMElectrostaticElementContainer()
    {
        fPointCloudContainer = new KFMObjectContainer<KFMPointCloud<SpatialDimension>>();
        fBoundingBallContainer = new KFMObjectContainer<KFMBall<SpatialDimension>>();
        fBasisDataContainer = new KFMObjectContainer<KFMBasisData<BasisDimension>>();
        fCentroidContainer = new KFMObjectContainer<KFMPoint<SpatialDimension>>();
        fNElements = 0;
    }

    ~KFMElectrostaticElementContainer() override
    {
        delete fPointCloudContainer;
        delete fBoundingBallContainer;
        delete fBasisDataContainer;
        delete fCentroidContainer;
    }

    unsigned int GetNElements() const override
    {
        return fNElements;
    };

    void AddElectrostaticElement(const KFMElectrostaticElement<SpatialDimension, BasisDimension>& elem) override
    {
        fPointCloudContainer->AddObject(elem.GetPointCloud());
        fBoundingBallContainer->AddObject(elem.GetBoundingBall());
        fBasisDataContainer->AddObject(elem.GetBasisData());
        fCentroidContainer->AddObject(elem.GetCentroid());
        fAspectRatio.push_back(elem.GetAspectRatio());
        fNElements++;
    }

    KFMElectrostaticElement<SpatialDimension, BasisDimension> GetElectrostaticElement(unsigned int id) override
    {
        KFMElectrostaticElement<SpatialDimension, BasisDimension> elem;
        elem.SetPointCloud(*(GetPointCloud(id)));
        elem.SetBasisData(*(GetBasisData(id)));
        elem.SetBoundingBall(*(GetBoundingBall(id)));
        elem.SetCentroid(*(GetCentroid(id)));
        elem.SetAspectRatio(fAspectRatio[id]);
        return elem;
    }


    KFMPointCloud<SpatialDimension>* GetPointCloud(unsigned int id) override
    {
        return fPointCloudContainer->GetObjectWithID(id);
    };
    KFMBall<SpatialDimension>* GetBoundingBall(unsigned int id) override
    {
        return fBoundingBallContainer->GetObjectWithID(id);
    };
    KFMBasisData<BasisDimension>* GetBasisData(unsigned int id) override
    {
        return fBasisDataContainer->GetObjectWithID(id);
    };
    KFMPoint<SpatialDimension>* GetCentroid(unsigned int id) override
    {
        return fCentroidContainer->GetObjectWithID(id);
    };

    const KFMPointCloud<SpatialDimension>* GetPointCloud(unsigned int id) const override
    {
        return fPointCloudContainer->GetObjectWithID(id);
    };
    const KFMBall<SpatialDimension>* GetBoundingBall(unsigned int id) const override
    {
        return fBoundingBallContainer->GetObjectWithID(id);
    };
    const KFMBasisData<BasisDimension>* GetBasisData(unsigned int id) const override
    {
        return fBasisDataContainer->GetObjectWithID(id);
    };
    const KFMPoint<SpatialDimension>* GetCentroid(unsigned int id) const override
    {
        return fCentroidContainer->GetObjectWithID(id);
    };
    double GetAspectRatio(unsigned int id) const override
    {
        return fAspectRatio[id];
    };

    //after the tree is constructed the bounding balls are no longer needed
    void ClearBoundingBalls() override
    {
        fBoundingBallContainer->DeleteAllObjects();
    }

    void Clear() override
    {
        fPointCloudContainer->DeleteAllObjects();
        fBoundingBallContainer->DeleteAllObjects();
        fBasisDataContainer->DeleteAllObjects();
        fCentroidContainer->DeleteAllObjects();
        fAspectRatio.clear();
        fNElements = 0;
    }

    KFMObjectContainer<KFMPointCloud<SpatialDimension>>* GetPointCloudContainer() override
    {
        return fPointCloudContainer;
    };
    KFMObjectContainer<KFMBall<SpatialDimension>>* GetBoundingBallContainer() override
    {
        return fBoundingBallContainer;
    };
    KFMObjectContainer<KFMBasisData<BasisDimension>>* GetChargeDensityContainer() override
    {
        return fBasisDataContainer;
    };
    KFMObjectContainer<KFMPoint<SpatialDimension>>* GetCentroidContainer() override
    {
        return fCentroidContainer;
    };

    const KFMObjectContainer<KFMPointCloud<SpatialDimension>>* GetPointCloudContainer() const override
    {
        return fPointCloudContainer;
    };
    const KFMObjectContainer<KFMBall<SpatialDimension>>* GetBoundingBallContainer() const override
    {
        return fBoundingBallContainer;
    };
    const KFMObjectContainer<KFMBasisData<BasisDimension>>* GetChargeDensityContainer() const override
    {
        return fBasisDataContainer;
    };
    const KFMObjectContainer<KFMPoint<SpatialDimension>>* GetCentroidContainer() const override
    {
        return fCentroidContainer;
    };

  private:
    unsigned int fNElements;

    KFMObjectContainer<KFMPointCloud<SpatialDimension>>* fPointCloudContainer;
    KFMObjectContainer<KFMBall<SpatialDimension>>* fBoundingBallContainer;
    KFMObjectContainer<KFMBasisData<BasisDimension>>* fBasisDataContainer;
    KFMObjectContainer<KFMPoint<SpatialDimension>>* fCentroidContainer;
    std::vector<double> fAspectRatio;
};

}  // namespace KEMField

#endif /* KFMElectrostaticElementContainer_H__ */
