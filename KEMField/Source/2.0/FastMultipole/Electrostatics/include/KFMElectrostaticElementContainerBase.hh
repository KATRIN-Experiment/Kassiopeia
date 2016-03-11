#ifndef KFMElectrostaticElementContainerBase_HH__
#define KFMElectrostaticElementContainerBase_HH__

#include "KFMElectrostaticElement.hh"

#include "KFMBasisData.hh"
#include "KFMBall.hh"
#include "KFMPointCloud.hh"
#include "KFMObjectContainer.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticElementContainerBase.hh
*@class KFMElectrostaticElementContainerBase
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug 28 19:01:39 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int SpatialDimension, unsigned int BasisDimension>
class KFMElectrostaticElementContainerBase
{
    public:

        KFMElectrostaticElementContainerBase(){;};
        virtual ~KFMElectrostaticElementContainerBase(){;};

        virtual unsigned int GetNElements() const = 0;

        virtual void AddElectrostaticElement(const KFMElectrostaticElement< SpatialDimension, BasisDimension>& elem) = 0;
        virtual KFMElectrostaticElement<SpatialDimension, BasisDimension> GetElectrostaticElement(unsigned int id) = 0;

        virtual KFMPointCloud<SpatialDimension>* GetPointCloud(unsigned int id) = 0;
        virtual KFMBall<SpatialDimension>* GetBoundingBall(unsigned int id) = 0;
        virtual KFMBasisData<BasisDimension>* GetBasisData(unsigned int id) = 0;
        virtual KFMPoint<SpatialDimension>* GetCentroid(unsigned int id) = 0;

        virtual const KFMPointCloud<SpatialDimension>* GetPointCloud(unsigned int id) const = 0;
        virtual const KFMBall<SpatialDimension>* GetBoundingBall(unsigned int id) const = 0;
        virtual const KFMBasisData<BasisDimension>* GetBasisData(unsigned int id) const = 0;
        virtual const KFMPoint<SpatialDimension>* GetCentroid(unsigned int id) const = 0;
        virtual double GetAspectRatio(unsigned int id) const = 0;

        virtual void ClearBoundingBalls(){;};
        virtual void Clear(){;};

        virtual KFMObjectContainer< KFMPointCloud<SpatialDimension> >* GetPointCloudContainer() = 0;
        virtual KFMObjectContainer< KFMBall<SpatialDimension> >* GetBoundingBallContainer() = 0;
        virtual KFMObjectContainer< KFMBasisData<BasisDimension> >* GetChargeDensityContainer() = 0;
        virtual KFMObjectContainer< KFMPoint<SpatialDimension> >* GetCentroidContainer() = 0;

        virtual const KFMObjectContainer< KFMPointCloud<SpatialDimension> >* GetPointCloudContainer() const = 0;
        virtual const KFMObjectContainer< KFMBall<SpatialDimension> >* GetBoundingBallContainer() const = 0;
        virtual const KFMObjectContainer< KFMBasisData<BasisDimension> >* GetChargeDensityContainer() const = 0;
        virtual const KFMObjectContainer< KFMPoint<SpatialDimension> >* GetCentroidContainer() const = 0;

    private:

};

}

#endif /* KFMElectrostaticElementContainerBase_H__ */
