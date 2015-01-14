#ifndef KFMElectrostaticElementContainer_HH__
#define KFMElectrostaticElementContainer_HH__

#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainerBase.hh"

#include "KFMBasisData.hh"
#include "KFMBall.hh"
#include "KFMPointCloud.hh"

#include "KFMObjectContainer.hh"

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
class KFMElectrostaticElementContainer: public KFMElectrostaticElementContainerBase< SpatialDimension, BasisDimension >
{
    public:

        KFMElectrostaticElementContainer()
        {
            fPointCloudContainer = new KFMObjectContainer< KFMPointCloud<SpatialDimension> >();
            fBoundingBallContainer = new KFMObjectContainer< KFMBall<SpatialDimension> >();
            fBasisDataContainer = new KFMObjectContainer< KFMBasisData<BasisDimension> >();
            fCentroidContainer = new KFMObjectContainer< KFMPoint<SpatialDimension> >();
            fNElements = 0;
        }

        virtual ~KFMElectrostaticElementContainer()
        {
            delete fPointCloudContainer;
            delete fBoundingBallContainer;
            delete fBasisDataContainer;
            delete fCentroidContainer;
        }

        unsigned int GetNElements() const {return fNElements;};

        void AddElectrostaticElement(const KFMElectrostaticElement< SpatialDimension, BasisDimension>& elem)
        {
            fPointCloudContainer->AddObject( elem.GetPointCloud() );
            fBoundingBallContainer->AddObject( elem.GetBoundingBall() );
            fBasisDataContainer->AddObject( elem.GetBasisData() );
            fCentroidContainer->AddObject( elem.GetCentroid() );
            fAspectRatio.push_back(elem.GetAspectRatio());
            fNElements++;
        }

        KFMElectrostaticElement<SpatialDimension, BasisDimension> GetElectrostaticElement(unsigned int id)
        {
            KFMElectrostaticElement<SpatialDimension, BasisDimension> elem;
            elem.SetPointCloud(*(GetPointCloud(id)));
            elem.SetBasisData(*(GetBasisData(id)));
            elem.SetBoundingBall(*(GetBoundingBall(id)));
            elem.SetCentroid(*(GetCentroid(id)));
            elem.SetAspectRatio(fAspectRatio[id]);
            return elem;
        }


        KFMPointCloud<SpatialDimension>* GetPointCloud(unsigned int id){return fPointCloudContainer->GetObjectWithID(id);};
        KFMBall<SpatialDimension>* GetBoundingBall(unsigned int id){return fBoundingBallContainer->GetObjectWithID(id);};
        KFMBasisData<BasisDimension>* GetBasisData(unsigned int id){return fBasisDataContainer->GetObjectWithID(id);};
        KFMPoint<SpatialDimension>* GetCentroid(unsigned int id){return fCentroidContainer->GetObjectWithID(id);};

        const KFMPointCloud<SpatialDimension>* GetPointCloud(unsigned int id) const {return fPointCloudContainer->GetObjectWithID(id);};
        const KFMBall<SpatialDimension>* GetBoundingBall(unsigned int id) const {return fBoundingBallContainer->GetObjectWithID(id);};
        const KFMBasisData<BasisDimension>* GetBasisData(unsigned int id) const {return fBasisDataContainer->GetObjectWithID(id);};
        const KFMPoint<SpatialDimension>* GetCentroid(unsigned int id) const {return fCentroidContainer->GetObjectWithID(id);};
        double GetAspectRatio(unsigned int id) const {return fAspectRatio[id];};


        void Clear()
        {
            fPointCloudContainer->DeleteAllObjects();
            fBoundingBallContainer->DeleteAllObjects();
            fBasisDataContainer->DeleteAllObjects();
            fCentroidContainer->DeleteAllObjects();
            fAspectRatio.clear();
            fNElements = 0;
        }

        KFMObjectContainer< KFMPointCloud<SpatialDimension> >* GetPointCloudContainer(){return fPointCloudContainer;};
        KFMObjectContainer< KFMBall<SpatialDimension> >* GetBoundingBallContainer(){return fBoundingBallContainer;};
        KFMObjectContainer< KFMBasisData<BasisDimension> >* GetChargeDensityContainer(){return fBasisDataContainer;};
        KFMObjectContainer< KFMPoint<SpatialDimension> >* GetCentroidContainer() {return fCentroidContainer;};

        const KFMObjectContainer< KFMPointCloud<SpatialDimension> >* GetPointCloudContainer() const {return fPointCloudContainer;};
        const KFMObjectContainer< KFMBall<SpatialDimension> >* GetBoundingBallContainer() const {return fBoundingBallContainer;};
        const KFMObjectContainer< KFMBasisData<BasisDimension> >* GetChargeDensityContainer() const {return fBasisDataContainer;};
        const KFMObjectContainer< KFMPoint<SpatialDimension> >* GetCentroidContainer() const {return fCentroidContainer;};

    private:

        unsigned int fNElements;

        KFMObjectContainer< KFMPointCloud<SpatialDimension> >* fPointCloudContainer;
        KFMObjectContainer< KFMBall<SpatialDimension> >* fBoundingBallContainer;
        KFMObjectContainer< KFMBasisData<BasisDimension> >* fBasisDataContainer;
        KFMObjectContainer< KFMPoint<SpatialDimension> >* fCentroidContainer;
        std::vector<double> fAspectRatio;

};

}

#endif /* KFMElectrostaticElementContainer_H__ */
