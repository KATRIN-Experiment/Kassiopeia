#ifndef KFMElectrostaticElementContainerFlyweight_HH__
#define KFMElectrostaticElementContainerFlyweight_HH__


#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainerBase.hh"

#include "KFMBasisData.hh"
#include "KFMBall.hh"
#include "KFMPointCloud.hh"

#include "KFMObjectContainer.hh"


#include "KFMPointCloudContainer.hh"
#include "KFMBoundingBallContainer.hh"
#include "KFMElectrostaticBasisDataContainer.hh"

#include "KSurfaceContainer.hh"
#include "KSortedSurfaceContainer.hh"

#include "KFMElementAspectRatioExtractor.hh"

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

class KFMElectrostaticElementContainerFlyweight: public KFMElectrostaticElementContainerBase< 3, 1 >
{
    public:
        KFMElectrostaticElementContainerFlyweight(const KSurfaceContainer& container);
        KFMElectrostaticElementContainerFlyweight(const KSortedSurfaceContainer& container);
        virtual ~KFMElectrostaticElementContainerFlyweight(){};

        virtual unsigned int GetNElements() const;

        virtual void AddElectrostaticElement(const KFMElectrostaticElement< 3, 1>& /*elem*/);

        virtual KFMElectrostaticElement<3, 1> GetElectrostaticElement(unsigned int id);

        virtual KFMPointCloud<3>* GetPointCloud(unsigned int id);
        virtual KFMBall<3>* GetBoundingBall(unsigned int id);
        virtual KFMBasisData<1>* GetBasisData(unsigned int id);

        virtual const KFMPointCloud<3>* GetPointCloud(unsigned int id) const;
        virtual const KFMBall<3>* GetBoundingBall(unsigned int id) const;
        virtual const KFMBasisData<1>* GetBasisData(unsigned int id) const;
        virtual double GetAspectRatio(unsigned int id) const;

        virtual KFMObjectContainer< KFMPointCloud<3> >* GetPointCloudContainer();
        virtual KFMObjectContainer< KFMBall<3> >* GetBoundingBallContainer();
        virtual KFMObjectContainer< KFMBasisData<1> >* GetChargeDensityContainer();

        virtual const KFMObjectContainer< KFMPointCloud<3> >* GetPointCloudContainer() const;
        virtual const KFMObjectContainer< KFMBall<3> >* GetBoundingBallContainer() const;
        virtual const KFMObjectContainer< KFMBasisData<1> >* GetChargeDensityContainer() const;


    private:

        const KSurfaceContainer* fSurfaceContainer;
        const KSortedSurfaceContainer* fSortedSurfaceContainer;
        bool fContainerIsSorted;
        KFMPointCloudContainer fPointCloudContainer;
        KFMBoundingBallContainer fBoundingBallContainer;
        KFMObjectContainer< KFMBasisData<1> > fBasisContainer;

        std::vector<double> fAspectRatio;

};

}


#endif /* KFMElectrostaticElementContainerFlyweight_H__ */
