#ifndef KGPointCloudToBoundingBallConverter_HH__
#define KGPointCloudToBoundingBallConverter_HH__

#include <vector>

#include "KGPointCloud.hh"
#include "KGBall.hh"
#include "KGBoundaryCalculator.hh"

namespace KGeoBag
{

/*
*
*@file KGPointCloudToBoundingBallConverter.hh
*@class KGPointCloudToBoundingBallConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 16:08:07 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<size_t NDIM>
class KGPointCloudToBoundingBallConverter
{
    public:
        KGPointCloudToBoundingBallConverter(){};
        virtual ~KGPointCloudToBoundingBallConverter(){};

        void Convert(const std::vector< KGPointCloud<NDIM> >* cloud_container, std::vector< KGBall<NDIM> >* ball_container) const
        {
            ball_container->clear();

            size_t n_clouds = cloud_container->size();

            for(size_t i=0; i<n_clouds; i++)
            {
                ball_container->push_back( Convert( &(cloud_container->at(t)) ) );
            }
        }

        KGBall<NDIM> Convert(const KGPointCloud<NDIM>* cloud) const
        {
            fCalculator.Reset();
            fCalculator.AddPointCloud(cloud);
            return fCalculator.GetMinimalBoundingBall();
        }

    private:

        mutable KGBoundaryCalculator<NDIM> fCalculator;

};


}


#endif /* KGPointCloudToBoundingBallConverter_H__ */
