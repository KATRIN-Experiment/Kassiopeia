#ifndef KFMPointCloud_HH__
#define KFMPointCloud_HH__

#include <vector>

#include "KFMPoint.hh"

namespace KEMField
{

/*
*
*@file KFMPointCloud.hh
*@class KFMPointCloud
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 14:27:45 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM>
class KFMPointCloud
{
    public:
        KFMPointCloud(){};
        virtual ~KFMPointCloud(){};

        KFMPointCloud(const KFMPointCloud& copyObject)
        {
            for(unsigned int i=0; i<copyObject.fPoints.size(); i++)
            {
                fPoints.push_back(copyObject.fPoints[i]);
            }
        }

        KFMPointCloud& operator=(const KFMPointCloud& copyObject)
        {
            if(&copyObject != this)
            {
                fPoints = copyObject.fPoints;
            }
            return *this;
        }

        unsigned int GetNPoints() const
        {
            return fPoints.size();
        }

        void AddPoint(const KFMPoint<NDIM>& point)
        {
            fPoints.push_back(point);
        }

        void Clear()
        {
            fPoints.clear();
        }

        KFMPoint<NDIM> GetPoint(unsigned int i) const {return fPoints[i];}; //no check performed

        KFMPoint<NDIM> GetCentroid() const
        {
            KFMPoint<NDIM> centroid;
            for(unsigned int i=0; i<fPoints.size(); i++)
            {
                centroid += fPoints[i];
            }
            double fac = 1.0/((double)fPoints.size());
            centroid *= fac;
            return centroid;
        };

        double GetRadiusAboutCentroid() const
        {
            KFMPoint<NDIM> centroid = GetCentroid();
            double max_radius = 0.0;
            for(unsigned int i=0; i<fPoints.size(); i++)
            {
                double r = (centroid - fPoints[i]).Magnitude();
                if(r > max_radius){max_radius = r;};
            }
            return max_radius;
        };

        void SetPoints(const std::vector< KFMPoint<NDIM> >* points)
        {
            fPoints = *points;
        }

        void GetPoints( std::vector< KFMPoint<NDIM> >* points) const
        {
            *points = fPoints;
        }

        std::vector< KFMPoint<NDIM> >* GetPoints()
        {
            return &fPoints;
        }


    private:

        std::vector< KFMPoint<NDIM> > fPoints;


};


}//end of KEMField



#endif /* KFMPointCloud_H__ */
