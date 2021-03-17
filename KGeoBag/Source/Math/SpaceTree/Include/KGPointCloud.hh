#ifndef KGPointCloud_HH__
#define KGPointCloud_HH__

#include "KGPoint.hh"

#include <vector>

namespace KGeoBag
{

/*
*
*@file KGPointCloud.hh
*@class KGPointCloud
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 14:27:45 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<size_t NDIM> class KGPointCloud
{
  public:
    KGPointCloud() = default;
    ;
    virtual ~KGPointCloud() = default;
    ;

    KGPointCloud(const KGPointCloud& copyObject)
    {
        for (size_t i = 0; i < copyObject.fPoints.size(); i++) {
            fPoints.push_back(copyObject.fPoints[i]);
        }
    }

    size_t GetNPoints() const
    {
        return fPoints.size();
    }

    void AddPoint(const KGPoint<NDIM>& point)
    {
        fPoints.push_back(point);
    }

    void Clear()
    {
        fPoints.clear();
    }

    KGPoint<NDIM> GetPoint(size_t i) const
    {
        return fPoints[i];
    };  //no check performed

    void SetPoints(const std::vector<KGPoint<NDIM>>* points)
    {
        fPoints = *points;
    }

    void GetPoints(std::vector<KGPoint<NDIM>>* points) const
    {
        *points = fPoints;
    }

    std::vector<KGPoint<NDIM>>* GetPoints()
    {
        return &fPoints;
    }


  private:
    std::vector<KGPoint<NDIM>> fPoints;
};


}  // namespace KGeoBag


#endif /* KGPointCloud_H__ */
