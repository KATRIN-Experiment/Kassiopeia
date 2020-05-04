#ifndef KFMBoxSupportSet_HH__
#define KFMBoxSupportSet_HH__


#include "KFMBox.hh"
#include "KFMPoint.hh"

namespace KEMField
{

/*
*
*@file KFMBoxSupportSet.hh
*@class KFMBoxSupportSet
*@brief computes axis aligned bounding box of a set of points
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Aug 18 15:50:46 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM> class KFMBoxSupportSet
{
  public:
    KFMBoxSupportSet() : fCurrentMinimalBoundingBox()
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fLength[i] = 0;
        }
        fCurrentMinimalBoundingBox.SetLength(fLength);
        fAllPoints.clear();
    };

    virtual ~KFMBoxSupportSet()
    {
        ;
    };

    unsigned int GetNSupportPoints()
    {
        if (fAllPoints.size() <= 1) {
            return fAllPoints.size();
        }
        else {
            return 2;
        }
    };

    unsigned int GetNPoints()
    {
        return fAllPoints.size();
    };

    bool AddPoint(const KFMPoint<NDIM>& point)
    {
        if (fAllPoints.size() == 0) {
            for (unsigned int i = 0; i < NDIM; i++) {
                fLength[i] = 0;
            }

            fCurrentMinimalBoundingBox.SetCenter(point);
            fCurrentMinimalBoundingBox.SetLength(fLength);  //no extent

            for (unsigned int i = 0; i < NDIM; i++) {
                fLowerLimits[i] = point[i];
                fUpperLimits[i] = point[i];
            }

            fAllPoints.push_back(point);
            return true;
        }
        else {
            fAllPoints.push_back(point);
            bool update;
            for (unsigned int i = 0; i < NDIM; i++) {
                update = false;

                if (point[i] < fLowerLimits[i]) {
                    fLowerLimits[i] = point[i];
                    update = true;
                };
                if (point[i] > fUpperLimits[i]) {
                    fUpperLimits[i] = point[i];
                    update = true;
                };

                if (update) {
                    fCenter[i] = (fLowerLimits[i] + fUpperLimits[i]) / 2.0;
                    fLength[i] = (fUpperLimits[i] - fLowerLimits[i]);
                }
            }

            fCurrentMinimalBoundingBox.SetCenter(fCenter);
            fCurrentMinimalBoundingBox.SetLength(fLength);

            return true;
        }
    }

    void GetAllPoints(std::vector<KFMPoint<NDIM>>* points) const
    {
        *points = fAllPoints;
    }

    void GetSupportPoints(std::vector<KFMPoint<NDIM>>* points) const
    {
        if (fAllPoints.size() <= 1) {
            *points = fAllPoints;
        }
        else {
            points->clear();
            points->push_back(fLowerLimits);
            points->push_back(fUpperLimits);
        }
    }

    void Clear()
    {
        fAllPoints.clear();
    }


    KFMBox<NDIM> GetMinimalBoundingBox() const
    {
        return fCurrentMinimalBoundingBox;
    }

  private:
    //the two support points
    KFMPoint<NDIM> fLowerLimits;
    KFMPoint<NDIM> fUpperLimits;

    KFMBox<NDIM> fCurrentMinimalBoundingBox;

    std::vector<KFMPoint<NDIM>> fAllPoints;

    //scratch space
    double fCenter[NDIM];
    double fLength[NDIM];
};


}  // namespace KEMField

#endif /* KFMBoxSupportSet_H__ */
