#ifndef KFMBox_HH__
#define KFMBox_HH__


#include "KFMPoint.hh"

#include <bitset>
#include <climits>
#include <cmath>
#include <limits>

namespace KEMField
{


/*
*
*@file KFMBox.hh
*@class KFMBox
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 13:41:27 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM> class KFMBox
{
  public:
    KFMBox()
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fCenter[i] = 0.0;
            fLength[i] = 0.0;
        }
    };


    KFMBox(const double* center, const double* length)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fCenter[i] = 0.0;
            fLength[i] = 0.0;
        }
        SetParameters(center, length);
    };

    virtual ~KFMBox() = default;
    ;

    unsigned int GetDimension() const
    {
        return NDIM;
    };

    inline KFMBox(const KFMBox& copyObject)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fCenter[i] = copyObject.fCenter[i];
            fLength[i] = copyObject.fLength[i];
        }
    }

    //geometric property assignment
    void SetParameters(const double* center, const double& length)
    {
        SetCenter(center);
        SetLength(length);
    }
    void SetLength(const double* len)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fLength[i] = len[i];
        }
    };
    void SetCenter(const double* center)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fCenter[i] = center[i];
        }
    }

    //geometric property retrieval
    KFMPoint<NDIM> GetCenter() const
    {
        return KFMPoint<NDIM>(fCenter);
    };
    double GetLength(unsigned int i) const
    {
        return fLength[i];
    };
    KFMPoint<NDIM> GetCorner(unsigned int i) const
    {
        KFMPoint<NDIM> corner;
        //convert the count number into a set of bools which we can use
        //to tell us which direction the corner is in for each dimension
        std::bitset<sizeof(unsigned int)* CHAR_BIT> twiddle_index = std::bitset<sizeof(unsigned int) * CHAR_BIT>(i);

        for (unsigned int j = 0; j < NDIM; j++) {
            if (twiddle_index[j]) {
                corner[j] = fCenter[j] + fLength[j] / 2.0;
            }
            else {
                corner[j] = fCenter[j] - fLength[j] / 2.0;
            }
        }
        return corner;
    }

    //navigation
    bool PointIsInside(const double* p) const
    {
        double distance;
        for (unsigned int i = 0; i < NDIM; i++) {
            distance = p[i] - fCenter[i];  //displacement from center in  i-th dimension
            if (distance < -1.0 * fLength[i] / 2.0) {
                return false;
            }
            if (distance > fLength[i] / 2.0) {
                return false;
            }
        }
        return true;
    }

    bool BoxIsInside(const KFMBox<NDIM>* box) const
    {
        double distance;
        for (unsigned int i = 0; i < NDIM; i++) {
            distance = std::fabs((*box)[i] - fCenter[i]);  //distance from center in  i-th dimension
            if (((fLength[NDIM] / 2.0 - distance) - (box->GetLength(i) / 2.0)) < 0) {
                return false;
            }
        }

        return true;
    }


    inline KFMBox& operator=(const KFMBox& rhs)
    {
        if (&rhs != this) {
            for (unsigned int i = 0; i < NDIM; i++) {
                fCenter[i] = rhs.fCenter[i];
                fLength[i] = rhs.fLength[i];
            }
        }
        return *this;
    }

    //access elements
    double& operator[](unsigned int i);
    const double& operator[](unsigned int i) const;

  protected:
    double fCenter[NDIM];
    double fLength[NDIM];
};

template<unsigned int NDIM> inline double& KFMBox<NDIM>::operator[](unsigned int i)
{
    if (i < NDIM) {
        return fCenter[i];
    }
    else {
        return fLength[i];
    }
}

template<unsigned int NDIM> inline const double& KFMBox<NDIM>::operator[](unsigned int i) const
{
    if (i < NDIM) {
        return fCenter[i];
    }
    else {
        return fLength[i];
    }
}


}  // namespace KEMField


#endif /* KFMBox_H__ */
