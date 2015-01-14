#ifndef KGAxisAlignedBox_HH__
#define KGAxisAlignedBox_HH__


#include <cmath>
#include <limits>
#include <bitset>
#include <climits>

#include "KGPoint.hh"

namespace KGeoBag
{


/*
*
*@file KGAxisAlignedBox.hh
*@class KGAxisAlignedBox
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 13:41:27 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<size_t NDIM>
class KGAxisAlignedBox
{
    public:

        KGAxisAlignedBox()
        {
            for(size_t i=0; i<NDIM; i++)
            {
                fCenter[i] = 0.0;
                fLength[i] = 0.0;
            }
        };


        KGAxisAlignedBox(const double* center, const double* length){SetParameters(center, length); };

        virtual ~KGAxisAlignedBox(){};

        size_t GetDimension() const {return NDIM;};

        inline KGAxisAlignedBox(const KGAxisAlignedBox &copyObject)
        {
            for(size_t i=0; i<NDIM; i++)
            {
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
        void SetLength(const double* len){for(size_t i=0; i<NDIM; i++){fLength[i] = len[i];} };
        void SetCenter(const double* center){ for(size_t i=0; i<NDIM; i++){fCenter[i] = center[i];} }

        //geometric property retrieval
        KGPoint<NDIM> GetCenter() const {return KGPoint<NDIM>(fCenter);};
        double GetLength(size_t i) const {return fLength[i];};
        KGPoint<NDIM> GetCorner(size_t i) const
        {
            KGPoint<NDIM> corner;
            //convert the count number into a set of bools which we can use
            //to tell us which direction the corner is in for each dimension
            std::bitset< sizeof(size_t)*CHAR_BIT > twiddle_index = std::bitset< sizeof(size_t)*CHAR_BIT >(i);

            for(size_t j=0; j<NDIM; j++)
            {
                if(twiddle_index[j])
                {
                    corner[j] = fCenter[j] + fLength[j]/2.0;
                }
                else
                {
                    corner[j] = fCenter[j] - fLength[j]/2.0;
                }
            }
            return corner;
       }

        //navigation
        bool PointIsInside(const double* p) const
        {
            double distance;
            for(size_t i=0; i<NDIM; i++)
            {
                distance = p[i] - fCenter[i]; //displacement from center in  i-th dimension
                if(distance < -1.0*fLength[i]/2.0){return false;}
                if(distance > fLength[i]/2.0){return false;}
            }
            return true;
        }

        bool BoxIsInside(const KGAxisAlignedBox<NDIM>* box) const
        {
            double distance;
            for(size_t i=0; i<NDIM; i++)
            {
                distance = (*box)[i] - fCenter[i]; //distance from center in  i-th dimension
                if( ( (fLength[NDIM]/2.0 - distance) - (box->GetLength(i)/2.0) ) < 0 )
                {
                    return false;
                }
            }

            return true;
        }



        inline KGAxisAlignedBox& operator= (const KGAxisAlignedBox& rhs)
        {
            if(&rhs != this)
            {
                for(size_t i=0; i<NDIM; i++)
                {
                    fCenter[i] = rhs.fCenter[i];
                    fLength[i] = rhs.fLength[i];
                }
            }
            return *this;
        }

        //access elements
        double& operator[](size_t i);
        const double& operator[](size_t i) const;

    protected:

        double fCenter[NDIM];
        double fLength[NDIM];
};

template<size_t NDIM>
inline double& KGAxisAlignedBox<NDIM>::operator[](size_t i)
{
    if(i < NDIM)
    {
        return fCenter[i];
    }
    else
    {
        return fLength[i];
    }
}

template<size_t NDIM>
inline const double& KGAxisAlignedBox<NDIM>::operator[](size_t i) const
{
    if(i < NDIM)
    {
        return fCenter[i];
    }
    else
    {
        return fLength[i];
    }
}



}//end of KGeoBag


#endif /* KGAxisAlignedBox_H__ */
