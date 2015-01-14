#ifndef KGCube_HH__
#define KGCube_HH__


#include <cmath>
#include <limits>
#include <bitset>
#include <climits>
#include <sstream>

#include "KGPoint.hh"

namespace KGeoBag
{


/*
*
*@file KGCube.hh
*@class KGCube
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 13:41:27 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<size_t NDIM>
class KGCube: public KSAFixedSizeInputOutputObject
{
    public:

        KGCube(){ for(size_t i=0; i<NDIM+1; i++){ fData[i] = 0.0; } };

        KGCube(const double* center, const double& length){SetParameters(center, length); };

        virtual ~KGCube(){};

        size_t GetDimension() const {return NDIM;};

        inline KGCube(const KGCube &copyObject){ for(size_t i=0; i<NDIM+1; i++){fData[i] = copyObject.fData[i];} }

        //geometric property assignment
        void SetParameters(const double* center, const double& length)
        {
            SetCenter(center);
            SetLength(length);
        }
        void SetLength(const double& len){fData[NDIM] = len;};
        void SetCenter(const double* center){ for(size_t i=0; i<NDIM; i++){fData[i] = center[i];} }
        void SetCenter(const KGPoint<NDIM>& center){ for(size_t i=0; i<NDIM; i++){fData[i] = center[i]; } }

        //geometric property retrieval
        KGPoint<NDIM> GetCenter() const {return KGPoint<NDIM>(fData);};
        void GetCenter(double* center) const { for(size_t i=0; i<NDIM; i++){center[i] = fData[i]; } }
        double GetLength() const {return fData[NDIM];};
        KGPoint<NDIM> GetCorner(size_t i) const
        {
            KGPoint<NDIM> corner;
            //convert the count number into a set of bools which we can use
            //to tell us which direction the corner is in for each dimension
            std::bitset< sizeof(size_t)*CHAR_BIT > twiddle_index = std::bitset< sizeof(size_t)*CHAR_BIT >(i);

            double length_over_two = fData[NDIM]/2.0;

            for(size_t j=0; j<NDIM; j++)
            {
                if(twiddle_index[j])
                {
                    corner[j] = fData[j] + length_over_two;
                }
                else
                {
                    corner[j] = fData[j] - length_over_two;
                }
            }
            return corner;
       }

        //navigation
        bool PointIsInside(const double* p) const
        {
            double length_over_two = fData[NDIM]/2.0;
            double distance;
            for(size_t i=0; i<NDIM; i++)
            {
                distance = p[i] - fData[i]; //distance from center in  i-th dimension
                if(distance < -1.0*length_over_two){return false;}
                if(distance > length_over_two){return false;}
            }
            return true;
        }

        bool CubeIsInside(const KGCube<NDIM>* cube) const
        {
            double distance;
            double cube_len_over_two = cube->GetLength()/2.0;
            for(size_t i=0; i<NDIM; i++)
            {
                distance = (*cube)[i] - fData[i]; //distance from center in  i-th dimension
                if( ( (fData[NDIM]/2.0 - distance) - cube_len_over_two ) < 0 )
                {
                    return false;
                }
            }
            return true;
        }


        inline KGCube& operator= (const KGCube& rhs)
        {
            if(&rhs != this)
            {
                for(size_t i=0; i<NDIM+1; i++){fData[i] = rhs.fData[i];}
            }
            return *this;
        }

        //access elements
        double& operator[](size_t i);
        const double& operator[](size_t i) const;

    protected:

        double fData[NDIM + 1]; //center position + length, last element is length

};

template<size_t NDIM>
inline double& KGCube<NDIM>::operator[](size_t i)
{
    return fData[i];
}

template<size_t NDIM>
inline const double& KGCube<NDIM>::operator[](size_t i) const
{
    return fData[i];
}

}//end of KGeoBag


#endif /* KGCube_H__ */
