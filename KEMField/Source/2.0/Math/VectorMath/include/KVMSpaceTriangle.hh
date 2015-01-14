#ifndef KVMSpaceTriangle_H
#define KVMSpaceTriangle_H

#include <limits>
#include <string>
#include <cmath>

#include "KEMThreeVector.hh"

namespace KEMField{

/**
*
*@file KVMSpaceTriangle.hh
*@class KVMSpaceTriangle
*@brief container class for an oriented triangle in 3-space, it obtains its
*       information from KTTriangleElectrode,
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jul 10 22:56:22 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/



class KVMSpaceTriangle
{
    public:
        KVMSpaceTriangle();
        ~KVMSpaceTriangle(){;};

        void SetAll(const double* point, const double* vec1, const double* vec2, double len1, double len2);
        void SetVertices(const double* point0, const double* point1, const double* point2);
        void GetVertices(double* point0, double* point1, double* point2) const;


        double GetCornerPoint(int i) const {return fP[i];};
        double GetBasisVectorN1(int i) const {return fN1[i];};
        double GetBasisVectorN2(int i) const {return fN2[i];};
        double GetLengthSide1() const {return fL1;};
        double GetLengthSide2() const {return fL2;};
        double GetArea() const {return 0.5*fL1*fL2*(fN1.Cross(fN2)).Magnitude();};

        inline KVMSpaceTriangle(const KVMSpaceTriangle &copyObject);

        inline KVMSpaceTriangle& operator=(const KVMSpaceTriangle &rhs);

    protected:

        void InitializeParameters();

        void SetCornerPoint(const double* point);
        void SetBasisVectorN1(const double* vec);
        void SetBasisVectorN2(const double* vec);
        void SetLengthSide1(double len){fL1 = len;};
        void SetLengthSide2(double len){fL2 = len;};

        //descriptor variables
        //indexing is: [0] = x, [1] = y, [2] = z
        KEMThreeVector fP; //corner point
        KEMThreeVector fN1; //1st basis vector
        KEMThreeVector fN2; //2nd basis vector (non-orthogonal)
        KEMThreeVector fN3; //normal vector
        double fL1; //length of side along N1
        double fL2; //length of side along N2
        double fSinTheta;
};

inline KVMSpaceTriangle::KVMSpaceTriangle(const KVMSpaceTriangle &copyObject)
{
    fP = copyObject.fP;
    fN1 = copyObject.fN1;
    fN2 = copyObject.fN2;
    fL1 = copyObject.fL1;
    fL2 = copyObject.fL2;
    InitializeParameters();
};

inline KVMSpaceTriangle& KVMSpaceTriangle::operator=(const KVMSpaceTriangle &rhs)
{
    if(this != &rhs)
    {
        fP = rhs.fP;
        fN1 = rhs.fN1;
        fN2 = rhs.fN2;
        fL1 = rhs.fL1;
        fL2 = rhs.fL2;
        InitializeParameters();
    }
    return *this;
}


}

#endif /* KVMSpaceTriangle_H */
