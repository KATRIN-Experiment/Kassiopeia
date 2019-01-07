#ifndef KVMSpaceRectangle_H
#define KVMSpaceRectangle_H

#include <string>
#include <limits>
#include <cmath>

#include "../../include/KThreeVector_KEMField.hh"



namespace KEMField {

/**
*
*@file KVMSpaceRectangle.hh
*@class KVMSpaceRectangle
*@brief container class for an oriented rectangle in 3-space, it obtains its
*       information from KTRectangleElectrode,
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul 11 09:29:21 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KVMSpaceRectangle
{
    public:
        KVMSpaceRectangle();
        ~KVMSpaceRectangle(){;};

        void SetAll(const double* point, const double* vec1, const double* vec2, double len1, double len2);
        void SetVertices(const double* point0, const double* point1, const double* point2, const double* point3);
        void GetVertices(double* point0, double* point1, double* point2, double* point3) const;


        double GetCornerPoint(int i) const {return fP[i];};
        double GetBasisVectorN1(int i) const {return fN1[i];};
        double GetBasisVectorN2(int i) const {return fN2[i];};
        double GetLengthSide1() const {return fL1;};
        double GetLengthSide2() const {return fL2;};
        double GetArea() const {return fL1*fL2;};

        inline KVMSpaceRectangle(const KVMSpaceRectangle &copyObject);

        inline KVMSpaceRectangle& operator=(const KVMSpaceRectangle& rhs);

    protected:


        void InitializeParameters(){;};

        void SetCornerPoint(const double* point);
        void SetBasisVectorN1(const double* vec);
        void SetBasisVectorN2(const double* vec);
        void SetLengthSide1(double len){fL1 = len;};
        void SetLengthSide2(double len){fL2 = len;};

        //indexing is: [0] = x, [1] = y, [2] = z
        KThreeVector fP; //corner point
        KThreeVector fN1; //1st basis vector
        KThreeVector fN2; //2nd basis vector
        double fL1; //length of side along N1
        double fL2; //length of side along N2

};


inline KVMSpaceRectangle::KVMSpaceRectangle(const KVMSpaceRectangle &copyObject)
{
    fP = copyObject.fP;
    fN1 = copyObject.fN1;
    fN2 = copyObject.fN2;
    fL1 = copyObject.fL1;
    fL2 = copyObject.fL2;
    InitializeParameters();
}

inline KVMSpaceRectangle& KVMSpaceRectangle::operator=(const KVMSpaceRectangle& rhs)
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

#endif /* KVMSpaceRectangle_H */
