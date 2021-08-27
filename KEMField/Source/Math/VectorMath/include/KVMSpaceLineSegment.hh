#ifndef KVMSpaceLineSegment_H
#define KVMSpaceLineSegment_H

#include "KThreeVector_KEMField.hh"

#include <cmath>
#include <string>


namespace KEMField
{

/**
*
*@file KVMSpaceLineSegment.hh
*@class KVMSpaceLineSegment
*@brief line segment in R^3, orientation is from point1 to point2
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul 11 11:17:42 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KVMSpaceLineSegment
{
  public:
    KVMSpaceLineSegment();
    ~KVMSpaceLineSegment()
    {
        ;
    };

    void SetAll(const double* point1, const double* point2);
    void SetVertices(const double* point1, const double* point2);
    void GetVertices(double* point1, double* point2) const;

    double GetPoint1(int i) const
    {
        return fP1[i];
    };
    double GetPoint2(int i) const
    {
        return fP2[i];
    };
    double GetLength() const
    {
        return fL;
    };
    double GetUnitVector(int i) const
    {
        return fN[i];
    };

    inline KVMSpaceLineSegment(const KVMSpaceLineSegment& copyObject);
    inline KVMSpaceLineSegment& operator=(const KVMSpaceLineSegment& rhs);

  protected:
    void InitializeParameters();

    void SetPoint1(const double* point);  //start point
    void SetPoint2(const double* point);  //end point

    KFieldVector fP1;  //start point
    KFieldVector fP2;  //end point
    KFieldVector fN;   //unit vector
    double fL;         //length
};

inline KVMSpaceLineSegment::KVMSpaceLineSegment(const KVMSpaceLineSegment& copyObject)
{
    fP1 = copyObject.fP1;
    fP2 = copyObject.fP2;
    InitializeParameters();
}

inline KVMSpaceLineSegment& KVMSpaceLineSegment::operator=(const KVMSpaceLineSegment& rhs)
{
    if (this != &rhs) {
        fP1 = rhs.fP1;
        fP2 = rhs.fP2;
        InitializeParameters();
    }
    return *this;
}


}  // namespace KEMField

#endif /* KVMSpaceLineSegment_H */
