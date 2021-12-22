#ifndef __KG2DLineSegment_H__
#define __KG2DLineSegment_H__

#include "KG2DShape.hh"

#include "KTwoVector.hh"

#define SMALLNUMBER 1e-9

namespace KGeoBag
{

/**
*
*@file KG2DLineSegment.hh
*@class KG2DLineSegment
*@brief utility class for 2d line segments
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul 27 16:58:02 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KG2DLineSegment : public KG2DShape
{
  public:
    KG2DLineSegment();
    KG2DLineSegment(double point1[2], double point2[2]);
    KG2DLineSegment(const katrin::KTwoVector& point1, const katrin::KTwoVector& point2);

    ~KG2DLineSegment() override
    {
        ;
    };

    //setters
    void SetPoints(const katrin::KTwoVector& point1, const katrin::KTwoVector& point2);
    void SetFirstPoint(const katrin::KTwoVector& point1);
    void SetSecondPoint(const katrin::KTwoVector& point2);
    void Initialize() override;

    //getters
    double GetLength() const
    {
        return fLength;
    };
    katrin::KTwoVector GetFirstPoint() const
    {
        return fP1;
    };
    katrin::KTwoVector GetSecondPoint() const
    {
        return fP2;
    };
    katrin::KTwoVector GetUnitVector() const
    {
        return fUnit;
    };

    //geometry utilities
    void NearestDistance(const katrin::KTwoVector& aPoint, double& aDistance) const override;
    katrin::KTwoVector Point(const katrin::KTwoVector& aPoint) const override;
    katrin::KTwoVector Normal(const katrin::KTwoVector& /*aPoint*/) const override;
    void NearestIntersection(const katrin::KTwoVector& aStart, const katrin::KTwoVector& anEnd, bool& aResult,
                             katrin::KTwoVector& anIntersection) const override;

    void NearestIntersection(const KG2DLineSegment& aSegment, bool& aResult, katrin::KTwoVector& anIntersection) const;

    //static helper function
    static int FindIntersection1D(double a, double b, double c, double d, double result[2]);

    static void Swap(double& val1, double& val2)
    {
        double temp;
        temp = val1;
        val1 = val2;
        val2 = temp;
    }

    static void Swap(int& val1, int& val2)
    {
        int temp;
        temp = val1;
        val1 = val2;
        val2 = temp;
    }

    //static utility functions for navigation
    static double NearestDistance(const katrin::KTwoVector& p0, const katrin::KTwoVector& p1,  //(p0,p1) describe 2d line sgment
                                  const katrin::KTwoVector& aPoint);                   // point to be tested against

    static katrin::KTwoVector NearestPoint(const katrin::KTwoVector& p0, const katrin::KTwoVector& p1,  //(p0,p1) describe 2d line sgment
                                   const katrin::KTwoVector& aPoint);                   // point to be tested against

    static katrin::KTwoVector NearestNormal(const katrin::KTwoVector& p0, const katrin::KTwoVector& p1,  //(p0,p1) describe 2d line sgment
                                    const katrin::KTwoVector& /*aPoint*/);               // point to be tested against

    static bool NearestIntersection(
        const katrin::KTwoVector& p0, const katrin::KTwoVector& p1,         //(p0,p1) describe 2d line sgment
        const katrin::KTwoVector& aStart, const katrin::KTwoVector& anEnd,  //line segment to be tested against
        katrin::KTwoVector& anIntersection);  //if return value is true, intersection point will be returned here


  protected:
    katrin::KTwoVector fP1;
    katrin::KTwoVector fP2;
    katrin::KTwoVector fDiff;
    katrin::KTwoVector fUnit;
    double fLength;
    double fLength2;
};

}  // namespace KGeoBag

#endif /* __KG2DLineSegment_H__ */
