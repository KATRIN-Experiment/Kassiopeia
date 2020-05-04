#ifndef __KG2DPolyLine_H__
#define __KG2DPolyLine_H__

#include "KG2DLineSegment.hh"
#include "KG2DShape.hh"
#include "KTwoVector.hh"

#include <cmath>
#include <vector>

#define SMALLNUMBER 1e-9

namespace KGeoBag
{

/**
*
*@file KG2DPolyLine.hh
*@class KG2DPolyLine
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 28 17:34:28 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KG2DPolyLine : public KG2DShape
{
  public:
    KG2DPolyLine();
    KG2DPolyLine(const std::vector<std::vector<double>>* ordered_vertices);
    KG2DPolyLine(const std::vector<KTwoVector>* ordered_vertices);

    ~KG2DPolyLine() override;

    ///create the polyline by setting the vertices
    ///sides are created from the vertices in a 'connect the dots' manner.
    void SetVertices(const std::vector<std::vector<double>>* ordered_vertices);
    void SetVertices(const std::vector<KTwoVector>* ordered_vertices);
    void Initialize() override;

    //getters
    void GetVertices(std::vector<KTwoVector>* vertices) const;
    void GetSides(std::vector<KG2DLineSegment>* sides) const;

    //geometry utilities
    void NearestDistance(const KTwoVector& aPoint, double& aDistance) const override;
    KTwoVector Point(const KTwoVector& aPoint) const override;
    KTwoVector Normal(const KTwoVector& aPoint) const override;
    void NearestIntersection(const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult,
                             KTwoVector& anIntersection) const override;

    ///returns true if polyline has no self intersections
    virtual bool IsSimple() const
    {
        return fIsSimple;
    };

    ///returns true it the polyline has been constructed properly
    ///must be checked before calling other functions otherwise
    ///results cannot be guaranteed to be correct
    virtual bool IsValid() const
    {
        return fIsValid;
    };

  protected:
    //function used to test if the polyline is simple
    void DetermineIfPolyLineIsSimple();

    std::vector<KTwoVector> fVertices;    //an ordered list of the polyline's vertices
    std::vector<KG2DLineSegment> fSides;  //an ordered list of the polyline's sides

    bool fIsSimple;
    bool fIsValid;
    int fNVertices;  //number of vertices
    int fNSides;     //number of sides, one less than the number of vertices

    //scratch space for point in polygon test
    mutable std::vector<KTwoVector> fDiff;
};

}  // namespace KGeoBag


#endif /* __KG2DPolyLine_H__ */
