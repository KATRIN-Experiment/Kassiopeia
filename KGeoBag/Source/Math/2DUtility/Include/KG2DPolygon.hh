#ifndef __KG2DPolygon_H__
#define __KG2DPolygon_H__

#include "KG2DArea.hh"
#include "KG2DLineSegment.hh"
#include "KTwoVector.hh"

#include <cmath>
#include <vector>

#define SMALLNUMBER 1e-9

namespace KGeoBag
{

/**
*
*@file KG2DPolygon.hh
*@class KG2DPolygon
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 28 17:34:28 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*July 2013: AreaMeasure added by Andreas Mueller
*
*/

class KG2DPolygon : public KG2DArea
{
  public:
    KG2DPolygon();
    KG2DPolygon(const KG2DPolygon& copyObject);
    KG2DPolygon(const std::vector<std::vector<double>>* ordered_vertices);
    KG2DPolygon(const std::vector<KTwoVector>* ordered_vertices);

    ~KG2DPolygon() override;

    ///create the polygon by setting the vertices
    ///sides are created from the vertices in a 'connect the dots' manner.
    ///The last point in the list is connected to the
    ///first by the final edge to close the polygon.
    ///There is no need to repeat the first vertex.
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

    ///returns true if point is inside the region enclosed by the polygon
    bool IsInside(const KTwoVector& point) const override;

    double Area() const override;

    ///returns true if polygon has no self intersections
    virtual bool IsSimple() const
    {
        return fIsSimple;
    };

    ///returns true it the polygon has been constructed properly
    ///must be checked before calling other functions otherwise
    ///results cannot be guaranteed to be correct
    virtual bool IsValid() const
    {
        return fIsValid;
    };

    //static utility functions for navigation
    static double NearestDistance(const std::vector<KTwoVector>* ordered_vertices, const KTwoVector& aPoint);

    static KTwoVector NearestPoint(const std::vector<KTwoVector>* ordered_vertices, const KTwoVector& aPoint);

    static KTwoVector NearestNormal(const std::vector<KTwoVector>* ordered_vertices, const KTwoVector& aPoint);

    static bool NearestIntersection(const std::vector<KTwoVector>* ordered_vertices, const KTwoVector& aStart,
                                    const KTwoVector& anEnd, KTwoVector& anIntersection);

    static bool IsInside(const std::vector<KTwoVector>* ordered_vertices, const KTwoVector& point);


    ///use our own modulus function, because c++'s is all screwy
    ///and implementation dependent when negative numbers are involved
    static int Modulus(int arg, int n);


  protected:
    //function used to test if the polygon is simple
    void DetermineIfPolygonIsSimple();

    ///determines whether the interior lies to the left or right
    ///of the sides that make up the (base) polygon
    void DetermineInteriorSide();

    std::vector<KTwoVector> fVertices;    //an ordered list of the polygon's vertices
    std::vector<KG2DLineSegment> fSides;  //an ordered list of the polygon's sides

    bool fIsSimple;
    bool fIsValid;
    bool fIsLeft;
    int fNVertices;  //number of vertices, same as number of sides

    //scratch space for point in polygon test
    mutable std::vector<KTwoVector> fDiff;
};

}  // namespace KGeoBag


#endif /* __KG2DPolygon_H__ */
