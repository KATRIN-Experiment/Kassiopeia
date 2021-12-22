#ifndef __KG2DPolygonWithArcs_H__
#define __KG2DPolygonWithArcs_H__

#include "KG2DArc.hh"
#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DShape.hh"
#include "KGVertexSideDescriptor.hh"

#include "KTwoVector.hh"

#include <cmath>
#include <vector>

#define SMALLNUMBER 1e-9

namespace KGeoBag
{

/**
*
*@file KG2DPolygonWithArcs.hh
*@class KG2DPolygonWithArcs
*@brief  a 2d polygon class whose sides are made of line segments or arcs
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug  1 11:51:03 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KG2DPolygonWithArcs : public KG2DArea
{
  public:
    KG2DPolygonWithArcs();
    KG2DPolygonWithArcs(const KG2DPolygonWithArcs& copyObject);
    KG2DPolygonWithArcs(const std::vector<KGVertexSideDescriptor>* ordered_descriptors);
    ~KG2DPolygonWithArcs() override;

    ///create the polygon with arcs by setting the 'descriptors'
    ///a descriptor consists of a vertex and a description of the side
    ///that it starts, if IsArc is false then the side it starts will be a
    ///line segment. If IsArc is true then an arc with the specified radius
    ///and other properties will attempt to be created, if the parameters
    ///are not valid (i.e. radius is too small for the distance between the
    ///two vertices then it will be replaced by a line segments
    void SetDescriptors(const std::vector<KGVertexSideDescriptor>* ordered_descriptors);


    void Initialize() override;

    //getters
    void GetVertices(std::vector<katrin::KTwoVector>* vertices) const;
    void GetSides(std::vector<KG2DShape*>* sides) const;

    //****************
    //geometric system
    //****************

    void NearestDistance(const katrin::KTwoVector& aPoint, double& aDistance) const override;
    katrin::KTwoVector Point(const katrin::KTwoVector& aPoint) const override;
    katrin::KTwoVector Normal(const katrin::KTwoVector& aPoint) const override;
    void NearestIntersection(const katrin::KTwoVector& aStart, const katrin::KTwoVector& anEnd, bool& aResult,
                             katrin::KTwoVector& anIntersection) const override;

    ///returns true if point is inside the region enclosed by the polygon
    bool IsInside(const katrin::KTwoVector& point) const override;

    double Area() const override
    {
        return 0.;
    };  //not yet implemented

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


  protected:
    ///function used to test if the polygon is simple
    void DetermineIfPolygonIsSimpleAndValid();

    ///determines whether the interior lies to the left or right
    ///of the sides that make up the (base) polygon
    void DetermineInteriorSide();

    std::vector<KGVertexSideDescriptor> fDescriptors;  //an ordered list of the discriptors
    std::vector<katrin::KTwoVector> fVertices;                 //an ordered list of the polygon's vertices
    std::vector<KG2DShape*> fSides;                    //an ordered list of the polygon's sides

    bool fIsSimple;
    bool fIsValid;
    bool fIsLeft;
    int fNVertices;  //number of vertices, same as number of sides

    ///the base polygon, constructed of line segments only, no arcs
    KG2DPolygon fBasePolygon;

    ///vector of pointers to all of the arcs that modify the base polygon
    int fNArcs;
    std::vector<KG2DArc*> fArcs;
    std::vector<bool> fIsArcAdditive;
};


}  // namespace KGeoBag


#endif /* __KG2DPolygonWithArcs_H__ */
