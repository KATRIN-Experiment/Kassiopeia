#ifndef __KG2DArc_H__
#define __KG2DArc_H__


#include "KG2DShape.hh"

#include <cmath>
#include <vector>

#define SMALLNUMBER 1e-9

namespace KGeoBag
{


/**
*
*@file KG2DArc.hh
*@class KG2DArc
*@brief angles must be in radians
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 28 12:58:49 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KG2DArc : public KG2DShape
{
  public:
    KG2DArc()
    {
        ;
    };
    KG2DArc(const KTwoVector& point1, const KTwoVector& point2, const double& radius,
            bool isRight = true,  //center on the right of line segment
            bool isCCW = true);   //arc goes ccw if true

    KG2DArc(const KTwoVector& center, const double& radius, const double& start_angle,
            const double& end_angle);  //start angle and end angle will be fixed to [0,2*pi]

    KG2DArc(const KTwoVector& center, const KTwoVector& point1,
            const double& angle);  //angle must be in [-2pi, +2pi]

    ~KG2DArc() override
    {
        ;
    };

    //setters

    ///construct arc from two points, a radius and orientation
    ///arc goes counterclockwise if isCCW is true, clockwise if false
    ///center of the arc lies to the right of the directed line segment
    ///going from point1 to point2 if isRight is true, otherwise it is placed
    ///on the left
    void SetPointsRadiusOrientation(const KTwoVector& point1, const KTwoVector& point2, const double& radius,
                                    bool isRight = true, bool isCCW = true);

    ///construct arc from a center, radius, and two angles
    ///arc is formed going from start angle to end angle
    ///start angle and end angle will be fixed to range [0,2*pi]
    ///if start_angle is less than end_angle the arc will be counterclockwise
    ///if start_angle is greater than end angle the arc will be clockwise
    void SetCenterRadiusAngles(const KTwoVector& center, const double& radius, const double& start_angle,
                               const double& end_angle);

    ///construct from a center, start point and an angle
    ///angle will be clamped between [-2pi, +2pi]
    ///negative angles indicate clockwise directionality
    ///positive angles indicate counterclockwise directionality
    void SetStartPointCenterAngle(const KTwoVector& center, const KTwoVector& point1, const double& angle);

    //initialization
    void Initialize() override;

    //getters
    double GetRadius() const
    {
        return fRadius;
    };
    double GetAngleSubtended() const
    {
        return fSubtendedAngle;
    };
    KTwoVector GetFirstPoint() const
    {
        return fP1;
    };
    KTwoVector GetSecondPoint() const
    {
        return fP2;
    };
    double GetAngleFirstPoint() const
    {
        return fA1;
    };
    double GetAngleSecondPoint() const
    {
        return fA2;
    };
    KTwoVector GetCenter() const
    {
        return fCenter;
    };
    bool IsCCW() const
    {
        return fIsCCW;
    };

    //geometry utilities
    void NearestDistance(const KTwoVector& aPoint, double& aDistance) const override;
    KTwoVector Point(const KTwoVector& aPoint) const override;
    KTwoVector Normal(const KTwoVector& aPoint) const override;

    void NearestIntersection(const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult,
                             KTwoVector& anIntersection) const override;

    ///computes the intersection with another arc
    ///flag indicates the various possibilities
    ///flag = 0 indicates there are not intersections
    ///flag = 1 indicates there is one intersection
    ///flag = 2 indicates there are two intersection
    ///flag = 3 indicates sections of the arcs are coincident
    ///so there are an inifinite number of intersections
    void NearestIntersection(const KG2DArc* arc, int& flag, std::vector<KTwoVector>* intersections) const;

    ///tests if a given point is inside the section of the circle
    ///that is defined by the arc and its chord
    bool IsInsideCircularSegment(const KTwoVector aPoint) const;

    ///tests if a given point is inside the wedge defined by
    ///the arc and the center of the circle
    bool IsInsideWedge(const KTwoVector aPoint) const;

    ///tests if a given point lies within the angles defined by the arc
    bool IsInAngularRange(const KTwoVector& aPoint) const;

    ///tests if a given angle is between angle1 and angle2
    ///assumes that angles are between [0,2pi]
    bool IsInAngularRange(const double& angle1, const double& angle2, const double& test) const;

    ///tests if a point is on the same side of the chord that the arc is on
    bool IsSameSideOfChordAsArc(const KTwoVector aPoint) const;

    int DoArcsOverlap(const KG2DArc* aArc) const;

  protected:
    bool fIsCCW;

    KTwoVector fP1;            //start point of arc
    KTwoVector fP2;            //end point of arc
    KTwoVector fCenter;        //center of circle arc lies on
    KTwoVector fChordUnit;     //unit vector pointing from fP1 to fP2
    KTwoVector fHalfwayPoint;  //point halfway between fP1 and fP2
    KTwoVector fBisector;      //vector which points from circle center to middle of arc

    double fHalfSubtendedAngle;
    double fCosHalfSubtendedAngle;
    double fSubtendedAngle;
    double fRadius;
    double fRadius2;
    double fA1;  //angle of point1;
    double fA2;  //angle of point2;
};


}  // namespace KGeoBag

#endif /* __KG2DArc_H__ */
