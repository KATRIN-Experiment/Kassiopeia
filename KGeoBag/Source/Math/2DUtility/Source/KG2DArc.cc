#include "KG2DArc.hh"

#include "KThreeVector.hh"

#include <iostream>

namespace KGeoBag{

KG2DArc::KG2DArc(const KTwoVector& point1,
                 const KTwoVector& point2,
                 const double& radius,
                 bool isRight,
                 bool isCCW)
{
    SetPointsRadiusOrientation(point1, point2, radius, isRight, isCCW);
}

KG2DArc::KG2DArc(const KTwoVector& center,
                 const double& radius,
                 const double& start_angle,
                 const double& end_angle)
{
    SetCenterRadiusAngles(center, radius, start_angle, end_angle);
}

KG2DArc::KG2DArc(const KTwoVector& center,
                 const KTwoVector& point1,
                 const double& angle)
{
    SetStartPointCenterAngle(center, point1, angle);
}


void
KG2DArc::SetPointsRadiusOrientation(const KTwoVector& point1,
                                    const KTwoVector& point2,
                                    const double& radius,
                                    bool isRight,
                                    bool isCCW)
{

    //check if radius given is big enough for a solution
    if( (point1 - point2).Magnitude() <= 2*radius)
    {
        //given just two points and a radius there are 4 possible arcs that
        //can be constructed, we will use the ordering of the the given points
        //to reduce that number down to 2, while the final selection is determined
        //by the orientation of the curve (ccw or cw)

        fP1 = point1;
        fP2 = point2;
        fRadius = radius;
        fRadius2 = fRadius*fRadius;
        fIsCCW = isCCW;

        //vector which points from p1 to p2, chord, and other properties
        KTwoVector del = fP2 - fP1;
        fChordUnit = del.Unit();
        double len = del.Magnitude();
        double half_len = len/2.0;
        double sagitta = fRadius - std::sqrt(fRadius*fRadius - half_len*half_len);
        double apothem = fRadius - sagitta;

        //figure out which side the center is on, cheating with 3-vectors
        KThreeVector z(0.,0.,1.);
        KThreeVector n;
        n.SetComponents(fChordUnit.X(), fChordUnit.Y(), 0.);

        //vector which points from center of chord towards center
        //currently it on the right of the directed line segment
        //pointing from p1 to p2
        KThreeVector r = z.Cross(n);

        if(isRight)
        {
            //reverse the direction so that the center lies on the right now
            r *= -1.0;
        }

        //compute the center point
        KTwoVector dir_to_center( r.X(), r.Y() );
        dir_to_center = dir_to_center.Unit();
        fHalfwayPoint = fP1 + fChordUnit*half_len;
        fCenter = fHalfwayPoint + apothem*dir_to_center;

        fA1 = (fP1 - fCenter).PolarAngle();
        fA1 = Limit_to_0_to_2pi(fA1);

        fA2 = (fP2 -fCenter).PolarAngle();
        fA2 = Limit_to_0_to_2pi(fA2);

        //now that we have the center point
        //we need to figure out which of the two possible arcs
        //we want to keep. This depends on if we are CCW or CW

        if(fIsCCW)
        {
            if(fA1 < fA2)
            {
                fSubtendedAngle = fA2 - fA1;
                fHalfSubtendedAngle = fSubtendedAngle/2.0;
                fCosHalfSubtendedAngle = std::cos(fHalfSubtendedAngle);
                fBisector = (fP1 - fCenter) + (fP2 - fCenter);
                //fBisector = (fP1-fCenter).Rotate(fHalfSubtendedAngle);
                //fBisector = Rotate_vector_by_angle(fP1-fCenter, fHalfSubtendedAngle);
                fBisector = fBisector.Unit();
                if(isRight)
                {
                    fBisector = -1*fBisector;
                }
            }
            else
            {
                fSubtendedAngle = fA2 + (2.0*M_PI - fA1);
                fHalfSubtendedAngle = fSubtendedAngle/2.0;
                fCosHalfSubtendedAngle = std::cos(fHalfSubtendedAngle);
                fBisector = (fP1 - fCenter) + (fP2 - fCenter);
                //fBisector = (fP1-fCenter).Rotate(fHalfSubtendedAngle);
                //fBisector = Rotate_vector_by_angle(fP1-fCenter, fHalfSubtendedAngle);
                fBisector = fBisector.Unit();
                if(isRight)
                {
                    fBisector = -1*fBisector;
                }
            }
        }
        else
        {
            if(fA1 < fA2)
            {
                fSubtendedAngle = fA1 + (2.0*M_PI - fA2);
                fHalfSubtendedAngle = fSubtendedAngle/2.0;
                fCosHalfSubtendedAngle = std::cos(fHalfSubtendedAngle);
                fBisector = (fP1 - fCenter) + (fP2 - fCenter);
                //fBisector = (fP1-fCenter).Rotate(-1.0*fHalfSubtendedAngle);
                //fBisector = Rotate_vector_by_angle(fP1-fCenter, -1.0*fHalfSubtendedAngle);
                fBisector = fBisector.Unit();
                if(!isRight)
                {
                    fBisector = -1*fBisector;
                }
            }
            else
            {
                fSubtendedAngle = fA1 - fA2;
                fHalfSubtendedAngle = fSubtendedAngle/2.0;
                fCosHalfSubtendedAngle = std::cos(fHalfSubtendedAngle);
                fBisector = (fP1 - fCenter) + (fP2 - fCenter);
                //fBisector = (fP1-fCenter).Rotate(-1.0*fHalfSubtendedAngle);
                //fBisector = Rotate_vector_by_angle(fP1-fCenter, -1.0*fHalfSubtendedAngle);
                fBisector = fBisector.Unit();
                if(!isRight)
                {
                    fBisector = -1*fBisector;
                }
            }
        }
    }
    else
    {
        //radius too small for a solution
        std::cout<<"KG2DArc::SetPointsRadiusOrientation: Warning! Given radius too small for a valid solution. Aborting, object will not be modified."<<std::endl;
        std::cout<<"Diameter = "<<2.0*radius<<std::endl;
        std::cout<<"Point 1 = ("<<point1[0]<<", "<<point1[1]<<")"<<std::endl;
        std::cout<<"Point 2 = ("<<point2[0]<<", "<<point2[1]<<")"<<std::endl;
        std::cout<<"Point separation = "<<(point1 - point2).Magnitude()<<std::endl;

    }
}

void
KG2DArc::SetCenterRadiusAngles(const KTwoVector& center,
                               const double& radius,
                               const double& start_angle,
                               const double& end_angle)
{
    double phi1 = Limit_to_0_to_2pi(start_angle);
    double phi2 = Limit_to_0_to_2pi(end_angle);

    fCenter = center;

    KTwoVector p1( fCenter.X() + radius*std::cos(start_angle),
                           fCenter.Y() + radius*std::sin(start_angle) );

    SetStartPointCenterAngle(center, p1, phi2-phi1);
}

void
KG2DArc::SetStartPointCenterAngle(const KTwoVector& center,
                                  const KTwoVector& point1,
                                  const double& angle)
{
    fP1 = point1;
    fCenter = center;

    KTwoVector del = fP1 - fCenter;
    fRadius = del.Magnitude();
    fRadius2 = fRadius*fRadius;

    double phi = angle;
    if(angle > 2.0*M_PI ){phi = 2.0*M_PI;};
    if(angle < -2.0*M_PI ){phi = 2.0*M_PI;};

    if(phi >= 0){fIsCCW = true;}
    else{fIsCCW = false;}

    //fP2 = fCenter + del.Rotate(phi);
    fP2 = fCenter + Rotate_vector_by_angle(del, phi);


    fSubtendedAngle = std::fabs(phi);
    fHalfSubtendedAngle = fSubtendedAngle/2.0;
    fCosHalfSubtendedAngle = std::cos(fHalfSubtendedAngle);

    fHalfwayPoint = 0.5*(fP1 + fP2);
    fChordUnit = (fP2 - fP1).Unit();
    //fBisector = (fP1-fCenter).Rotate(phi/2.0);
    fBisector = Rotate_vector_by_angle(fP1-fCenter, phi/2.0);

    fA1 = (fP1 - fCenter).PolarAngle();
    fA1 = Limit_to_0_to_2pi(fA1);

    fA2 = (fP2 -fCenter).PolarAngle();
    fA2 = Limit_to_0_to_2pi(fA2);

}


void KG2DArc::Initialize()
{
    //all done in setters
}


void KG2DArc::NearestDistance( const KTwoVector& aPoint, double& aDistance ) const
{
    KTwoVector near;
    near = Point(aPoint);
    aDistance = (aPoint - near).Magnitude();
}

  KTwoVector KG2DArc::Point( const KTwoVector& aPoint ) const
{
  KTwoVector aNearest;
    KTwoVector del = (aPoint - fCenter);
    KTwoVector u;
    u = del.Unit();

    KTwoVector proj_on_circle = fCenter + fRadius*u;

    if( u*fBisector > fCosHalfSubtendedAngle )
    {
        aNearest = proj_on_circle;
        return aNearest;
    }
    else
    {
        double dist2_to_p1 = (aPoint - fP1).Magnitude();
        double dist2_to_p2 = (aPoint - fP2).Magnitude();

        if(dist2_to_p1 < dist2_to_p2)
        {
            aNearest = fP1;
            return aNearest;
        }
        else
        {
            aNearest = fP2;
            return aNearest;
        }
    }
    return aNearest;
}

  KTwoVector KG2DArc::Normal( const KTwoVector& aPoint ) const
{
  KTwoVector aNormal;
    KTwoVector del = aPoint - fCenter;

    if( IsInAngularRange(aPoint) )
    {
        if(del.MagnitudeSquared() > fRadius2)
        {
            aNormal = del.Unit();
        }
        else
        {
            aNormal = -1.0*del.Unit();
        }
        return aNormal;
    }

    //note that the nearest normal is not defined if the closest points
    //are the end points, so just return the vector pointing from the center to the end point

    KTwoVector u = del.Unit();
    KTwoVector proj_on_circle = fRadius*u;

    double dist2_to_p1 = (proj_on_circle - fP1).MagnitudeSquared();
    double dist2_to_p2 = (proj_on_circle - fP2).MagnitudeSquared();

    if(dist2_to_p1 < dist2_to_p2)
    {
        //aNormal = (aPoint - fP1).Unit();
        aNormal = (fP1 - fCenter).Unit();
        return aNormal;
    }
    else
    {
        //aNormal = (aPoint - fP2).Unit();
        aNormal = (fP2 - fCenter).Unit();
        return aNormal;
    }
}

void KG2DArc::NearestIntersection( const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult, KTwoVector& anIntersection ) const
{
    KTwoVector seg = anEnd - aStart;
    KTwoVector diff = aStart - fCenter;
    double len2 = seg.MagnitudeSquared();

    double a = len2;
    double b = 2.0*(diff*seg);
    double c = diff*diff - fRadius2;
    double disc = b*b - 4.0*a*c;

    //if the discriminant is less than zero there is no intersection with the circle at all
    if(disc < 0)
    {
        aResult = false;
        return;
    }

    double t1 = (-1.0*b + std::sqrt(disc))/(2.0*a);
    double t2 = (-1.0*b - std::sqrt(disc))/(2.0*a);

    bool t1_valid = !( (t1 < 0.0) || (1.0 < t1) );
    bool t2_valid = !( (t2 < 0.0) || (1.0 < t2) );


    if( !t1_valid && !t2_valid )
    {
        //both intersections are not on line segment
        aResult = false;
        return;
    }

    if( !t1_valid && t2_valid )
    {
        //only t2 is on the line segment
        KTwoVector p = aStart + t2*seg;
        //check if p is on the arc
        if( IsInAngularRange(p) )
        {
            //its on the arc and its the only solution
            aResult = true;
            anIntersection = p;
            return;
        }
        else
        {
            //its not on the arc, and there are no other solutions
            aResult = false;
            return;
        }
    }

    if( t1_valid && !t2_valid )
    {
        //only t1 is on the line segment
        KTwoVector p = aStart + t1*seg;
        //check if p is on the arc
        if( IsInAngularRange(p) )
        {
            //its on the arc and its the only solution
            aResult = true;
            anIntersection = p;
            return;
        }
        else
        {
            //its not on the arc, and there are no other solutions
            aResult = false;
            return;
        }
    }

    if(t1_valid && t2_valid)
    {
        //both t1 and t2 are on the line segment
        KTwoVector p1 = aStart + t1*seg;
        KTwoVector p2 = aStart + t2*seg;

        bool p1_on_arc = IsInAngularRange(p1);
        bool p2_on_arc = IsInAngularRange(p2);

        if( !p1_on_arc && !p2_on_arc)
        {
            //neither is on the arc
            aResult = false;
            return;
        }
        else if( !p2_on_arc && p1_on_arc)
        {
            //only p1 is on the arc
            aResult = true;
            anIntersection = p1;
            return;
        }
        else if( !p1_on_arc && p2_on_arc)
        {
            //only p2 is on the arc
            aResult = true;
            anIntersection = p2;
            return;
        }
        else if( p1_on_arc && p2_on_arc )
        {
            //both points are on the line segment and the arc
            //figure out which one is closest to aStart
            if( t1 < t2 )
            {
                aResult = true;
                anIntersection = p1;
                return;
            }
            else
            {
                aResult = true;
                anIntersection = p2;
                return;
            }
        }
    }


    //should never reach here, but if we do lets say we didn't find anything
    aResult = false;
    return;

}

void
KG2DArc::NearestIntersection( const KG2DArc* arc, int& flag, std::vector<KTwoVector>* intersections) const
{
    intersections->clear();
    //we refer to the circle that contains the arc under test as c2
    //while this is c1

    //see if the circles of the arcs are close enough to intersect
    KTwoVector c = arc->GetCenter();
    double r = arc->GetRadius();
    KTwoVector del = c - fCenter;
    double d = del.Magnitude();

    if(d > r + fRadius)
    {
        //too far apart to intersect
        flag = 0;
        return;
    }

    if(d < std::fabs(r - fRadius))
    {
        //one circle inside the other
        flag = 0;
        return;
    }

    if( (d < SMALLNUMBER) && (std::fabs(r - fRadius) < SMALLNUMBER) )
    {
        switch(DoArcsOverlap(arc))
        {
            case 0:
                flag = 0; return; //no overlap, no intersection
            break;
            case -1:
                flag = -1; return; //substantial overlap, no well determined intersection point
            break;
            case -2:
                flag = -2; //both endpoints just touch
                intersections->push_back(fP1);
                intersections->push_back(fP2);
                return;
            break;
            case -3:
                flag = -3; //one pair of endpoints just touch
                intersections->push_back(fP1);
                return;
            break;
            case -4:
                flag = -4; //one pair of endpoints just touch
                intersections->push_back(fP2);
                return;
            break;
            case -5:
                flag = -5; //one pair of endpoints just touch
                intersections->push_back(fP1);
                return;
            break;
            case -6:
                flag = -6; //one pair of endpoints just touch
                intersections->push_back(fP2);
                return;
            break;
        }
    }

    //distance and radii are in the right range, so compute the intersections
    double a = (fRadius2 - r*r + d*d)/(2.0*d);
    double h = std::sqrt(fRadius2 - a*a);

    KTwoVector u = del*(1.0/d); //unit vector pointing from c1 to c2
    KThreeVector uprime(u.X(), u.Y(), 0.);
    KThreeVector z(0.,0.,1.);
    KThreeVector vprime = z.Cross(uprime);
    KTwoVector v(vprime.X(), vprime.Y());

    KTwoVector p1 = fCenter + a*u + h*v;
    KTwoVector p2 = fCenter + a*u - h*v;

    //now we have to check if these points are on both the arcs

    if( IsInAngularRange(p1) && arc->IsInAngularRange(p1) )
    {
        intersections->push_back(p1);
    }

    if( IsInAngularRange(p2) && arc->IsInAngularRange(p2) )
    {
        intersections->push_back(p2);
    }

    flag = intersections->size();

}


bool
KG2DArc::IsInsideCircularSegment(const KTwoVector aPoint) const
{
    KTwoVector del = aPoint - fCenter;
    if( del.MagnitudeSquared() > fRadius2 ){ return false;}; //not even inside circle

    //we are inside the circle, now need to figure out if we are on the proper
    //side of the chord
    return IsSameSideOfChordAsArc(aPoint);

}


bool
KG2DArc::IsInsideWedge(const KTwoVector aPoint) const
{
    KTwoVector del = aPoint - fCenter;
    double MagnitudeSquared = del.MagnitudeSquared();
    if( MagnitudeSquared > fRadius2 ){return false;}; //not even inside circle

    double phi = del.PolarAngle();
    phi = Limit_to_0_to_2pi(phi);

    //return IsInAngularRange(fA1, fA2, phi);
    return IsInAngularRange(aPoint);


    //we are inside the circle, now need to figure out if we are in the right
    //range of angles
//    double dot = del*fBisector;
//    if( dot*dot < MagnitudeSquared*fCosHalfSubtendedAngle2 )
//    {
//        return true;
//    }
//    else
//    {
//        return false;
//    }
}


bool
KG2DArc::IsInAngularRange(const KTwoVector& aPoint) const
{
    KTwoVector del = aPoint - fCenter;
    double phi = del.PolarAngle();
    phi = Limit_to_0_to_2pi(phi);

    return IsInAngularRange(fA1, fA2, phi);

//    KTwoVector del = aPoint - fCenter;
//   // double MagnitudeSquared = del.MagnitudeSquared();
//    double dot = (del.Unit())*(fBisector.Unit());
//    if( dot*dot < fCosHalfSubtendedAngle*fCosHalfSubtendedAngle )
//    {
//        if(dot > 0)
//        {
//            return true;
//        }
//        else
//        {
//            return false;
//        }
//    }
//    else
//    {
//        return false;
//    }

//    double dot = del*fBisector;
//    if( dot*dot < MagnitudeSquared*fCosHalfSubtendedAngle*fCosHalfSubtendedAngle )
//    {
//        return true;
//    }
//    else
//    {
//        return false;
//    }

}

bool
KG2DArc::IsInAngularRange(const double& angle1, const double& angle2, const double& test) const
{
    if (angle1 < angle2)
    {
        if(fIsCCW)
        {
            return ( (angle1 <= test) && (test <= angle2 ) );
        }
        else
        {
            return ( (test <= angle1) || (angle2 <= test) );
        }
    }
    else
    {
        if(fIsCCW)
        {
            return ( (angle1 <= test) || (test <= angle2) );
        }
        else
        {
            return ( (angle2 <= test) && (test <= angle1 ) );
        }
    }
}

bool
KG2DArc::IsSameSideOfChordAsArc(const KTwoVector aPoint) const
{
    KTwoVector test = aPoint - fHalfwayPoint;
//    std::cout<<" point = "<<aPoint.X()<<", "<<aPoint.Y()<<std::endl;
//    std::cout<<"half = "<<fHalfwayPoint.X()<<", "<<fHalfwayPoint.Y()<<std::endl;
//    std::cout<<" test = "<<test.X()<<", "<<test.Y()<<std::endl;
//    std::cout<<"dot = "<<test*fBisector<<std::endl;
    if(test*fBisector >= 0 )
    {
        return true;
    }
    else
    {
        return false;
    }
}

int
KG2DArc::DoArcsOverlap(const KG2DArc* aArc) const
{

    double b1 = aArc->GetAngleFirstPoint();
    double b2 = aArc->GetAngleSecondPoint();

    if( IsInAngularRange(fA1, b1, fA2) || IsInAngularRange(fA1, b2, fA2) )
    {
        //there is some overlap, but the end points might be very close
        //so we need to check this
        if( aArc->IsCCW() == fIsCCW )
        {
            //both ccw or cw
            double delta1 = (fP1 - aArc->GetSecondPoint() ).Magnitude();
            double delta2 = (fP2 - aArc->GetFirstPoint() ).Magnitude();

            if(delta1 > SMALLNUMBER && delta2 > SMALLNUMBER)
            {
                return -1; //endpoints are not close and they do overlap
            }
            else if( delta1 < SMALLNUMBER && delta2 < SMALLNUMBER)
            {
                return -2; //both endpoints just touch
            }
            else if( delta1 < SMALLNUMBER)
            {
                return -3; //point1 of this arc and point2 of the test arc touch
            }
            else
            {
                return -4;//point2 of this arc and point1 of the test arc touch
            }
        }
        else
        {
            //one is ccw, the other is cw
            double delta1 = (fP1 - aArc->GetFirstPoint() ).Magnitude();
            double delta2 = (fP2 - aArc->GetSecondPoint() ).Magnitude();

            if(delta1 > SMALLNUMBER && delta2 > SMALLNUMBER)
            {
                return -1; //endpoints are not close and they do overlap
            }
            else if( delta1 < SMALLNUMBER && delta2 < SMALLNUMBER)
            {
                return -2; //both endpoints just touch
            }
            else if( delta1 < SMALLNUMBER)
            {
                return -5; //point1 of this arc and point1 of the test arc touch
            }
            else
            {
                return -6;//point2 of this arc and point2 of the test arc touch
            }
        }

    }
    else
    {
        //there is no overlap, but the end points might be very close
        //so we need to check this
        if( aArc->IsCCW() == fIsCCW )
        {
            //both ccw or cw
            double delta1 = (fP1 - aArc->GetSecondPoint() ).Magnitude();
            double delta2 = (fP2 - aArc->GetFirstPoint() ).Magnitude();

            if(delta1 > SMALLNUMBER && delta2 > SMALLNUMBER)
            {
                return 0; //endpoints are not close no overlap
            }
            else if( delta1 < SMALLNUMBER && delta2 < SMALLNUMBER)
            {
                return -2; //both endpoints just touch
            }
            else if( delta1 < SMALLNUMBER)
            {
                return -3; //point1 of this arc and point2 of the test arc touch
            }
            else
            {
                return -4;//point2 of this arc and point1 of the test arc touch
            }
        }
        else
        {
            //one is ccw, the other is cw
            double delta1 = (fP1 - aArc->GetFirstPoint() ).Magnitude();
            double delta2 = (fP2 - aArc->GetSecondPoint() ).Magnitude();

            if(delta1 > SMALLNUMBER && delta2 > SMALLNUMBER)
            {
                return 0; //endpoints are not close no overlap
            }
            else if( delta1 < SMALLNUMBER && delta2 < SMALLNUMBER)
            {
                return -2; //both endpoints just touch
            }
            else if( delta1 < SMALLNUMBER)
            {
                return -5; //point1 of this arc and point1 of the test arc touch
            }
            else
            {
                return -6;//point2 of this arc and point2 of the test arc touch
            }
        }
    }
}







}//end of kgeobag namespace
