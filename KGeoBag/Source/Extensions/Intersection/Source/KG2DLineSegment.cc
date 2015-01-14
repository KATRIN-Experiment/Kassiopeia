#include "KG2DLineSegment.hh"

#include <iostream>

namespace KGeoBag{

KG2DLineSegment::KG2DLineSegment():
fP1(0,0),
fP2(0,0),
fDiff(0,0),
fUnit(0,0),
fLength(0),
fLength2(0)
{;};


KG2DLineSegment::KG2DLineSegment(double point1[2], double point2[2]):
fP1(point1[0], point1[1]),
fP2(point2[0], point2[1])
{
    fDiff = fP2 - fP1;
    fUnit = fDiff;
    fLength = fUnit.Magnitude();
    fLength2 = fLength*fLength;
    fUnit = fUnit.Unit();
}


KG2DLineSegment::KG2DLineSegment(const KTwoVector& point1, const KTwoVector& point2):
fP1(point1),
fP2(point2)
{
    fDiff = fP2 - fP1;
    fUnit = fDiff;
    fLength = fUnit.Magnitude();
    fLength2 = fLength*fLength;
    fUnit = fUnit.Unit();
}

void
KG2DLineSegment::SetPoints(const KTwoVector& point1, const KTwoVector& point2)
{
    fP1 = point1;
    fP2 = point2;
    Initialize();
}

void
KG2DLineSegment::SetFirstPoint(const KTwoVector& point1)
{
    fP1 = point1;
    Initialize();
}

void
KG2DLineSegment::SetSecondPoint(const KTwoVector& point2)
{
    fP2 = point2;
    Initialize();
}

void
KG2DLineSegment::Initialize()
{
    fDiff = fP2 - fP1;
    fUnit = fDiff;
    fLength = fUnit.Magnitude();
    fLength2 = fLength*fLength;
    fUnit = fUnit.Unit();
}

void
KG2DLineSegment::NearestDistance( const KTwoVector& aPoint, double& aDistance ) const
{
    KTwoVector del = aPoint - fP1;
    double dot = del*fDiff/fLength2;
    if(dot < 0. ){aDistance = del.Magnitude(); return;}
    if(dot > 1.){aDistance = (aPoint - fP2).Magnitude(); return;}
    KTwoVector nearestpoint = fP1 + dot*fDiff;
    aDistance = (aPoint - nearestpoint).Magnitude();
}

KTwoVector
KG2DLineSegment::Point( const KTwoVector& aPoint ) const
{
  KTwoVector aNearest;
    KTwoVector del = aPoint - fP1;
    double dot = del*fDiff/fLength2;
    if(dot < 0. ){aNearest = fP1; return aNearest;}
    if(dot > 1.){aNearest = fP2; return aNearest;}
    aNearest = fP1 + dot*fDiff;
    return aNearest;
}

KTwoVector
KG2DLineSegment::Normal( const KTwoVector& aPoint ) const
{
  KTwoVector aNormal;
    //normal vector depends on which side of the line segment we are on
    //otherwise we have to pick a convention
    KTwoVector del = aPoint - fP1;
//    if( (del^fUnit) > 0)
//    {

      //normal always points in same direction (depends of the order of the points fP1, fP2)
      aNormal = KTwoVector(fUnit.Y(), -1.0*fUnit.X());


//    }
//    else
//    {
//        aNormal = KTwoVector(-1.0*fUnit.Y(), fUnit.X());
//    }
      return aNormal;
}


void
KG2DLineSegment::NearestIntersection( const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult, KTwoVector& anIntersection ) const
{
    //aStart is refered to as p3
    //anEnd is refered to as p4

//    KTwoVector p4_p3 = anEnd - aStart;
//    KTwoVector p1_p3 = fP1 - aStart;

//    double denom = p4_p3^fDiff;

    double x1,x2,x3,x4;
    double y1,y2,y3,y4;

    x1 = fP1.X();
    x2 = fP2.X();
    x3 = aStart.X();
    x4 = anEnd.X();

    y1 = fP1.Y();
    y2 = fP2.Y();
    y3 = aStart.Y();
    y4 = anEnd.Y();

    double denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2 - y1);
    double t, s;

    t = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3))/denom;
    s = ( (x2-x1)*(y1-y3) - (y2-y1)*(x1-x3) )/denom;


//    if( denom*denom > SMALLNUMBER*(p4_p3.MagnitudeSquared())*(fLength2))
    if(denom*denom > SMALLNUMBER)
    {
        //lines are not parallel, compute intersections
        //t = (p4_p3^p1_p3)/denom;
        if( t < 0. || 1. < t)
        {
            //intersection is not on this segment
            aResult = false;
            return;
        }

        //an intersection exists on this segment
        //now test the other one
        //s = (fDiff^p1_p3)/denom;

        if( s < 0. || 1. < s )
        {
            aResult = false;
            return;
        }

        //we have an intersection on both segments
        aResult = true;
        anIntersection = (fP1 + t*fDiff);
//        std::cout<<"lin seg 157: t = "<<t<<std::endl;
        return;
    }

    //if we've made it here then the segments are parallel
    //t = p1_p3^fP1;
//    if( t*t > SMALLNUMBER*fLength2*p1_p3.MagnitudeSquared() )
    if(t*t > SMALLNUMBER)
    {
        //lines are parallel but separated
        aResult = false;
        return;
    }

    //if we've made it here then the segments are on the same line
    //and we have to check if they overlap
    double p4dist = (anEnd - fP1)*fUnit;
    double p3dist = (aStart - fP1)*fUnit;
    bool isRight = false;
    if(p4dist > p3dist){isRight = true;}

    double overlap[2];
    int overlap_status;

    overlap_status = FindIntersection1D(0, fLength, p3dist, p4dist, overlap);

    if(overlap_status == 0)
    {
        //no intersection
        aResult = false;
        return;
    }
    else if(overlap_status == 1)
    {
        //intersection at endpoints, overlap[0] = 0 or fLength
        aResult = true;
        anIntersection = fP1 + overlap[0]*fUnit;
        return;
    }
    else if(overlap_status == 2)
    {
        //overlap, technically infinite number of intersections
        //but we will take the one that is closest to aStart

        KTwoVector inter1 = fP1 + overlap[0]*fUnit;
        KTwoVector inter2 = fP1 + overlap[1]*fUnit;
        double dist1 = (aStart - inter1).Magnitude();
        double dist2 = (aStart - inter2).Magnitude();

        aResult = true;
        if(dist1 < dist2)
        {
            anIntersection = inter1;
        }
        else
        {
            anIntersection = inter2;
        }
        return;
    }


    //should never reach here...probably ought to issue a warning or something
    //but if we get here lets say we didn't find anything
    aResult = false;
    return;

}

void KG2DLineSegment::NearestIntersection(const KG2DLineSegment& aSegment, bool& aResult, KTwoVector& anIntersection ) const
{
    NearestIntersection(aSegment.GetFirstPoint(), aSegment.GetSecondPoint(), aResult, anIntersection);
}


int
KG2DLineSegment::FindIntersection1D(double a, double b, double c, double d, double result[2])
{
    //looks for overlap between the intervals
    //[a,b] and [c,d]
    //a,b and c,d do not necessarily have to be ordered

    double arr[4];
    int index[4];

    arr[0] = a; index[0] = 0;
    arr[1] = b; index[1] = 0;
    arr[2] = c; index[2] = 1;
    arr[3] = d; index[3] = 1;

    if(arr[1] > arr[3]){Swap(arr[1], arr[3]); Swap(index[1], index[3]); };
    if(arr[0] > arr[2]){Swap(arr[0], arr[2]); Swap(index[0], index[2]); };
    if(arr[0] > arr[1]){Swap(arr[0], arr[1]); Swap(index[0], index[1]); };
    if(arr[2] > arr[3]){Swap(arr[2], arr[3]); Swap(index[2], index[3]); };
    if(arr[1] > arr[2]){Swap(arr[1], arr[2]); Swap(index[1], index[2]); };

    //now the values in arr should be sorted in increasing order
    //and the values in index should show which interval's end-points they belong to

    //if the values in index have the form:
    //0011 or 1100 then there is no overlap...although the end points may
    //just touch

    //if the values in the index have the form:
    // 1001, 0110, 0101, or 1010 then there is overlap and the overlap interval
    //is {arr[1], arr[2]}

    int sum;
    sum = index[0] + index[1];

    if( (sum ==0) || (sum == 2) )
    {
        //there is no overlap, but check how close the endpoints are
        if( arr[2] - arr[1] > SMALLNUMBER)
        {
            return 0;
        }
        else
        {
            //endpoints are within epsilon of each other
            //call this an intersection of 1 point
            result[0] = arr[1];
            return 1;
        }
    }
    else
    {
        //there is overlap, but check how big the overlap interval is
        if( arr[2] - arr[1] > SMALLNUMBER)
        {
            //there is a large overlap, return the interval of overlap
            result[0] = arr[1];
            result[1] = arr[2];
            return 2;
        }
        else
        {
            //there is no overlap but the
            //endpoints are within epsilon of each other
            //call this an intersection of 1 point
            result[0] = arr[1];
            return 1;
        }
    }
}



}//end of namespace
