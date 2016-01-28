#include "KG2DPolyLine.hh"

#include <iostream>

namespace KGeoBag
{


KG2DPolyLine::KG2DPolyLine():
fIsValid(false)
{;}


KG2DPolyLine::KG2DPolyLine(const std::vector< std::vector<double> >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( KTwoVector( (*ordered_vertices)[i][0], (*ordered_vertices)[i][1] ) );
    }
    Initialize();
}

KG2DPolyLine::KG2DPolyLine(const std::vector< KTwoVector >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( (*ordered_vertices)[i] );
    }
    Initialize();
}

KG2DPolyLine::~KG2DPolyLine(){;}

void
KG2DPolyLine::SetVertices(const std::vector< std::vector<double> >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( KTwoVector( (*ordered_vertices)[i][0], (*ordered_vertices)[i][1] ) );
    }
    Initialize();
}

void
KG2DPolyLine::SetVertices(const std::vector< KTwoVector >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( (*ordered_vertices)[i] );
    }
    Initialize();
}

void KG2DPolyLine::GetVertices(std::vector<KTwoVector>* vertices) const
{
    vertices->clear();
    if(fIsValid)
    {
        vertices->resize(fNVertices);
        for(int i =0; i < fNVertices; i++)
        {
            (*vertices)[i] = fVertices[i];
        }
    }
}

void KG2DPolyLine::GetSides( std::vector<KG2DLineSegment>* sides) const
{
    sides->clear();
    sides->resize(fSides.size());
    for(unsigned int i =0; i < fSides.size(); i++)
    {
        (*sides)[i] = fSides[i];
    }

}

void
KG2DPolyLine::Initialize()
{
    if(fVertices.size() < 2)
    {
        //this is not a polyline, what were you thinking?
        std::cout<<"KG2DPolyLine::Initialize() Warning! Polyline has been assigned less than two vertices."<<std::endl;
        fIsValid = false;
        return;
    }
    else
    {
        fNVertices = fVertices.size();
        //ok, so now build the sides;
        int i,j;
        fSides.clear();
        for(i=0; i<fNVertices-1; i++)
        {
            j = i+1;
            fSides.push_back( KG2DLineSegment(fVertices[i], fVertices[j]) );
        }

        fDiff.resize(fNVertices);
        fNSides = fSides.size();
        DetermineIfPolyLineIsSimple();

        fIsValid = true;

    }
}

void
KG2DPolyLine::NearestDistance( const KTwoVector& aPoint, double& aDistance ) const
{
    double min, dist;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polyline
    fSides[0].NearestDistance(aPoint, dist);
    min = dist;
    aDistance = dist;

    //loop over the rest of the polyline's sides looking for a minimum
    for(int i=0; i<fNSides; i++)
    {
        fSides[i].NearestDistance(aPoint, dist);
        if(dist <= min)
        {
            aDistance = dist;
            min = dist;
        }
    }

}



KTwoVector
KG2DPolyLine::Point( const KTwoVector& aPoint) const
{
  KTwoVector aNearest;
    double min2, dist2;
    KTwoVector near;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polyline
    near = fSides[0].Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    aNearest = near;

    //loop over the rest of the polyline's sides looking for a minimum
    for(int i=0; i<fNSides; i++)
    {
        near = fSides[i].Point(aPoint);
        dist2 = (aPoint - near).MagnitudeSquared();
        if(dist2 <= min2)
        {
            aNearest = near;
            min2 = dist2;
        }
    }
    return aNearest;
}

KTwoVector
KG2DPolyLine::Normal( const KTwoVector& aPoint ) const
{
  KTwoVector aNormal;
    //first we have to find the side with the nearest point
    double min2, dist2;
    KTwoVector near, nearest, normal;
    int index;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polyline
    near = fSides[0].Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    nearest = near;
    index = 0;

    //loop over the rest of the polyline's sides looking for a minimum
    for(int i=0; i<fNSides; i++)
    {
      near = fSides[i].Point(aPoint);
        dist2 = (aPoint - near).MagnitudeSquared();
        if(dist2 <= min2)
        {
            nearest = near;
            index = i;
            min2 = dist2;
        }
    }

    //We can have a situation where the nearest point is a vertex.
    //At a vertex the normal isn't defined. So we fudge things and
    //return the vector which points from the vertex to the point
    //in question.

    //first we check if the nearest point is really close to a vertex
//    if( (nearest - fSides[index].GetFirstPoint() ).MagnitudeSquared() < SMALLNUMBER )
//    {
//        aNormal = (aPoint - fSides[index].GetFirstPoint()).Unit();
//    }
//    else if( (nearest - fSides[index].GetSecondPoint() ).MagnitudeSquared() < SMALLNUMBER )
//    {
//        aNormal = (aPoint - fSides[index].GetSecondPoint()).Unit();
//    }
//    else //no troubles with vertices
//    {

        //ignore issues with the vertices...not sure if this may cause inconsitencies later...
        aNormal = fSides[index].Normal(aPoint);

//    }

	return aNormal;
}

void
KG2DPolyLine::NearestIntersection( const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult, KTwoVector& anIntersection ) const
{
    bool found;
    double min2, dist2;
    KTwoVector inter;


    //initialize minimum distance from start point of segment to 2x the max
    //possible value
    min2 = 2.0*(anEnd - aStart).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt

    //loop over all sides looking for an intersection
    for(int i=0; i<fNSides; i++)
    {
        found = false;
        fSides[i].NearestIntersection(aStart, anEnd, found, inter);
        if(found)
        {
            dist2 = (inter - aStart).MagnitudeSquared();
            if(dist2 <= min2)
            {
                aResult = true;
                anIntersection = inter;
            }
        }
    }


}

void
KG2DPolyLine::DetermineIfPolyLineIsSimple()
{
    //determine if the polyline is simple or not,
    //this is the slowest O(n^2) and stupidest way of finding self-intersections
    //may replace this in the future with something faster
    KTwoVector inter;
    bool test = false;
    fIsSimple = true;
    int i,j,k;
    i=0;
    do
    {
        //loop only non-adjacent vertices
        j = i+1;
        k = i-1;
        for(int x = 0; x < fNSides; x++)
        {
            if(x!=i && x!=j && x!=k)
            {
                fSides[i].NearestIntersection(fSides[x], test, inter);
                if(test)
                {
                    fIsSimple = false;
                }
            }
        }
        i++;
    }
    while( fIsSimple && (i<fNSides) );

    //since we only looped over non-adjacent vertices in the last test
    //we might have missed cases where two adjacent sides retrace previous sides
    //i.e. something like {(0,0), (1,0), (2,0), (-1,0)}
    //would qualify as a simple polyline under the last test
    //so we need to check if the angle between any adjacent sides is 180 degrees
    double costheta;
    for(i=1; i<fNSides; i++)
    {
        j = i-1;
        costheta = (fSides[i].GetUnitVector() )*(fSides[j].GetUnitVector());
        if( costheta + 1. < SMALLNUMBER)
        {
            fIsSimple = false;
        }
    }

}

}//end kgeobag namespace
