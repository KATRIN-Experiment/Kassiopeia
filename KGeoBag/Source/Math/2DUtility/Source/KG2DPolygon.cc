#include "KG2DPolygon.hh"

#include <iostream>

namespace KGeoBag
{


KG2DPolygon::KG2DPolygon():
fIsValid(false){;}

KG2DPolygon::KG2DPolygon(const KG2DPolygon& copyObject):
KG2DArea(copyObject)
{
    //not a true copy constructor, since we are re-initalizing from the
    //vertices, may want to rethink this if we are doing a lot of copying of
    //large objects
    SetVertices( &(copyObject.fVertices) );
}

KG2DPolygon::KG2DPolygon(const std::vector< std::vector<double> >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( KTwoVector( (*ordered_vertices)[i][0], (*ordered_vertices)[i][1] ) );
    }
    Initialize();
}

KG2DPolygon::KG2DPolygon(const std::vector< KTwoVector >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( (*ordered_vertices)[i] );
    }
    Initialize();
}

KG2DPolygon::~KG2DPolygon(){;}

void
KG2DPolygon::SetVertices(const std::vector< std::vector<double> >* ordered_vertices)
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
KG2DPolygon::SetVertices(const std::vector< KTwoVector >* ordered_vertices)
{
    fIsValid = false;
    fVertices.clear();
    for(unsigned int i=0; i< ordered_vertices->size(); i++)
    {
        fVertices.push_back( (*ordered_vertices)[i] );
    }
    Initialize();
}

void KG2DPolygon::GetVertices(std::vector<KTwoVector>* vertices) const
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

void KG2DPolygon::GetSides( std::vector<KG2DLineSegment>* sides) const
{
    sides->clear();
    if(fIsValid)
    {
        sides->resize(fNVertices);
        for(int i =0; i < fNVertices; i++)
        {
            (*sides)[i] = fSides[i];
        }
    }
}

void
KG2DPolygon::Initialize()
{
    if(fVertices.size() < 3)
    {
        std::cout<<"KG2DPolygon::Initialize() Warning! Polygon has been assigned less than three vertices."<<std::endl;
        fIsValid = false;
        return;
    }
    else
    {
        fNVertices = fVertices.size();
        //ok, so now build the sides;
        int i,j;
        fSides.clear();
        for(i=0; i<fNVertices; i++)
        {
            j = Modulus(i+1, fNVertices);
            fSides.push_back( KG2DLineSegment(fVertices[i], fVertices[j]) );
        }

        DetermineIfPolygonIsSimple();

        fDiff.resize(fNVertices);

        fIsValid = true;
    }

//    if(fIsValid){std::cout<<"plain polygon is valid"<<std::endl;};
//    if(!fIsValid){std::cout<<"plain polygon is not valid"<<std::endl;};
//    if(fIsSimple){std::cout<<"plain polygon is simple"<<std::endl;};
//    if(!fIsSimple){std::cout<<"plain polygon is not simple"<<std::endl;};
//    if(!fIsSimple)
//    {
//        for(int i=0; i<fNVertices; i++)
//        {
//            std::cout<<fVertices[i]<<std::endl;
//        }

//    };


//    std::cout<<"plain polygon has "<<fSides.size()<<" sides"<<std::endl;

    DetermineInteriorSide();


}

void
KG2DPolygon::NearestDistance( const KTwoVector& aPoint, double& aDistance ) const
{
    double min, dist;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polygon
    fSides[0].NearestDistance(aPoint, dist);
    min = dist;
    aDistance = dist;

    //loop over the rest of the polygon's sides looking for a minimum
    for(int i=1; i<fNVertices; i++)
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
KG2DPolygon::Point( const KTwoVector& aPoint ) const
{
    KTwoVector aNearest;
    double min2, dist2;
    KTwoVector near;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polygon
    near = fSides[0].Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    aNearest = near;

    //loop over the rest of the polygon's sides looking for a minimum
    for(int i=1; i<fNVertices; i++)
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
KG2DPolygon::Normal( const KTwoVector& aPoint ) const
{
    KTwoVector aNormal;
    //first we have to find the side with the nearest point
    double min2, dist2;
    KTwoVector near, nearest, normal;
    int index;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polygon
    near = fSides[0].Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    nearest = near;
    index = 0;

    //loop over the rest of the polygon's sides looking for a minimum
    for(int i=1; i<fNVertices; i++)
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
    // At a vertex the normal isn't defined, but we can fudge things and take
    //the average of the normals of the two sides that meet at that vertex.
    //If we aren't near a vertex then we are in the clear.

    //first we check if the nearest point is really close to a vertex
//    if( (nearest - fSides[index].GetFirstPoint() ).MagnitudeSquared() < SMALLNUMBER )
//    {
//        //ask the two sides that share the vertex for their normals
//        //and average, these are sides i and i-1
//        fSides[index].NearestNormal(aPoint, normal);
//        aNormal = normal;
//        index_adjacent = index - 1;
//        if(index_adjacent == -1){index_adjacent = fNVertices-1;};
//        fSides[index_adjacent].NearestNormal(aPoint, normal);
//        aNormal += normal;
//        aNormal *= 0.5;
//        aNormal = aNormal.Unit();
//    }
//    else if( (nearest - fSides[index].GetSecondPoint() ).MagnitudeSquared() < SMALLNUMBER )
//    {
//        //ask the two sides that share the vertex for their normals
//        //and average, these are sides i and i+1
//        fSides[index].NearestNormal(aPoint, normal);
//        aNormal = normal;
//        index_adjacent = index + 1;
//        if(index_adjacent == fNVertices ){index_adjacent = 0;};
//        fSides[index_adjacent].NearestNormal(aPoint, normal);
//        aNormal += normal;
//        aNormal *= 0.5;
//        aNormal = aNormal.Unit();
//    }
//    else //no troubles with vertices
//    {


      //ignore any possible issues with the vertices for now
      aNormal = fSides[index].Normal(aPoint);


//    }
      return aNormal;
}

void
KG2DPolygon::NearestIntersection( const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult, KTwoVector& anIntersection ) const
{
    bool found;
    double min2, dist2;
    KTwoVector inter;


    //initialize minimum distance from start point of segment to the max
    //possible value
    min2 = 2.0*(anEnd - aStart).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt

    //loop over all sides looking for an intersection
    for(int i=0; i<fNVertices; i++)
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

int
KG2DPolygon::Modulus(int arg, int n)
{
    double div = ( (double)arg )/( (double) n);
    return (int)(std::fabs( (double)arg - std::floor(div)*((double)n) ) );
}

bool
KG2DPolygon::IsInside(const KTwoVector& point) const
{
    //The Point in Polygon Problem for Arbitrary Polygons
    //Kai Hormann, Alexander Agathos
    //University of Erlangen,  University of Athens
    //Algorithm #4

    int wn = 0;    // the winding number counter
    int i, j;
    bool IsUpward;

    //first compute displacement of each vertex from the point
    //we need each of these more than once,
    //so we might as well get this done first
    for(i=0; i < fNVertices; i++)
    {
        fDiff[i] = fVertices[i] - point;
    }

    //looping over all sides of the polygon
    for(i=0; i < fNVertices; i++)
    {
        //index of next vertex, if i is the last vertex, loop back to beginning
        j = i+1;
        if(j == fNVertices){j = 0;}

        //horizontal line crossed?
        if( ( (fDiff[i].Y() < 0. ) && (fDiff[j].Y() >= 0. ) ) ||
            ( (fDiff[i].Y() >= 0. ) && (fDiff[j].Y() < 0. ) )    )
        {
            //crossing to the right?
            IsUpward = (fVertices[j].Y() > fVertices[i].Y() );
            if( (fDiff[i]^fDiff[j]) > 0 )
            {
                if(IsUpward)
                {
                    wn++;
                }
            }
            else
            {
                if(!IsUpward)
                {
                    wn--;
                }
            }
        }
    }


    if( wn != 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void
KG2DPolygon::DetermineIfPolygonIsSimple()
{
    //determine if the polygon is simple or not,
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
        j = Modulus(i+1,fNVertices);
        k = Modulus(i-1,fNVertices);
        for(int x = 0; x < fNVertices; x++)
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
    while( fIsSimple && (i<fNVertices) );

    //since we only looped over non-adjacent vertices in the last test
    //we might have missed cases where two adjacent sides retrace previous sides
    //i.e. something like {(0,0), (1,0), (2,0), (-1,0)}
    //would qualify as a simple polygon under the last test
    //so we need to check if the angle between any adjacent sides is 180 degrees
    double costheta;
    for(i=0; i<fNVertices; i++)
    {
        j = Modulus(i+1, fNVertices);
        costheta = (fSides[i].GetUnitVector() )*(fSides[j].GetUnitVector());
        if( costheta + 1. < SMALLNUMBER)
        {
            fIsSimple = false;
        }
    }

}

void
KG2DPolygon::DetermineInteriorSide()
{
    int i,j,k;
    KTwoVector first;
    KTwoVector second;
    double costheta;
    double theta;
    double total_angle = 0;

    for(i = 0; i<fNVertices; i++)
    {
        j = KG2DPolygon::Modulus( i+1, fNVertices);
        k = KG2DPolygon::Modulus( i-1, fNVertices);
        first = fVertices[i] - fVertices[k];
        second = fVertices[j] - fVertices[i];
        first = first.Unit();
        second = second.Unit();
        costheta = first*second;
        theta = std::fabs( std::acos(costheta) );
        if( (first^second) > 0)
        {
            //we have a left hand turn
            total_angle += theta;
        }
        else
        {
            //we have a right hand turn
            total_angle -= theta;
        }
    }

    if(total_angle > 0)
    {
        fIsLeft = true;
    }
    else
    {
        fIsLeft = false;
    }


}

double KG2DPolygon::Area() const
{
  double aResult = 0.;
         if ( fIsSimple )
         {
                 int j = fNVertices - 1;
                for(int i=0; i < fNVertices; i++)
                 {
                         aResult += (fVertices[j].X()+fVertices[i].X())*(fVertices[i].Y()-fVertices[j].Y());
                         j=i;
                 }
                 aResult = fabs(aResult * 0.5);

         }
         else
         {
            std::cout<<"KG2DPolygon::AreaMeasure() Warning! Unable to calculate the area, because given polygon is too complex."<<std::endl;
         }
	 return aResult;
}


//static utility functions for navigation
double
KG2DPolygon::NearestDistance(const std::vector< KTwoVector >* ordered_vertices,
                             const KTwoVector& aPoint)
{
    //initialize min search with distance to first segment
    double min = KG2DLineSegment::NearestDistance( (*ordered_vertices)[0], (*ordered_vertices)[1], aPoint);

    unsigned int n_vertices = ordered_vertices->size();
    for(unsigned int i=1; i<n_vertices; i++)
    {
        unsigned int a = i;
        unsigned int b = i+1; if(b == n_vertices){b = 0;};
        double dist = KG2DLineSegment::NearestDistance( (*ordered_vertices)[a], (*ordered_vertices)[b], aPoint);
        if(dist <= min)
        {
            min = dist;
        }
    }
    return min;
}

KTwoVector
KG2DPolygon::NearestPoint(const std::vector< KTwoVector >* ordered_vertices,
                          const KTwoVector& aPoint)
{
    //initialize min search with distance to first segment
    KTwoVector nearest = KG2DLineSegment::NearestPoint( (*ordered_vertices)[0], (*ordered_vertices)[1], aPoint);
    double min = (nearest - aPoint).Magnitude();
    KTwoVector test;

    unsigned int n_vertices = ordered_vertices->size();
    for(unsigned int i=1; i<n_vertices; i++)
    {
        unsigned int a = i;
        unsigned int b = i+1; if(b == n_vertices){b = 0;};
        test = KG2DLineSegment::NearestPoint( (*ordered_vertices)[a], (*ordered_vertices)[b], aPoint);
        double dist = (test - aPoint).Magnitude();
        if(dist <= min)
        {
            min = dist;
            nearest = test;
        }
    }
    return nearest;
}

KTwoVector
KG2DPolygon::NearestNormal(const std::vector< KTwoVector >* ordered_vertices,
                                const KTwoVector& aPoint)
{

    //initialize min search with distance to first segment
    double min = KG2DLineSegment::NearestDistance( (*ordered_vertices)[0], (*ordered_vertices)[1], aPoint);
    unsigned int n_vertices = ordered_vertices->size();
    unsigned int side_index = 0;
    for(unsigned int i=1; i<n_vertices; i++)
    {
        unsigned int a = i;
        unsigned int b = i+1; if(b == n_vertices){b = 0;};
        double dist = KG2DLineSegment::NearestDistance( (*ordered_vertices)[a], (*ordered_vertices)[b], aPoint);
        if(dist <= min)
        {
            min = dist;
            side_index = i;
        }
    }

    unsigned int a = side_index;
    unsigned int b = side_index+1; if(b == n_vertices){b = 0;};
    KTwoVector normal = KG2DLineSegment::NearestNormal( (*ordered_vertices)[a], (*ordered_vertices)[b], aPoint);
    return normal;
}

bool
KG2DPolygon::NearestIntersection(const std::vector< KTwoVector >* ordered_vertices,
                                const KTwoVector& aStart, const KTwoVector& anEnd,
                                KTwoVector& anIntersection )
{
    bool found, global_found;
    double min;
    KTwoVector inter;

    //initialize minimum distance from start point of segment to the max
    //possible value
    min = (anEnd - aStart).Magnitude();  //use MagnitudeSquared to avoid computing sqrt
    global_found = false;

    unsigned int n_vertices = ordered_vertices->size();
    for(unsigned int i=0; i<n_vertices; i++)
    {
        unsigned int a = i;
        unsigned int b = i+1; if(b == n_vertices){b = 0;};
        //initialize min search with distance to first segment
        found = KG2DLineSegment::NearestIntersection( (*ordered_vertices)[a], (*ordered_vertices)[b], aStart, anEnd, inter);
        if(found)
        {
            double dist = (inter - aStart).Magnitude();
            if(dist < min)
            {
                min = dist;
                anIntersection = inter;
                global_found = true;
            }
        }
    }

    return global_found;
}


bool
KG2DPolygon::IsInside(const std::vector< KTwoVector >* ordered_vertices,
                      const KTwoVector& point)
{
    //The Point in Polygon Problem for Arbitrary Polygons
    //Kai Hormann, Alexander Agathos
    //University of Erlangen,  University of Athens
    //Algorithm #4

    int wn = 0;    // the winding number counter
    unsigned int i, j;
    bool IsUpward;
    unsigned int n_vertices = ordered_vertices->size();
    std::vector< KTwoVector > diff;

    //first compute displacement of each vertex from the point
    //we need each of these more than once,
    //so we might as well get this done first
    for(i=0; i < n_vertices; i++)
    {
        diff.push_back( (*ordered_vertices)[i] - point );
    }

    //looping over all sides of the polygon
    for(i=0; i < n_vertices; i++)
    {
        //index of next vertex, if i is the last vertex, loop back to beginning
        j = i+1;
        if(j == n_vertices){j = 0;}

        //horizontal line crossed?
        if( ( (diff[i].Y() < 0. ) && (diff[j].Y() >= 0. ) ) ||
            ( (diff[i].Y() >= 0. ) && (diff[j].Y() < 0. ) )    )
        {
            //crossing to the right?
            IsUpward = ((*ordered_vertices)[j].Y() > (*ordered_vertices)[i].Y() );
            if( (diff[i]^diff[j]) > 0 )
            {
                if(IsUpward)
                {
                    wn++;
                }
            }
            else
            {
                if(!IsUpward)
                {
                    wn--;
                }
            }
        }
    }

    if( wn != 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

}//end kgeobag namespace
