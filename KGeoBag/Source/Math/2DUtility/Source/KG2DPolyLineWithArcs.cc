#include "KG2DPolyLineWithArcs.hh"

#include <iostream>

namespace KGeoBag
{


KG2DPolyLineWithArcs::KG2DPolyLineWithArcs() : fIsValid(false)
{
    ;
}

KG2DPolyLineWithArcs::KG2DPolyLineWithArcs(const std::vector<KGVertexSideDescriptor>* ordered_descriptors) :
    fIsValid(false)
{
    SetDescriptors(ordered_descriptors);
    Initialize();
}

KG2DPolyLineWithArcs::~KG2DPolyLineWithArcs()
{
    for (unsigned int i = 0; i < fSides.size(); i++) {
        delete fSides[i];
    }
}

void KG2DPolyLineWithArcs::SetDescriptors(const std::vector<KGVertexSideDescriptor>* ordered_descriptors)
{
    fDescriptors = *ordered_descriptors;
    Initialize();
}

void KG2DPolyLineWithArcs::Initialize()
{
    if (fDescriptors.size() < 2) {
        std::cout << "KG2DPolyLineWithArcs::Initialize() Warning! polyline has been assigned less than two vertices."
                  << std::endl;
        fIsValid = false;
    }
    else {
        //start by extracting the vertices
        fNVertices = fDescriptors.size();
        fVertices.clear();
        fVertices.resize(fNVertices);
        for (int i = 0; i < fNVertices; i++) {
            fVertices[i] = fDescriptors[i].Vertex;
        }

        //if for some odd reason there are sides already, eliminate them
        if (fSides.size() > 0) {
            for (unsigned int i = 0; i < fSides.size(); i++) {
                delete fSides[i];
            }
        }

        fSides.clear();
        fArcs.clear();
        fNArcs = 0;

        int j;
        for (int i = 0; i < fNVertices - 1; i++) {
            j = i + 1;
            if (fDescriptors[i].IsArc) {
                //create an arc
                fArcs.push_back(new KG2DArc(fVertices[i],
                                            fVertices[j],
                                            fDescriptors[i].Radius,
                                            fDescriptors[i].IsRight,
                                            fDescriptors[i].IsCCW));
                fNArcs++;
                fSides.push_back(fArcs.back());
            }
            else {
                //create a line segment
                fSides.push_back(new KG2DLineSegment(fVertices[i], fVertices[j]));
            }
        }

        fNSides = fSides.size();

        DetermineIfPolyLineIsSimple();

        fIsValid = true;
    }
}


void KG2DPolyLineWithArcs::GetVertices(std::vector<KTwoVector>* vertices) const
{
    vertices->clear();
    if (fIsValid) {
        vertices->resize(fNVertices);
        for (int i = 0; i < fNVertices; i++) {
            (*vertices)[i] = fVertices[i];
        }
    }
}

void KG2DPolyLineWithArcs::GetSides(std::vector<KG2DShape*>* sides) const
{
    sides->clear();
    if (fIsValid) {
        sides->resize(fNSides);
        for (int i = 0; i < fNSides; i++) {
            (*sides)[i] = fSides[i];
        }
    }
}

void KG2DPolyLineWithArcs::NearestDistance(const KTwoVector& aPoint, double& aDistance) const
{
    double min, dist;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polyline
    fSides[0]->NearestDistance(aPoint, dist);
    min = dist;
    aDistance = dist;

    //loop over the rest of the polyline's sides looking for a minimum
    for (int i = 0; i < fNSides; i++) {
        fSides[i]->NearestDistance(aPoint, dist);
        if (dist <= min) {
            aDistance = dist;
            min = dist;
        }
    }
}

KTwoVector KG2DPolyLineWithArcs::Point(const KTwoVector& aPoint) const
{
    KTwoVector aNearest;
    double min2, dist2;
    KTwoVector near;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polyline
    near = fSides[0]->Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    aNearest = near;

    //loop over the rest of the polyline's sides looking for a minimum
    for (int i = 0; i < fNSides; i++) {
        near = fSides[i]->Point(aPoint);
        dist2 = (aPoint - near).MagnitudeSquared();
        if (dist2 <= min2) {
            aNearest = near;
            min2 = dist2;
        }
    }
    return aNearest;
}

KTwoVector KG2DPolyLineWithArcs::Normal(const KTwoVector& aPoint) const
{
    KTwoVector aNormal;
    //first we have to find the side with the nearest point
    double min2, dist2;
    KTwoVector near, nearest, normal;
    int index;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polyline
    near = fSides[0]->Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    nearest = near;
    index = 0;

    //loop over the rest of the polyline's sides looking for a minimum
    for (int i = 0; i < fNSides; i++) {
        near = fSides[i]->Point(aPoint);
        dist2 = (aPoint - near).MagnitudeSquared();
        if (dist2 <= min2) {
            nearest = near;
            index = i;
        }
    }

    //We can have a situation where the nearest point is a vertex.
    //At a vertex the normal isn't defined. So we fudge things and
    //return the vector which points from the vertex to the point
    //in question.

    //    //first we check if the nearest point is really close to a vertex
    //    if( (nearest - fVertices[index]).MagnitudeSquared() < SMALLNUMBER*SMALLNUMBER )
    //    {
    //        aNormal = ((nearest - fVertices[index])).Unit();
    //        return;
    //    }
    //    else if( index + 1 < fNVertices )
    //    {
    //        if( (nearest - fVertices[index +1 ]).MagnitudeSquared() < SMALLNUMBER*SMALLNUMBER )
    //        {
    //            aNormal = ((nearest - fVertices[index + 1])).Unit();
    //            return;
    //        }
    //    }
    //    else //no troubles with vertices
    //    {


    //ignore any possible issues with vertices for now
    aNormal = fSides[index]->Normal(aPoint);


    //        return;
    //    }
    return aNormal;
}

void KG2DPolyLineWithArcs::NearestIntersection(const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult,
                                               KTwoVector& anIntersection) const
{
    bool found;
    double min2, dist2;
    KTwoVector inter;

    //initialize minimum distance from start point of segment to the max
    //possible value
    min2 = (anEnd - aStart).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt

    //loop over all sides looking for an intersection
    for (int i = 0; i < fNSides; i++) {
        found = false;
        fSides[i]->NearestIntersection(aStart, anEnd, found, inter);
        if (found) {
            dist2 = (inter - aStart).MagnitudeSquared();
            if (dist2 <= min2) {
                aResult = true;
                anIntersection = inter;
            }
        }
    }
}


void KG2DPolyLineWithArcs::DetermineIfPolyLineIsSimple()
{
    KG2DLineSegment* aLine;
    KG2DArc* aArc;
    KG2DArc* ArcUnderTest;
    KG2DLineSegment* LineUnderTest;
    int test_type = 0;


    int i = 0;
    //determine if the polyline is simple or not,
    //this is the slowest O(n^2) and stupidest way of finding self-intersections
    //may replace this in the future with something faster
    KTwoVector inter;
    std::vector<KTwoVector> inters;
    bool test;
    int flag;
    fIsSimple = true;

    do {
        test = false;
        aLine = nullptr;
        aArc = nullptr;
        LineUnderTest = nullptr;
        ArcUnderTest = nullptr;
        inters.clear();
        flag = 0;

        aLine = dynamic_cast<KG2DLineSegment*>(fSides[i]);
        aArc = dynamic_cast<KG2DArc*>(fSides[i]);

        for (int x = 0; x < fNSides; x++) {
            if (x != i)  //no tests against self
            {
                LineUnderTest = dynamic_cast<KG2DLineSegment*>(fSides[x]);
                ArcUnderTest = dynamic_cast<KG2DArc*>(fSides[x]);

                if (aLine && LineUnderTest) {
                    test_type = 0;
                };
                if (aLine && ArcUnderTest) {
                    test_type = 1;
                };
                if (aArc && LineUnderTest) {
                    test_type = 2;
                };
                if (aArc && ArcUnderTest) {
                    test_type = 3;
                };

                switch (test_type) {
                    case 0:
                        aLine->NearestIntersection(*LineUnderTest, test, inter);
                        break;
                    case 1:
                        ArcUnderTest->NearestIntersection(aLine->GetFirstPoint(), aLine->GetSecondPoint(), test, inter);
                        break;
                    case 2:
                        aArc->NearestIntersection(LineUnderTest->GetFirstPoint(),
                                                  LineUnderTest->GetSecondPoint(),
                                                  test,
                                                  inter);
                        break;
                    case 3:
                        aArc->NearestIntersection(ArcUnderTest, flag, &inters);
                        break;
                }


                if (test || flag != 0) {
                    //we found an intersection somewhere but
                    //check to make sure it is not just overlap between the end points
                    //this can only happen if the segments are adjacent
                    if (x == i + 1 || x == i - 1) {
                        if (test_type == 0 || test_type == 1)  //intersection with a line segment
                        {
                            if (!(((inter - aLine->GetFirstPoint()).Magnitude() < SMALLNUMBER) ||
                                  ((inter - aLine->GetSecondPoint()).Magnitude() < SMALLNUMBER))) {
                                //not close to end point
                                fIsSimple = false;
                                return;
                            }
                        }

                        if (test_type == 2)  //intersection with an arc segment
                        {
                            if (!(((inter - aArc->GetFirstPoint()).Magnitude() < SMALLNUMBER) ||
                                  ((inter - aArc->GetSecondPoint()).Magnitude() < SMALLNUMBER))) {
                                //not close to end point
                                fIsSimple = false;
                                return;
                            }
                        }


                        if (test_type == 3 && flag > 0)  //intersection with an arc segment
                        {
                            for (unsigned int m = 0; m < inters.size(); m++) {
                                if (!(((inters[m] - aArc->GetFirstPoint()).Magnitude() < SMALLNUMBER) ||
                                      ((inters[m] - aArc->GetSecondPoint()).Magnitude() < SMALLNUMBER))) {
                                    //not close to end point
                                    fIsSimple = false;
                                    return;
                                }
                            }
                        }

                        if (flag == -1)  //large overlap between arcs
                        {
                            fIsSimple = false;
                            return;
                        }
                    }
                    else {
                        fIsSimple = false;
                        return;
                    }
                }
            }
        }
        i++;
    } while (fIsSimple && (i < fNSides));
}


}  // namespace KGeoBag
