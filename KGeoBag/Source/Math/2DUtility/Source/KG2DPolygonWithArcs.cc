#include "KG2DPolygonWithArcs.hh"

#include <iostream>

namespace KGeoBag
{


KG2DPolygonWithArcs::KG2DPolygonWithArcs()
{
    fIsValid = false;
}

KG2DPolygonWithArcs::KG2DPolygonWithArcs(const std::vector<KGVertexSideDescriptor>* ordered_descriptors)
{
    SetDescriptors(ordered_descriptors);
}

KG2DPolygonWithArcs::KG2DPolygonWithArcs(const KG2DPolygonWithArcs& copyObject) : KG2DArea(copyObject)
{
    //not a true copy constructor, since we are re-initalizing from the
    //descriptors, may want to rethink this if we are doing a lot of copying of
    //large objects
    SetDescriptors(&(copyObject.fDescriptors));
}


KG2DPolygonWithArcs::~KG2DPolygonWithArcs()
{
    for (auto& side : fSides) {
        delete side;
    }
}

void KG2DPolygonWithArcs::SetDescriptors(const std::vector<KGVertexSideDescriptor>* ordered_descriptors)
{
    fDescriptors = *ordered_descriptors;
    Initialize();
}

void KG2DPolygonWithArcs::Initialize()
{
    if (fDescriptors.size() < 3) {
        // TO DO: make this check work!!!
        // std::cout<<"KG2DPolygonWithArcs::Initialize() Warning! Polygon has been assigned less than three vertices."<<std::endl;
        fIsValid = false;
        return;
    }

    //ok, so now lets build the base polygon (unmodified by arcs)
    //start by extracting the vertices
    fNVertices = fDescriptors.size();
    fVertices.clear();
    fVertices.resize(fNVertices);
    for (int i = 0; i < fNVertices; i++) {
        fVertices[i] = fDescriptors[i].Vertex;
    }

    //build the base polygon
    fBasePolygon.SetVertices(&fVertices);

    //If the base polygon is not simple then it is extremely
    //difficult to determine a priori whether the interior of the polygon
    //lies to the left or right of a side. For some non-simple polygons
    //this is not even defined. Since non-simple base polygons also
    //complicate determining whether a point is inside or outside the polygons
    //once the arc segments are added, we will not allow construction of
    //KG2DPolygonWithArcs whose base polygon is non-simple


    if (fBasePolygon.IsValid() && fBasePolygon.IsSimple()) {

        //Now that that base polygon has been constructed
        //we need to determine which side of each line the interior
        //of the polygon lies on. To do this we loop over each vertex
        //and check whether the turn it makes is left or right. If
        //the (angle weighted) majority of the turns taken are left,
        //then the interior lies to the left of each line segment,
        //if the majority of the turns taken are right then the
        //interior lies to the right of each segment.
        //turns taken to the left count as positive angles,
        //turns taken to the right count as negative angles.

        DetermineInteriorSide();

        //now we can construct the sides (line segments and arcs)
        if (!fSides.empty()) {
            for (auto& side : fSides) {
                delete side;
            }
        }

        fSides.clear();
        fSides.resize(fNVertices);
        fArcs.clear();
        fNArcs = 0;
        fIsArcAdditive.clear();


        int j;
        for (int i = 0; i < fNVertices; i++) {
            j = KG2DPolygon::Modulus(i + 1, fNVertices);

            if (fDescriptors[i].IsArc) {

                //std::cout<<"side "<<i<<" is an arc:"<<std::endl;
                //create an arc
                //                std::cout<<"arc vertex 1 = "<<fVertices[i]<<std::endl;
                //                std::cout<<"arc vertex 2  = "<<fVertices[j]<<std::endl;
                //                std::cout<<"arc radius = "<<fDescriptors[i].Radius<<std::endl;
                //                std::cout<<"arc is right = "<<fDescriptors[i].IsRight<<std::endl;
                //                std::cout<<"arc is ccw = "<<fDescriptors[i].IsCCW<<std::endl;

                fArcs.push_back(new KG2DArc(fVertices[i],
                                            fVertices[j],
                                            fDescriptors[i].Radius,
                                            fDescriptors[i].IsRight,
                                            fDescriptors[i].IsCCW));
                fNArcs++;
                fSides[i] = fArcs.back();


                if (fIsLeft) {
                    if (fDescriptors[i].IsCCW) {
                        fIsArcAdditive.push_back(true);
                    }
                    else {
                        fIsArcAdditive.push_back(false);
                    }
                }
                else {
                    if (fDescriptors[i].IsCCW) {
                        fIsArcAdditive.push_back(false);
                    }
                    else {
                        fIsArcAdditive.push_back(true);
                    }
                }
            }
            else {
                //create a line segment
                fSides[i] = new KG2DLineSegment(fVertices[i], fVertices[j]);
            }
        }

        DetermineIfPolygonIsSimpleAndValid();
    }
    else {
        fIsValid = false;
    }


    //    if(fIsValid){std::cout<<"polygon is valid"<<std::endl;};
    //    if(!fIsValid){std::cout<<"polygon is not valid"<<std::endl;};
    //    std::cout<<"polygon has "<<fSides.size()<<" sides"<<std::endl;
}

void KG2DPolygonWithArcs::GetVertices(std::vector<KTwoVector>* vertices) const
{
    vertices->clear();
    if (fIsValid) {
        vertices->resize(fNVertices);
        for (int i = 0; i < fNVertices; i++) {
            (*vertices)[i] = fVertices[i];
        }
    }
}

void KG2DPolygonWithArcs::GetSides(std::vector<KG2DShape*>* sides) const
{
    sides->clear();
    if (fIsValid) {
        sides->resize(fNVertices);
        for (int i = 0; i < fNVertices; i++) {
            (*sides)[i] = fSides[i];
        }
    }
}

void KG2DPolygonWithArcs::NearestDistance(const KTwoVector& aPoint, double& aDistance) const
{
    double min, dist;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polygon
    fSides[0]->NearestDistance(aPoint, dist);
    min = dist;
    aDistance = dist;

    //loop over the rest of the polygon's sides looking for a minimum
    for (int i = 1; i < fNVertices; i++) {
        fSides[i]->NearestDistance(aPoint, dist);
        if (dist <= min) {
            aDistance = dist;
            min = dist;
        }
    }
}

KTwoVector KG2DPolygonWithArcs::Point(const KTwoVector& aPoint) const
{
    KTwoVector aNearest;
    double min2, dist2;
    KTwoVector near;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polygon
    near = fSides[0]->Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    aNearest = near;

    //loop over the rest of the polygon's sides looking for a minimum
    for (int i = 0; i < fNVertices; i++) {
        near = fSides[i]->Point(aPoint);
        dist2 = (aPoint - near).MagnitudeSquared();
        if (dist2 <= min2) {
            aNearest = near;
            min2 = dist2;
        }
    }
    return aNearest;
}

KTwoVector KG2DPolygonWithArcs::Normal(const KTwoVector& aPoint) const
{
    KTwoVector aNormal;
    //first we have to find the side with the nearest point
    double min2, dist2;
    KTwoVector near, nearest, normal;
    int index;

    //initialize minimum distance from point to the nearest distance
    //of the first side of the polygon
    near = fSides[0]->Point(aPoint);
    min2 = (aPoint - near).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt
    nearest = near;
    index = 0;

    //loop over the rest of the polygon's sides looking for a minimum
    for (int i = 1; i < fNVertices; i++) {
        near = fSides[i]->Point(aPoint);
        dist2 = (aPoint - near).MagnitudeSquared();
        if (dist2 <= min2) {
            nearest = near;
            index = i;
        }
    }

    //We can have a situation where the nearest point is a vertex.
    // At a vertex the normal isn't defined, but we can fudge things and take
    //the average of the normals of the two sides that meet at that vertex.
    //If we aren't near a vertex then we are in the clear.

    //check if the nearest point is very close to one of the vertices of the
    //closest side
    //    KTwoVector start_point, end_point;
    //    start_point = fVertices[index];
    //    end_point = fVertices[KG2DPolygon::Modulus(index+1, fNVertices)];

    //    if( (nearest - start_point ).MagnitudeSquared() < SMALLNUMBER )
    //    {
    //        //ask the two sides that share the vertex for their normals
    //        //and average, these are sides i and i-1
    //        fSides[index]->NearestNormal(aPoint, normal);
    //        aNormal = normal;
    //        index_adjacent = index - 1;
    //        if(index_adjacent == -1){index_adjacent = fNVertices-1;};
    //        fSides[index_adjacent]->NearestNormal(aPoint, normal);
    //        aNormal += normal;
    //        aNormal *= 0.5;
    //        aNormal = aNormal.Unit();
    //    }
    //    else if( (nearest - end_point ).MagnitudeSquared() < SMALLNUMBER )
    //    {
    //        //ask the two sides that share the vertex for their normals
    //        //and average, these are sides i and i+1
    //        fSides[index]->NearestNormal(aPoint, normal);
    //        aNormal = normal;
    //        index_adjacent = index + 1;
    //        if(index_adjacent == fNVertices ){index_adjacent = 0;};
    //        fSides[index_adjacent]->NearestNormal(aPoint, normal);
    //        aNormal += normal;
    //        aNormal *= 0.5;
    //        aNormal = aNormal.Unit();
    //    }
    //    else //no troubles with vertices
    //    {


    //for now we are ignoring issues with vertices
    aNormal = fSides[index]->Normal(aPoint);

    return aNormal;
    //}
}

void KG2DPolygonWithArcs::NearestIntersection(const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult,
                                              KTwoVector& anIntersection) const
{
    bool found;
    double min2, dist2;
    KTwoVector inter;

    //initialize minimum distance from start point of segment to the max
    //possible value
    min2 = (anEnd - aStart).MagnitudeSquared();  //use MagnitudeSquared to avoid computing sqrt

    //    std::cout<<"n sides = "<<fSides.size()<<std::endl;
    //    std::cout<<"aStart = "<<aStart.X()<<", "<<aStart.Y()<<std::endl;
    //    std::cout<<"anEnd = "<<anEnd.X()<<", "<<anEnd.Y()<<std::endl;


    //loop over all sides looking for an intersection
    for (auto* side : fSides) {
        found = false;
        side->NearestIntersection(aStart, anEnd, found, inter);
        if (found) {
            dist2 = (inter - aStart).MagnitudeSquared();
            if (dist2 <= min2) {
                aResult = true;
                anIntersection = inter;
            }
        }
    }
}

bool KG2DPolygonWithArcs::IsInside(const KTwoVector& point) const
{
    bool IsInsideBase = fBasePolygon.IsInside(point);

    int plus_score = 0;
    int minus_score = 0;

    for (int i = 0; i < fNArcs; i++) {
        if (fArcs[i]->IsInsideCircularSegment(point)) {
            if (fIsArcAdditive[i]) {
                plus_score++;
            }
            else {
                minus_score++;
            }
        }
    }

    bool IsInBaseOrAdditive = false;

    if (IsInsideBase || plus_score > 0) {
        IsInBaseOrAdditive = true;
    }

    if (IsInBaseOrAdditive) {
        if (minus_score == 0) {
            return true;
        }
        else if (minus_score % 2 == 1) {
            return false;
        }
        else {
            return true;
        }
    }
    else {
        if (minus_score % 2 == 1) {
            return true;
        }
        else {
            return false;
        }
    }
}

void KG2DPolygonWithArcs::DetermineIfPolygonIsSimpleAndValid()
{

    //since we are guaranteed that the base polygon is simple
    //we only need to test the arcs to see if they intersect
    //each other or the other straight segments

    KTwoVector inter;
    std::vector<KTwoVector> inters;
    bool test;
    int flag;
    fIsSimple = true;
    fIsValid = true;

    KG2DLineSegment* aLine;
    KG2DArc* ArcUnderTest;
    KG2DArc* aArc;

    int i = 0;

    if (fNArcs > 0) {
        do {
            ArcUnderTest = fArcs[i];
            //loop over all other sides except this one
            for (int j = 0; j < fNVertices; j++) {
                if (ArcUnderTest != fSides[j])  //don't test the arc against itself
                {
                    aLine = dynamic_cast<KG2DLineSegment*>(fSides[j]);
                    if (aLine) {
                        //test for intersection with line
                        ArcUnderTest->NearestIntersection(aLine->GetFirstPoint(), aLine->GetSecondPoint(), test, inter);
                        if (test) {
                            //check to make sure its not an end point
                            bool close_to_end_point = false;
                            if ((inter - ArcUnderTest->GetFirstPoint()).Magnitude() < SMALLNUMBER) {
                                close_to_end_point = true;
                            }
                            if ((inter - ArcUnderTest->GetSecondPoint()).Magnitude() < SMALLNUMBER) {
                                close_to_end_point = true;
                            }

                            if (!close_to_end_point) {
                                //the intersection was somewhere else
                                fIsSimple = false;
                                //std::cout<<"intersection of arc with line!"<<std::endl;
                            }
                        }
                    }
                    else {
                        aArc = dynamic_cast<KG2DArc*>(fSides[j]);
                        if (aArc) {
                            //test for intersection with arc
                            ArcUnderTest->NearestIntersection(aArc, flag, &inters);
                            if (flag == -1) {
                                fIsValid = false;
                            }
                            if (flag > 0) {
                                //check to make sure intersections are not an end points
                                bool close_to_end_point = true;

                                for (auto& inter : inters) {
                                    if ((inter - ArcUnderTest->GetFirstPoint()).Magnitude() > SMALLNUMBER) {
                                        close_to_end_point = false;
                                    }
                                    if ((inter - ArcUnderTest->GetSecondPoint()).Magnitude() > SMALLNUMBER) {
                                        close_to_end_point = false;
                                    }
                                }

                                if (!close_to_end_point) {
                                    //the intersection was somewhere else
                                    fIsSimple = false;
                                    //std::cout<<"intersection of arc with arc!"<<std::endl;
                                    //std::cout<<"segment "<<i<<" with segment "<<j<<std::endl;
                                    //                                    for(unsigned int n=0; n < inters.size(); n++)
                                    //                                    {
                                    //                                        std::cout<<"intersection "<<n<<" is: "<<inters[n]<<std::endl;
                                    //                                    }
                                }
                            }
                        }
                    }
                }
            }
            i++;
        } while (fIsSimple && (i < fNArcs));
    }
}


void KG2DPolygonWithArcs::DetermineInteriorSide()
{
    int i, j, k;
    KTwoVector first;
    KTwoVector second;
    double costheta;
    double theta;
    double total_angle = 0;

    for (i = 0; i < fNVertices; i++) {
        j = KG2DPolygon::Modulus(i + 1, fNVertices);
        k = KG2DPolygon::Modulus(i - 1, fNVertices);
        first = fVertices[i] - fVertices[k];
        second = fVertices[j] - fVertices[i];
        first = first.Unit();
        second = second.Unit();
        costheta = first * second;
        theta = std::fabs(std::acos(costheta));
        if ((first ^ second) > 0) {
            //we have a left hand turn
            total_angle += theta;
        }
        else {
            //we have a right hand turn
            total_angle -= theta;
        }
    }

    if (total_angle > 0) {
        fIsLeft = true;
    }
    else {
        fIsLeft = false;
    }
}


}  // namespace KGeoBag
