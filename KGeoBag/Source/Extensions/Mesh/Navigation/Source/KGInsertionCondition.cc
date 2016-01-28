#include "KGInsertionCondition.hh"
#include "KGPointCloud.hh"
#include "KGAxisAlignedBox.hh"

namespace KGeoBag
{

bool
KGInsertionCondition::ElementIntersectsCube(const KGNavigableMeshElement* element, const KGCube<KGMESH_DIM>* cube) const
{
    if(element != NULL && cube !=NULL)
    {
        //get the cube center and nearest point on the element to the center
        KGPoint<KGMESH_DIM> center = cube->GetCenter();
        KThreeVector nearest_point = element->GetMeshElement()->NearestPoint( KThreeVector(center[0], center[1], center[2]) );
        //the ratio of the cube's diagonal to its side length is sqrt(3/4)
        //since we use the square distance to compare, this gives us the factor of 0.75
        double dist2 = (nearest_point - KThreeVector(center)).MagnitudeSquared();

        //if nearest point is within cube bounding ball insert the element
        //this is not optimal, but robust
        double len = cube->GetLength();
        if(dist2 <= 0.75*len*len )
        {
            return true;
        }

        //
        //
        // if(dist2 > 0.75*(cube->GetLength())*(cube->GetLength()) )
        // {
        //     //quick culling of this element, no possible intersection
        //     //since it is outside the bounding sphere of the cube
        //     return false;
        // }
        //
        // //check if the nearest point is inside the cube
        // if( cube->PointIsInside(nearest_point) )
        // {
        //     //nearest point is inside the cube
        //     return true;
        // }
        //
        // //now check if any of the vertices are inside the cube
        // KGPointCloud<KGMESH_DIM> cloud = element->GetMeshElement()->GetPointCloud();
        // for(unsigned int i=0; i<cloud.GetNPoints(); i++)
        // {
        //     if( cube->PointIsInside(cloud.GetPoint(i)) )
        //     {
        //         //at least one vertex is inside the cube
        //         return true;
        //     }
        // }
        //
        // //more difficult case, vertices lie outside the cube, but it is
        // //close enough that it might clip a corner or edge of the cube
        //
        // //now we check if any of the edges of the mesh element intersects the cube
        // unsigned int n_edges = element->GetMeshElement()->GetNumberOfEdges();
        // KThreeVector start, end;
        // for(unsigned int i=0; i<n_edges; i++)
        // {
        //     element->GetMeshElement()->GetEdge(start, end, i);
        //     if(LineSegmentIntersectsCube(start, end, cube))
        //     {
        //         return true;
        //     }
        // }
        //
        // //finally we need to check if the mesh element clips a corner
        // //in this case, no edges of the element will intersect the cube,
        // //but at least one edge of the cube should intersect the mesh element
        // //probably should find a more elegant way to do this but for now use brute force
        // KThreeVector temp;
        // //0->1 edge
        // start = KThreeVector(cube->GetCorner(0));
        // end = KThreeVector(cube->GetCorner(1));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //0->2 edge
        // end = KThreeVector(cube->GetCorner(2));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //0->4 edge
        // end = KThreeVector(cube->GetCorner(4));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //1->3 edge
        // start = KThreeVector(cube->GetCorner(1));
        // end = KThreeVector(cube->GetCorner(3));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //1->5 edge
        // end = KThreeVector(cube->GetCorner(5));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //2->3 edge
        // start = KThreeVector(cube->GetCorner(2));
        // end = KThreeVector(cube->GetCorner(3));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //2->6 edge
        // end = KThreeVector(cube->GetCorner(6));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //3->7 edge
        // start = KThreeVector(cube->GetCorner(3));
        // end = KThreeVector(cube->GetCorner(7));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //4->5 edge
        // start = KThreeVector(cube->GetCorner(4));
        // end = KThreeVector(cube->GetCorner(5));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //4->6 edge
        // end = KThreeVector(cube->GetCorner(6));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //5->7 edge
        // start = KThreeVector(cube->GetCorner(5));
        // end = KThreeVector(cube->GetCorner(7));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};
        //
        // //6->7 edge
        // start = KThreeVector(cube->GetCorner(6));
        // end = KThreeVector(cube->GetCorner(7));
        // if(element->GetMeshElement()->NearestIntersection(start, end, temp)){return true;};

        //element is close but has no intersection with cube
        return false;

    }
    else
    {
        return false;
    }
}

bool
KGInsertionCondition::ElementEnclosedByCube(const KGNavigableMeshElement* element, const KGCube<KGMESH_DIM>* cube) const
{
    if(element != NULL && cube !=NULL)
    {
        KGPointCloud<KGMESH_DIM> point_cloud = element->GetMeshElement()->GetPointCloud();
        fBoundaryCalculator.Reset();
        fBoundaryCalculator.AddPointCloud(&point_cloud);
        KGAxisAlignedBox<KGMESH_DIM> aabb = fBoundaryCalculator.GetMinimalBoundingBox();

        for(unsigned int i=0; i<point_cloud.GetNPoints(); i++)
        {
            if(!(cube->PointIsInside(point_cloud.GetPoint(i)) ) )
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool
KGInsertionCondition::LineSegmentIntersectsCube(KThreeVector start, KThreeVector end, const KGCube<KGMESH_DIM>* cube) const
{
    //uses 'slab clipping' to check if line segment intersects the cube
    //as specified in chapter 11.2 of Geometric Tools for Computer Graphics
    //by Schneider & Eberly

    KThreeVector lower_corner( cube->GetCorner(0) );
    KThreeVector upper_corner( cube->GetCorner(7) );

    KThreeVector direction = (end - start);
    double length = direction.Magnitude();
    direction = direction.Unit();

    //compute displacement from upper and lower corners
    KThreeVector lower_diplacement = lower_corner - start;
    KThreeVector upper_diplacement = upper_corner - start;

    double t_min = 0;
    double t_max = length;
    double tmp;

    //check x planes
    double t0x = lower_diplacement.X()/direction.X();
    double t1x = upper_diplacement.X()/direction.X();
    //swap if out of order
    if(t0x > t1x){tmp = t0x; t0x = t1x; t1x = tmp;};

    //update valid interval
    if(t0x > t_min){t_min = t0x;};
    if(t1x < t_max){t_max = t1x;};

    //check y planes
    double t0y = lower_diplacement.Y()/direction.Y();
    double t1y = upper_diplacement.Y()/direction.Y();
    //swap if out of order
    if(t0y > t1y){tmp = t0y; t0y = t1y; t1y = tmp;};

    //update valid interval
    if(t0y > t_min){t_min = t0y;};
    if(t1y < t_max){t_max = t1y;};

    //check z planes
    double t0z = lower_diplacement.Z()/direction.Z();
    double t1z = upper_diplacement.Z()/direction.Z();
    //swap if out of order
    if(t0z > t1z){tmp = t0z; t0z = t1z; t1z = tmp;};

    //update valid interval
    if(t0z > t_min){t_min = t0z;};
    if(t1z < t_max){t_max = t1z;};

    //interval is invalid, no intersection
    if(t_min > t_max)
    {
        return false;
    }
    return true;
}



}
