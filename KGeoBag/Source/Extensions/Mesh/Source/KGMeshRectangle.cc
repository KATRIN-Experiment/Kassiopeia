#include "KGMeshRectangle.hh"

#include "KTwoVector.hh"

#include <cmath>
#include <vector>

#define RECTANGLE_EPS 1e-6

namespace KGeoBag
{
KGMeshRectangle::KGMeshRectangle(const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1,
                                 const KThreeVector& n2) :
    KGMeshElement(),
    fA(a),
    fB(b),
    fP0(p0),
    fN1(n1),
    fN2(n2)
{}
KGMeshRectangle::KGMeshRectangle(const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& /*p2*/,
                                 const KThreeVector& p3) :
    KGMeshElement()
{
    fP0 = p0;
    fN1 = p1 - p0;
    fA = fN1.Magnitude();
    fN1 = fN1.Unit();
    fN2 = p3 - p0;
    fB = fN2.Magnitude();
    fN2 = fN2.Unit();
}
KGMeshRectangle::KGMeshRectangle(const KGMeshRectangle& r) :
    KGMeshElement(r),
    fA(r.fA),
    fB(r.fB),
    fP0(r.fP0),
    fN1(r.fN1),
    fN2(r.fN2)
{}
KGMeshRectangle::~KGMeshRectangle() {}

double KGMeshRectangle::Area() const
{
    return fA * fB;
}
double KGMeshRectangle::Aspect() const
{
    if (fA > fB) {
        return fA / fB;
    }
    else {
        return fB / fA;
    }
}

void KGMeshRectangle::Transform(const KTransformation& transform)
{
    transform.Apply(fP0);
    transform.ApplyRotation(fN1);
    transform.ApplyRotation(fN2);
}

double KGMeshRectangle::NearestDistance(const KThreeVector& aPoint) const
{
    KThreeVector nearest = NearestPoint(aPoint);
    return (aPoint - nearest).Magnitude();
}

KThreeVector KGMeshRectangle::NearestPoint(const KThreeVector& aPoint) const
{
    KThreeVector del = aPoint - fP0;

    double x_coord = del.Dot(fN1);
    double y_coord = del.Dot(fN2);

    if (x_coord > fA) {
        x_coord = fA;
    };
    if (x_coord < 0) {
        x_coord = 0;
    };

    if (y_coord > fB) {
        y_coord = fB;
    };
    if (y_coord < 0) {
        y_coord = 0;
    };

    KThreeVector nearest = fP0 + x_coord * fN1 + y_coord * fN2;
    return nearest;
}

KThreeVector KGMeshRectangle::NearestNormal(const KThreeVector& /*aPoint*/) const
{
    return (fN1.Cross(fN2)).Unit();
}

bool KGMeshRectangle::NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd,
                                          KThreeVector& anIntersection) const
{
    //we compute the intersection with the plane associated with this rectangle

    //first construct the coordinate system for plane defined by rectangle
    //origin is defined to be at p0
    KThreeVector z_axis = (fN1.Cross(fN2)).Unit();

    KThreeVector v = (anEnd - aStart);
    double len = v.Magnitude();
    v = v.Unit();
    double ndotv = z_axis.Dot(v);

    if (fabs(ndotv) < RECTANGLE_EPS) {
        //The line segment is ~parallel to the plane, note the line segment
        //could be in the plane, but this results in a infinite number of
        //intersections we do not check for this, since with floating point
        //math this is pretty unlikely.
        return false;
    }

    double t = (z_axis.Dot(fP0) - z_axis.Dot(aStart)) / ndotv;

    if (t < 0) {
        //plane is behind the first point of the line segment
        //no intersection
        return false;
    }

    if (t > len) {
        //intersection is beyond the last point of the line
        //segment, so no intersection
        return false;
    }

    //passed simple checks, so there is a possible intersection given by;
    KThreeVector possible_inter = aStart + t * v;

    //project the possible interesction onto the rectangle
    KThreeVector del = possible_inter - fP0;
    KTwoVector projection(del.Dot(fN1), del.Dot(fN2));

    if ((projection.X() <= fA) && (projection.X() >= 0) && (projection.Y() <= fB) && (projection.Y() >= 0)) {
        //we have a true intersection
        anIntersection = possible_inter;
        return true;
    }

    return false;
}

KGPointCloud<KGMESH_DIM> KGMeshRectangle::GetPointCloud() const
{
    KGPointCloud<KGMESH_DIM> point_cloud;
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0));
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fA * fN1));
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fA * fN1 + fB * fN2));
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fB * fN2));
    return point_cloud;
}


void KGMeshRectangle::GetEdge(KThreeVector& start, KThreeVector& end, unsigned int index) const
{
    if (index == 0) {
        start = fP0;
        end = fP0 + fA * fN1;
        return;
    }

    if (index == 1) {
        start = fP0 + fA * fN1;
        end = fP0 + fA * fN1 + fB * fN2;
    }

    if (index == 2) {
        start = fP0 + fA * fN1 + fB * fN2;
        end = fP0 + fB * fN2;
    }

    if (index == 3) {
        start = fP0 + fB * fN2;
        end = fP0;
    }
}

bool KGMeshRectangle::SameSide(KThreeVector point, KThreeVector A, KThreeVector B, KThreeVector C) const
{
    KThreeVector cp1 = (B - A).Cross(point - A);
    KThreeVector cp2 = (B - A).Cross(C - A);
    if (cp1.Dot(cp2) > 0) {
        return true;
    }
    return false;
}

}  // namespace KGeoBag
