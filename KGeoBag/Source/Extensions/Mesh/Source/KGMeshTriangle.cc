#include "KGMeshTriangle.hh"

#include <cmath>
#include <vector>

#define TRIANGLE_EPS 1e-6

namespace KGeoBag
{
KGMeshTriangle::KGMeshTriangle(const double& a, const double& b, const katrin::KThreeVector& p0, const katrin::KThreeVector& n1,
                               const katrin::KThreeVector& n2) :
    fA(a), fB(b), fP0(p0), fN1(n1), fN2(n2)
{}
KGMeshTriangle::KGMeshTriangle(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, const katrin::KThreeVector& p2)
{
    fP0 = p0;
    fN1 = p1 - p0;
    fA = fN1.Magnitude();
    fN1 = fN1.Unit();
    fN2 = p2 - p0;
    fB = fN2.Magnitude();
    fN2 = fN2.Unit();
}
KGMeshTriangle::KGMeshTriangle(const KGTriangle& t) :
    fA(t.GetA()),
    fB(t.GetB()),
    fP0(t.GetP0()),
    fN1(t.GetN1()),
    fN2(t.GetN2())
{}
KGMeshTriangle::KGMeshTriangle(const KGMeshTriangle&) = default;
KGMeshTriangle::~KGMeshTriangle() = default;

double KGMeshTriangle::Area() const
{
    return .5 * fA * fB * fN1.Cross(fN2).Magnitude();
}
double KGMeshTriangle::Aspect() const
{

    double c, max;
    double delx, dely, delz;
    double p1[3];
    double p2[3];

    for (int i = 0; i < 3; i++) {
        p1[i] = fP0[i] + fA * fN1[i];
        p2[i] = fP0[i] + fB * fN2[i];
    }

    delx = p1[0] - p2[0];
    dely = p1[1] - p2[1];
    delz = p1[2] - p2[2];
    c = std::sqrt(delx * delx + dely * dely + delz * delz);

    katrin::KThreeVector PA;
    katrin::KThreeVector PB;
    katrin::KThreeVector PC;
    katrin::KThreeVector V;
    katrin::KThreeVector X;
    katrin::KThreeVector Y;
    katrin::KThreeVector Q;
    katrin::KThreeVector SUB;

    //find the longest size:
    if (fA > fB) {
        max = fA;
        PA = katrin::KThreeVector(p2[0], p2[1], p2[2]);
        PB = katrin::KThreeVector(fP0[0], fP0[1], fP0[2]);
        PC = katrin::KThreeVector(p1[0], p1[1], p1[2]);
    }
    else {
        max = fB;
        PA = katrin::KThreeVector(p1[0], p1[1], p1[2]);
        PB = katrin::KThreeVector(p2[0], p2[1], p2[2]);
        PC = katrin::KThreeVector(fP0[0], fP0[1], fP0[2]);
    }

    if (c > max) {
        max = c;
        PA = katrin::KThreeVector(fP0[0], fP0[1], fP0[2]);
        PB = katrin::KThreeVector(p1[0], p1[1], p1[2]);
        PC = katrin::KThreeVector(p2[0], p2[1], p2[2]);
    }

    //the line pointing along v is the y-axis
    V = PC - PB;
    Y = V.Unit();

    //q is closest point to fP[0] on line connecting fP[1] to fP[2]
    double t = (PA.Dot(V) - PB.Dot(V)) / (V.Dot(V));
    Q = PB + t * V;

    //the line going from fP[0] to fQ is the x-axis
    X = Q - PA;
    //gram-schmidt out any y-axis component in the x-axis
    double proj = X.Dot(Y);
    SUB = proj * Y;
    X = X - SUB;
    double H = X.Magnitude();  //compute triangle height along x

    //compute the triangles aspect ratio
    double ratio = max / H;

    return ratio;
}
katrin::KThreeVector KGMeshTriangle::Centroid() const
{
    return fP0 + (fA * fN1 + fB * fN2) / 3.;
}

void KGMeshTriangle::Transform(const katrin::KTransformation& transform)
{
    transform.Apply(fP0);
    transform.ApplyRotation(fN1);
    transform.ApplyRotation(fN2);
}

KGPointCloud<KGMESH_DIM> KGMeshTriangle::GetPointCloud() const
{
    KGPointCloud<KGMESH_DIM> point_cloud;
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0));
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fA * fN1));
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fB * fN2));
    return point_cloud;
}

void KGMeshTriangle::GetEdge(katrin::KThreeVector& start, katrin::KThreeVector& end, unsigned int index) const
{
    if (index == 0) {
        start = fP0;
        end = fP0 + fA * fN1;
        return;
    }

    if (index == 1) {
        start = fP0 + fA * fN1;
        end = fP0 + fB * fN2;
    }

    if (index == 2) {
        start = fP0 + fB * fN2;
        end = fP0;
    }
}

double KGMeshTriangle::NearestDistance(const katrin::KThreeVector& aPoint) const
{
    katrin::KThreeVector nearest = NearestPoint(aPoint);
    return (aPoint - nearest).Magnitude();
}

katrin::KThreeVector KGMeshTriangle::NearestPoint(const katrin::KThreeVector& aPoint) const
{
    //first construct the coordinate system for plane defined by triangle
    //origin is defined to be at p0
    katrin::KThreeVector z_axis = (fN1.Cross(fN2)).Unit();
    katrin::KThreeVector y_axis = z_axis.Cross(fN1);

    //triangle properties
    double height = fB * fN2.Dot(y_axis);
    double p2_x = fB * fN2.Dot(fN1);

    //2d proxies of triangle points
    katrin::KThreeVector p0_proxy(0, 0, 0);
    katrin::KThreeVector p1_proxy(fA, 0, 0);
    katrin::KThreeVector p2_proxy(p2_x, height, 0);

    //project the point into the plane of the triangle
    katrin::KThreeVector del = aPoint - fP0;
    katrin::KThreeVector proj(del.Dot(fN1), del.Dot(y_axis), 0);

    if (SameSide(proj, p0_proxy, p1_proxy, p2_proxy) && SameSide(proj, p1_proxy, p2_proxy, p0_proxy) &&
        SameSide(proj, p2_proxy, p0_proxy, p1_proxy)) {
        //projection onto triangle is the closest point
        return fP0 + (proj.X()) * fN1 + (proj.Y()) * y_axis;
    }

    //made it here, the nearest point on the plane is not inside the triangle
    //calculate distance to edges
    katrin::KThreeVector ab = NearestPointOnLineSegment(fP0, fP0 + fA * fN1, aPoint);
    double dist_ab = (ab - aPoint).Magnitude();
    katrin::KThreeVector bc = NearestPointOnLineSegment(fP0 + fA * fN1, fP0 + fB * fN2, aPoint);
    double dist_bc = (bc - aPoint).Magnitude();
    katrin::KThreeVector ca = NearestPointOnLineSegment(fP0 + fB * fN2, fP0, aPoint);
    double dist_ca = (ca - aPoint).Magnitude();

    if (dist_ab < dist_bc && dist_ab < dist_ca) {
        return ab;
    }

    if (dist_bc < dist_ab && dist_bc < dist_ca) {
        return bc;
    }

    return ca;
}

katrin::KThreeVector KGMeshTriangle::NearestNormal(const katrin::KThreeVector& /*aPoint*/) const
{
    return (fN1.Cross(fN2)).Unit();
}

bool KGMeshTriangle::NearestIntersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                                         katrin::KThreeVector& anIntersection) const
{
    //we compute the intersection with the plane associated with this triangle

    //first construct the coordinate system for plane defined by triangle
    //origin is defined to be at p0
    katrin::KThreeVector z_axis = (fN1.Cross(fN2)).Unit();
    katrin::KThreeVector y_axis = z_axis.Cross(fN1);

    katrin::KThreeVector v = (anEnd - aStart);
    double len = v.Magnitude();
    v = v.Unit();
    double ndotv = z_axis.Dot(v);

    if (fabs(ndotv) < TRIANGLE_EPS) {
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
    katrin::KThreeVector possible_inter = aStart + t * v;

    //project the possible interesction onto the triangle
    // katrin::KThreeVector del = possible_inter - fP0;
    // katrin::KTwoVector projection(del.Dot(fN1), del.Dot(y_axis));
    // katrin::KTwoVector proj_prime = projection - katrin::KTwoVector(fA, 0);

    // double eps = TRIANGLE_EPS*std::sqrt(fA*fB);
    // double neps = -1.0*eps;

    // double height = fB*fN2.Dot(y_axis);
    // double p2_x = fB*fN2.Dot(fN1);
    // double max_x = p2_x; if(fA > max_x){max_x = fA;};

    //triangle properties
    double height = fB * fN2.Dot(y_axis);
    double p2_x = fB * fN2.Dot(fN1);

    //2d proxies of triangle points
    katrin::KThreeVector p0_proxy(0, 0, 0);
    katrin::KThreeVector p1_proxy(fA, 0, 0);
    katrin::KThreeVector p2_proxy(p2_x, height, 0);

    //project the point into the plane of the triangle
    katrin::KThreeVector del = possible_inter - fP0;
    katrin::KThreeVector proj(del.Dot(fN1), del.Dot(y_axis), 0);

    if (SameSide(proj, p0_proxy, p1_proxy, p2_proxy) && SameSide(proj, p1_proxy, p2_proxy, p0_proxy) &&
        SameSide(proj, p2_proxy, p0_proxy, p1_proxy)) {
        //have intersection in triangle
        anIntersection = possible_inter;
        return true;
    }

    return false;
}


katrin::KThreeVector KGMeshTriangle::NearestPointOnLineSegment(const katrin::KThreeVector& a, const katrin::KThreeVector& b,
                                                       const katrin::KThreeVector& point)
{
    katrin::KThreeVector diff = b - a;
    double t = ((point - a) * diff);
    if (t < 0.) {
        return a;
    };
    if (t > 1.) {
        return b;
    };
    return a + t * diff;
}


bool KGMeshTriangle::SameSide(const katrin::KThreeVector& point, const katrin::KThreeVector& A, const katrin::KThreeVector& B,
                              const katrin::KThreeVector& C)
{
    katrin::KThreeVector cp1 = (B - A).Cross(point - A);
    katrin::KThreeVector cp2 = (B - A).Cross(C - A);
    if (cp1.Dot(cp2) > 0) {
        return true;
    }
    return false;
}

}  // namespace KGeoBag
