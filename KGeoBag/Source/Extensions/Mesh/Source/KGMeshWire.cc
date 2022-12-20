#include "KGMeshWire.hh"

#include <cmath>

namespace KGeoBag
{
KGMeshWire::KGMeshWire(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, const double& diameter) :
    fP0(p0),
    fP1(p1),
    fDiameter(diameter)
{}
KGMeshWire::~KGMeshWire() = default;

double KGMeshWire::Area() const
{
    return (.5 * M_PI * fDiameter + (fP1 - fP0).Magnitude()) * M_PI * fDiameter;
}
double KGMeshWire::Aspect() const
{
    return ((fP1 - fP0).Magnitude()) / fDiameter;
}
katrin::KThreeVector KGMeshWire::Centroid() const
{
    return (fP0 + fP1) * .5;
}

void KGMeshWire::Transform(const katrin::KTransformation& transform)
{
    transform.Apply(fP0);
    transform.Apply(fP1);
}

double KGMeshWire::NearestDistance(const katrin::KThreeVector& aPoint) const
{
    katrin::KThreeVector del = aPoint - fP0;
    katrin::KThreeVector diff = fP1 - fP0;
    double len = diff.Magnitude();
    double dot = (del * diff) / (len * len);


    //here we fudge things by making
    //the assumption that the diameter is negligible
    //compared to the length of the wire
    //(true nearest distance should be to a cylinder, rather than a line segment)

    //nearest point is p0
    if (dot < 0.) {
        return del.Magnitude();
    }
    //nearest point is p1
    if (dot > 1.) {
        return (aPoint - fP1).Magnitude();
    }

    katrin::KThreeVector nearestpoint = fP0 + dot * diff;
    double dist = (aPoint - nearestpoint).Magnitude();

    //minor correction for wire diameter
    if (dist < fDiameter) {
        dist = 0;
    }
    else {
        dist -= fDiameter;
    }

    return dist;
}

katrin::KThreeVector KGMeshWire::NearestPoint(const katrin::KThreeVector& aPoint) const
{
    katrin::KThreeVector del = aPoint - fP0;
    katrin::KThreeVector diff = fP1 - fP0;
    double len = diff.Magnitude();
    double dot = (del * diff) / (len * len);
    if (dot < 0.) {
        return fP0;
    }
    if (dot > 1.) {
        return fP1;
    }
    //correction for the wire diameter
    katrin::KThreeVector point_on_axis = fP0 + dot * diff;
    katrin::KThreeVector normal = (aPoint - point_on_axis).Unit();
    return point_on_axis + fDiameter * normal;
}

katrin::KThreeVector KGMeshWire::NearestNormal(const katrin::KThreeVector& aPoint) const
{
    katrin::KThreeVector del = aPoint - fP0;
    katrin::KThreeVector diff = fP1 - fP0;
    double len = diff.Magnitude();
    double dot = (del * diff) / (len * len);
    katrin::KThreeVector point_on_axis = fP0 + dot * diff;
    return (aPoint - point_on_axis).Unit();  //normal vector always points away from axis
}

double KGMeshWire::ClosestApproach(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd) const
{
    //computes the closest distance between the line segment specified by aStart and anEnd and this
    //t is the parameter associated with the segment under test
    //s is the parameter asscoiated with this line segment

    katrin::KThreeVector fDiff = fP1 - fP0;
    katrin::KThreeVector tDiff = anEnd - aStart;
    katrin::KThreeVector w = aStart - fP0;
    double a = fDiff * fDiff;
    double b = tDiff * fDiff;
    double c = tDiff * tDiff;
    double det = a * c - b * b;

    if (std::fabs(det) >= 1e-12) {
        double d = fDiff * w;
        double e = tDiff * w;
        double t = (a * e - b * d) / (det);
        double s = (b * e - c * d) / (det);

        katrin::KThreeVector Ps;
        katrin::KThreeVector Qt;

        if (s <= 1.) {
            if (s >= 0.) {
                Ps = fP0 + s * fDiff;
            }  //Ps is on this line segment
            else {
                Ps = fP0;
            }  //Ps is below fP1
        }
        else {
            Ps = fP1;
        }  //Ps is above fP2

        if (t <= 1.) {
            if (t >= 0.) {
                Qt = aStart + t * tDiff;
            }  //Qt is on line segment under test
            else {
                Qt = aStart;
            }  //Qt is below aStart
        }
        else {
            Qt = anEnd;
        }  //Qt is above anEnd

        return (Ps - Qt).Magnitude();
    }
    else  //lines are parallel
    {
        return (fP0 - aStart).Magnitude();
    }
}

bool KGMeshWire::NearestIntersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                                     katrin::KThreeVector& anIntersection) const
{
    double closest_approach = ClosestApproach(aStart, anEnd);
    if (closest_approach > fDiameter) {
        //no intersection
        return false;
    }

    //have to solve quadratic equation for cylinder/line intersection
    //equation for an infinite cylinder directed along
    //the unit vector A, through point P, with a radius R
    //is given by all points Q such that :
    //[ (Q - P) x A ]^2 = R^2
    //for the intersection with a line, we have Q = aStart + (anEnd - aStart)*t
    katrin::KThreeVector fDiff = fP1 - fP0;
    katrin::KThreeVector tDiff = anEnd - aStart;
    katrin::KThreeVector g = tDiff.Cross(fDiff.Unit());
    katrin::KThreeVector h = (aStart - fP0).Cross(fDiff.Unit());

    double a = g * g;
    double b = 2.0 * g * h;
    double c = h * h - fDiameter * fDiameter;
    double disc = b * b - 4.0 * a * c;

    if (disc < 0.) {
        //no intersections at all
        return false;
    }

    double t1 = (-1.0 * b - std::sqrt(disc)) / (2.0 * a);
    double t2 = (-1.0 * b + std::sqrt(disc)) / (2.0 * a);

    bool inter1_possible = false;
    katrin::KThreeVector inter1;

    bool inter2_possible = false;
    katrin::KThreeVector inter2;

    if ((t1 <= 1.0) && (0. <= t1)) {
        //solution given by t1 is on line segment so compute the point
        inter1 = aStart + t1 * tDiff;
        inter1_possible = true;
    }

    if ((t2 <= 1.0) && (0. <= t2)) {
        //solution given by t2 is on line segment so compute the point
        inter2 = aStart + t2 * tDiff;
        inter2_possible = true;
    }


    if (inter1_possible) {
        double dist;
        dist = (inter1 - fP0) * fDiff.Unit();
        //check to make sure the point inter1 is on the cylinder
        if (dist >= 0.0 && dist <= fDiff.Magnitude()) {
            inter1_possible = true;
        }
        else {
            inter1_possible = false;
        }
    }

    if (inter2_possible) {
        //check to make sure the point inter1 is on the cylinder
        double dist;
        dist = (inter2 - fP0) * fDiff.Unit();
        //check to make sure the point inter1 is on the cylinder
        if (dist >= 0.0 && dist <= fDiff.Magnitude()) {
            inter2_possible = true;
        }
        else {
            inter2_possible = false;
        }
    }


    if (inter1_possible) {
        if (inter2_possible) {
            //both intersections are possible, return the point nearest to the start
            if (t1 <= t2) {
                anIntersection = inter1;
                return true;
            }
            else {
                anIntersection = inter2;
                return true;
            }
        }
        else {
            //only the t1 intersection is on both line segment and cylinder surface
            anIntersection = inter1;
            return true;
        }
    }
    else {
        if (inter2_possible) {
            //only the t2 intersection is on both line segment and cylinder surface
            anIntersection = inter2;
            return true;
        }
        else {
            //no intersections are both on the line segment and cylinder surface
            return false;
        }
    }

    //should never reach here, something failed
    return false;
}


KGPointCloud<KGMESH_DIM> KGMeshWire::GetPointCloud() const
{
    KGPointCloud<KGMESH_DIM> point_cloud;
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0));
    point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP1));
    return point_cloud;
}


void KGMeshWire::GetEdge(katrin::KThreeVector& start, katrin::KThreeVector& end, unsigned int /*index*/) const
{
    start = fP0;
    end = fP1;
}

}  // namespace KGeoBag
