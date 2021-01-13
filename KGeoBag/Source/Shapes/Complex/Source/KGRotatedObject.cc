#include "KGRotatedObject.hh"

#include "KRandom.h"

#include <limits>

namespace KGeoBag
{
KGRotatedObject::~KGRotatedObject()
{
    for (auto& segment : fSegments)
        delete segment;
}

KGRotatedObject* KGRotatedObject::Clone() const
{
    auto* tClone = new KGRotatedObject();
    tClone->fP1[0] = fP1[0];
    tClone->fP1[1] = fP1[1];
    tClone->fP2[0] = fP2[0];
    tClone->fP2[1] = fP2[1];
    tClone->fNPolyBegin = fNPolyBegin;
    tClone->fNPolyEnd = fNPolyEnd;
    tClone->fNSegments = fNSegments;
    tClone->fDiscretizationPower = fDiscretizationPower;

    for (auto* segment : fSegments)
        tClone->fSegments.push_back(segment->Clone(tClone));

    return tClone;
}

void KGRotatedObject::AddLine(const double p1[2], const double p2[2])
{
    // Adds line segment (p1,p2) to the rotated surface.

    for (int i = 0; i < 2; i++) {
        if (fNSegments == 0)
            fP1[i] = p1[i];
        fP2[i] = p2[i];
    }

    fSegments.push_back(new KGRotatedObject::Line(this, p1, p2));
    fSegments.back()->SetOrder(fNSegments++);
}

void KGRotatedObject::AddArc(const double p1[2], const double p2[2], const double radius,
                             const bool positiveOrientation)
{
    // Adds arc segment (p1,p2) to the rotated surface.

    for (int i = 0; i < 2; i++) {
        if (fNSegments == 0)
            fP1[i] = p1[i];
        fP2[i] = p2[i];
    }

    fSegments.push_back(new KGRotatedObject::Arc(this, p1, p2, radius, positiveOrientation));
    fSegments.back()->SetOrder(fNSegments++);
}

void KGRotatedObject::AddSegment(KGRotatedObject::Line* line)
{
    // Adds line segment (p1,p2) to the rotated surface.

    line->SetRotated(this);
    fSegments.push_back(line);
    fSegments.back()->SetOrder(fNSegments++);
}

bool KGRotatedObject::ContainsPoint(const double* P) const
{
    // Determines if point <aPoint> is contained by the geometry.

    for (auto* segment : fSegments)
        if (segment->ContainsPoint(P))
            return true;

    return false;
}

double KGRotatedObject::Area() const
{
    double area = 0.;
    for (auto* segment : fSegments)
        area += segment->Area();
    return area;
}

double KGRotatedObject::Volume() const
{
    double volume = 0.;
    for (auto* segment : fSegments)
        volume += segment->Volume();
    return volume;
}

double KGRotatedObject::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    double distance = std::numeric_limits<double>::max();

    double distance1;
    double P_in1[3];
    double P_norm1[3];

    for (auto* segment : fSegments) {
        distance1 = segment->DistanceTo(P, P_in1, P_norm1);
        if (distance1 < distance) {
            for (unsigned int j = 0; j < 3; j++) {
                if (P_in)
                    P_in[j] = P_in1[j];
                if (P_norm)
                    P_norm[j] = P_norm1[j];
            }
        }
    }
    return distance;
}

KGRotatedObject::Line::Line(KGRotatedObject* rO, const double p1[2], const double p2[2])
{
    fRotated = rO;

    for (int i = 0; i < 2; i++) {
        fP1[i] = p1[i];
        fP2[i] = p2[i];
    }
    Initialize();
}

void KGRotatedObject::Line::Initialize() const
{
    // Sets geometry-specific parameters used for discretization and navigation.

    double zmax = fP2[0];
    double rmax = fP2[1];
    double zmin = fP1[0];
    double rmin = fP1[1];

    if (zmax < zmin) {
        zmax = zmin;
        rmax = rmin;
        zmin = fP2[0];
        rmin = fP2[1];
    }

    // fLengthSquared: the squared length of the flat section of the unrolled
    // conic section
    fLengthSquared = ((fP2[0] - fP1[0]) * (fP2[0] - fP1[0]) + (fP2[1] - fP1[1]) * (fP2[1] - fP1[1]));

    // fLength: the length of the flat section of the unrolled conic section
    fLength = sqrt(fLengthSquared);

    // fAlpha: half the opening angle
    if (fabs(fP2[0] - fP1[0]) < 1.e-10)
        fAlpha = M_PI / 2.;
    else
        fAlpha = atan(fabs((rmax - rmin) / (zmax - zmin)));

    if (fAlpha > 1.e-6)  // we are not dealing with a cylinder
    {
        // fOpensUp: bool that states whether the conic section opens up in z
        if (rmax > rmin)
            fOpensUp = true;
        else
            fOpensUp = false;

        // zIntercept: the r-position where the generating line reaches z=0
        if (fabs(zmax - zmin) < 1.e-10)
            fZIntercept = zmax;
        else
            fZIntercept = (rmin - ((rmax - rmin) / (zmax - zmin)) * zmin);
        ;

        // fUnrolledRadius2: the radius of the outer boundary of the unrolled
        // conic section
        fUnrolledRadius2 = (fP1[1] > fP2[1] ? fP1[1] : fP2[1]) / sin(fAlpha);
        fUnrolledRadius2Squared = fUnrolledRadius2 * fUnrolledRadius2;

        // fTheta: the angle subtended by the unrolled conic section
        fTheta = 2. * M_PI * fUnrolledRadius2 / (fP1[1] > fP2[1] ? fP1[1] : fP2[1]);

        // fUnrolledRadius1: the radius of the inner boundary of the unrolled
        // conic section
        fUnrolledRadius1 = fUnrolledRadius2 - fLength;
        fUnrolledRadius1Squared = fUnrolledRadius1 * fUnrolledRadius1;

        // fUnrolledBoundingBox: (x,y) of min, max of box that bounds the unrolled
        // conic section
        fUnrolledBoundingBox[0] = -fUnrolledRadius2;  // min x
        if (fTheta < M_PI * .5)
            fUnrolledBoundingBox[0] = fUnrolledRadius1 * cos(fTheta);
        else if (fTheta < M_PI)
            fUnrolledBoundingBox[0] = fUnrolledRadius2 * cos(fTheta);
        fUnrolledBoundingBox[1] = 0.;  // min y
        if (fTheta > M_PI) {
            if (fTheta < 3. * M_PI / 2.)
                fUnrolledBoundingBox[1] = fUnrolledRadius2 * sin(fTheta);
            else
                fUnrolledBoundingBox[1] = -fUnrolledRadius2;
        }
        fUnrolledBoundingBox[2] = fUnrolledRadius2;  // max x
        fUnrolledBoundingBox[3] = fUnrolledRadius2;  // max y
        if (fTheta < M_PI * .5)
            fUnrolledBoundingBox[3] = fUnrolledRadius2 * sin(fTheta);
    }

    // fVolume: the volume of the conic section
    fVolume = (1. / 3. * M_PI * (zmax - zmin) * (rmin * rmin + rmax * rmax + rmin * rmax));

    // fArea: the area of the sheath of the conic section
    fArea = M_PI * (fP1[1] + fP2[1]) * fLength;
}

KGRotatedObject::Line* KGRotatedObject::Line::Clone(KGRotatedObject* eO) const
{
    auto* tClone = new Line();

    tClone->fOrder = fOrder;
    tClone->fP1[0] = fP1[0];
    tClone->fP1[1] = fP1[1];
    tClone->fP2[0] = fP2[0];
    tClone->fP2[1] = fP2[1];
    tClone->fNPolyBegin = fNPolyBegin;
    tClone->fNPolyEnd = fNPolyEnd;
    tClone->fAlpha = fAlpha;
    tClone->fTheta = fTheta;
    for (unsigned int i = 0; i < 4; i++)
        tClone->fUnrolledBoundingBox[i] = fUnrolledBoundingBox[i];
    tClone->fUnrolledRadius2 = fUnrolledRadius2;
    tClone->fUnrolledRadius2Squared = fUnrolledRadius2Squared;
    tClone->fUnrolledRadius1 = fUnrolledRadius1;
    tClone->fUnrolledRadius1Squared = fUnrolledRadius1Squared;
    tClone->fLengthSquared = fLengthSquared;
    tClone->fLength = fLength;
    tClone->fZIntercept = fZIntercept;
    tClone->fOpensUp = fOpensUp;
    tClone->fVolume = fVolume;
    tClone->fArea = fArea;
    tClone->fRotated = eO;

    return tClone;
}

bool KGRotatedObject::Line::ContainsPoint(const double* P) const
{
    // Determines whether or not a point is contained by the conic section

    double zmax = fP2[0];
    double rmax = fP2[1];
    double zmin = fP1[0];
    double rmin = fP1[1];

    if (zmax < zmin) {
        zmax = zmin;
        rmax = rmin;
        zmin = fP2[0];
        rmin = fP2[1];
    }

    if (P[2] < zmin || P[2] > zmax)
        return false;

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);

    double r_line;
    if (zmin == zmax)
        r_line = (rmin > rmax ? rmin : rmax);
    else
        r_line = rmin + (rmax - rmin) * (P[2] - zmin) / (zmax - zmin);

    if (r > r_line)
        return false;

    return true;
}

double KGRotatedObject::Line::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    if ((P_norm) && !(P_in)) {
        double p_in[3];
        return DistanceTo(P, p_in, P_norm);
    }

    double distance;

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);

    double u = ((r - fP1[1]) * (fP2[1] - fP1[1]) + (P[2] - fP1[0]) * (fP2[0] - fP1[0])) / (fLength * fLength);

    double cos = 0., sin = 0.;
    if (P_in) {
        cos = P[0] / sqrt(P[0] * P[0] + P[1] * P[1]);
        sin = P[1] / sqrt(P[0] * P[0] + P[1] * P[1]);
    }

    if (u <= 0.) {
        distance = sqrt((r - fP1[1]) * (r - fP1[1]) + (P[2] - fP1[0]) * (P[2] - fP1[0]));
        if (P_in) {
            P_in[0] = fP1[1] * cos;
            P_in[1] = fP1[1] * sin;
            P_in[2] = fP1[0];
        }
    }
    else if (u >= 1.) {
        distance = sqrt((r - fP2[1]) * (r - fP2[1]) + (P[2] - fP2[0]) * (P[2] - fP2[0]));
        if (P_in) {
            P_in[0] = fP2[1] * cos;
            P_in[1] = fP2[1] * sin;
            P_in[2] = fP2[0];
        }
    }
    else {
        double r_int = fP1[1] + u * (fP2[1] - fP1[1]);
        double z_int = fP1[0] + u * (fP2[0] - fP1[0]);

        distance = sqrt((r - r_int) * (r - r_int) + (P[2] - z_int) * (P[2] - z_int));

        if (P_in) {
            P_in[0] = r_int * cos;
            P_in[1] = r_int * sin;
            P_in[2] = z_int;
        }
    }

    if (P_norm) {
        // Returns the normal vector to the surface (index = 0,3) at the point
        // <P> on the surface.  The normal points outwards if the vector
        // pointing from <fP1> to <fP2> is in the first or fourth quadrant in the
        // r-z plane and points inwards if it is in the second or third quadrant.

        // First, we figure out the unit normal if it were lying in the x-z plane.
        P_norm[1] = 0.;

        // If the line describing the CS is vertical...
        if (fabs(fP1[1] - fP2[1]) < 1.e-10) {
            P_norm[0] = 0.;
            // ... and the line is pointed in positive x, the unit normal points in
            // negative z.  Otherwise, it points in positive z.
            if (fP1[0] < fP2[0])
                P_norm[2] = -1.;
            else
                P_norm[2] = 1.;
        }
        // Otherwise, if the line describing the CS is horizontal...
        else if (fabs(fP1[0] - fP2[0]) < 1.e-10) {
            P_norm[2] = 0.;
            // ... and the line is pointed in positive z, the unit normal points in
            // positive x.  Otherwise, it points in negative x.
            if (fP1[1] < fP2[1])
                P_norm[0] = 1.;
            else
                P_norm[0] = -1.;
        }
        // Otherwise, the unit normal is just the negative slope of the generating
        // line.
        else {
            P_norm[0] = ((fP1[1] - fP2[1]) / (fP2[0] - fP1[0])) /
                        sqrt(1. + ((fP1[1] - fP2[1]) / (fP2[0] - fP1[0])) * ((fP1[1] - fP2[1]) / (fP2[0] - fP1[0])));
            P_norm[1] = 0.;
            P_norm[2] =
                1. / sqrt(1. + ((fP1[1] - fP2[1]) / (fP2[0] - fP1[0])) * ((fP1[1] - fP2[1]) / (fP2[0] - fP1[0])));
        }

        // Now that the normal in the x-z plane is sorted out, we rotate to get
        // x, y sorted out.
        double r = sqrt(P_in[0] * P_in[0] + P_in[1] * P_in[1]);
        P_norm[1] = P_norm[0] * P_in[1] / r;
        P_norm[0] = P_norm[0] * P_in[0] / r;
    }
    return distance;
}

KGRotatedObject::Arc::Arc(KGRotatedObject* rO, const double p1[2], const double p2[2], const double radius,
                          const bool positiveOrientation) :
    KGRotatedObject::Line(rO, p1, p2),
    fRadius(radius),
    fPositiveOrientation(positiveOrientation)
{
    Initialize();
}

void KGRotatedObject::Arc::Initialize() const
{
    KGRotatedObject::Line::Initialize();

    FindCenter();

    // fArea: the area of the sheath of the rotated arc
    fArea = 2. * M_PI *
            (fCenter[1] * fRadius * (fPhiEnd - fPhiStart) + fRadius * fRadius * (cos(fPhiStart) - cos(fPhiEnd)));

    // fVolume: the volume of the rotated arc.  Hooray calculus!
    fVolume = -fRadius * M_PI / 12. *
              ((-3. * (4. * fCenter[1] * fCenter[1] + 3. * fRadius * fRadius * fRadius) * cos(fPhiEnd) +
                fRadius * (fRadius * cos(3. * fPhiEnd) - 6. * fCenter[1] * (-2. * fPhiEnd + sin(2. * fPhiEnd)))) -
               (-3. * (4. * fCenter[1] * fCenter[1] + 3. * fRadius * fRadius * fRadius) * cos(fPhiStart) +
                fRadius * (fRadius * cos(3. * fPhiStart) - 6. * fCenter[1] * (-2. * fPhiStart + sin(2. * fPhiStart)))));

    fPhiStart = 0;

    if (fabs(fabs(fP1[0] - fCenter[0]) - fRadius) < 1.e-6) {
        if (fP1[0] > fCenter[0])
            fPhiStart = 0.;
        else
            fPhiStart = M_PI;
    }
    else
        fPhiStart = acos((fP1[0] - fCenter[0]) / fRadius);

    if (fP1[1] < fCenter[1])
        fPhiStart = 2. * M_PI - fPhiStart;

    fPhiEnd = 0;

    if (fabs(fabs(fP2[0] - fCenter[0]) - fRadius) < 1.e-6) {
        if (fP2[0] > fCenter[0])
            fPhiEnd = 0.;
        else
            fPhiEnd = M_PI;
    }
    else
        fPhiEnd = acos((fP2[0] - fCenter[0]) / fRadius);

    if (fP2[1] < fCenter[1])
        fPhiEnd = 2. * M_PI - fPhiEnd;

    if (fPositiveOrientation && fPhiStart > fPhiEnd)
        fPhiEnd += 2. * M_PI;

    if (!fPositiveOrientation && fPhiStart < fPhiEnd)
        fPhiStart += 2. * M_PI;

    fPhiMid = (fPhiStart + fPhiEnd) * .5;

    fPhiBoundary = fPhiMid + M_PI;

    if ((fP1[1] < fCenter[1] && fP2[1] > fCenter[1] && !fPositiveOrientation) ||
        (fP1[1] > fCenter[1] && fP2[1] < fCenter[1] && fPositiveOrientation))
        fRMax = fCenter[0] + fRadius;
    else
        fRMax = (fP1[0] > fP2[0] ? fP1[0] : fP2[0]);
}

KGRotatedObject::Arc* KGRotatedObject::Arc::Clone(KGRotatedObject* rO) const
{
    auto* tClone = new Arc();

    tClone->fOrder = fOrder;
    tClone->fP1[0] = fP1[0];
    tClone->fP1[1] = fP1[1];
    tClone->fP2[0] = fP2[0];
    tClone->fP2[1] = fP2[1];
    tClone->fNPolyBegin = fNPolyBegin;
    tClone->fNPolyEnd = fNPolyEnd;
    tClone->fAlpha = fAlpha;
    tClone->fTheta = fTheta;
    for (unsigned int i = 0; i < 4; i++)
        tClone->fUnrolledBoundingBox[i] = fUnrolledBoundingBox[i];
    tClone->fUnrolledRadius2 = fUnrolledRadius2;
    tClone->fUnrolledRadius2Squared = fUnrolledRadius2Squared;
    tClone->fUnrolledRadius1 = fUnrolledRadius1;
    tClone->fUnrolledRadius1Squared = fUnrolledRadius1Squared;
    tClone->fLengthSquared = fLengthSquared;
    tClone->fLength = fLength;
    tClone->fZIntercept = fZIntercept;
    tClone->fOpensUp = fOpensUp;
    tClone->fVolume = fVolume;
    tClone->fArea = fArea;

    tClone->fRadius = fRadius;
    tClone->fCenter[0] = fCenter[0];
    tClone->fCenter[1] = fCenter[1];
    tClone->fPhiStart = fPhiStart;
    tClone->fPhiEnd = fPhiEnd;
    tClone->fPhiMid = fPhiMid;
    tClone->fPhiBoundary = fPhiBoundary;
    tClone->fRMax = fRMax;
    tClone->fPositiveOrientation = fPositiveOrientation;

    tClone->fRotated = rO;

    return tClone;
}

bool KGRotatedObject::Arc::ContainsPoint(const double* P) const
{
    double zmax = fP2[0];
    double zmin = fP1[0];

    bool arcOpensUp = (fPhiBoundary > 0 && fPhiBoundary < M_PI);

    if (zmax < zmin) {
        zmax = zmin;
        zmin = fP2[0];
    }

    if (P[2] < zmin || P[2] > zmax)
        return false;

    // From here, compare against the distance b/t center and transformed point

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);

    if (arcOpensUp) {
        if (r >= fCenter[1])
            return false;

        if (sqrt((r - fCenter[1]) * (r - fCenter[1]) + (P[2] - fCenter[0]) * (P[2] - fCenter[0])) < fRadius)
            return false;

        return true;
    }
    else {
        if (r <= fCenter[1])
            return true;

        if (sqrt((r - fCenter[1]) * (r - fCenter[1]) + (P[2] - fCenter[0]) * (P[2] - fCenter[0])) < fRadius)
            return true;

        return false;
    }
}

double KGRotatedObject::Arc::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    if ((P_norm) && !(P_in)) {
        double p_in[3];
        return DistanceTo(P, p_in, P_norm);
    }

    double distance;

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    if (r < 1.e-18)
        r = 1.e-18;
    double z = P[2];

    double phi_P = atan((r - fCenter[1]) / (z - fCenter[0]));
    if (fCenter[0] > z)
        phi_P += M_PI;

    phi_P = NormalizeAngle(phi_P);

    if (AngleIsWithinRange(phi_P, fPhiStart, fPhiEnd, fPositiveOrientation)) {
        distance = fabs(sqrt((fCenter[1] - r) * (fCenter[1] - r) + (fCenter[0] - z) * (fCenter[0] - z)) - fRadius);

        if (P_in) {
            double r_closest = fRadius * sin(phi_P) + fCenter[1];
            double z_closest = fRadius * cos(phi_P) + fCenter[0];

            double cosine = P[0] / r;
            double sine = P[1] / r;
            if (r <= 1.e-16)
                cosine = 1.;
            P_in[0] = cosine * r_closest;
            P_in[1] = sine * r_closest;
            P_in[2] = z_closest;
        }
    }
    else if (AngleIsWithinRange(phi_P, fPhiEnd, fPhiBoundary, fPositiveOrientation)) {
        distance = sqrt((r - fP2[1]) * (r - fP2[1]) + (z - fP2[0]) * (z - fP2[0]));

        if (P_in) {
            double cosine = P[0] / r;
            double sine = P[1] / r;
            if (r <= 1.e-16)
                cosine = 1.;
            P_in[0] = cosine * fP2[1];
            P_in[1] = sine * fP2[1];
            P_in[2] = fP2[0];
        }
    }
    else {
        distance = sqrt((r - fP1[1]) * (r - fP1[1]) + (z - fP1[0]) * (z - fP1[0]));

        if (P_in) {
            double cosine = P[0] / r;
            double sine = P[1] / r;
            if (r <= 1.e-16)
                cosine = 1.;
            P_in[0] = cosine * fP1[1];
            P_in[1] = sine * fP1[1];
            P_in[2] = fP1[0];
        }
    }

    if (P_norm) {
        double r = sqrt(P_in[0] * P_in[0] + P_in[1] * P_in[1]);

        double theta = atan((P_in[2] - fCenter[0]) / (r - fCenter[1]));
        double phi;
        if (fabs(P_in[0]) > 1.e-10)
            phi = atan(P_in[1] / P_in[0]);
        else if (P_in[1] == fabs(P_in[1]))
            phi = M_PI * .5;
        else
            phi = -M_PI * .5;
        P_norm[0] = sin(theta) * cos(phi);
        P_norm[1] = sin(theta) * sin(phi);
        P_norm[2] = cos(theta);
    }

    return distance;
}

void KGRotatedObject::Arc::FindCenter() const
{
    // Finds the center of the circle from which the arc is formed

    // midpoint between p1 and p2
    double pmid[2] = {(fP1[0] + fP2[0]) * .5, (fP1[1] + fP2[1]) * .5};

    // unit vector pointing from p1 to p2
    double unit[2] = {fP2[0] - fP1[0], fP2[1] - fP1[1]};
    double chord = sqrt(unit[0] * unit[0] + unit[1] * unit[1]);
    for (double& i : unit)
        i /= chord;

    // unit vector normal to line connecting p1 and p2
    double norm[2] = {-unit[1], unit[0]};

    if (!fPositiveOrientation)
        for (double& i : norm)
            i *= -1.;

    double theta = 2. * asin(chord / (2. * fRadius));

    for (int i = 0; i < 2; i++)
        fCenter[i] = pmid[i] + fRadius * norm[i] * cos(theta * .5);

    fPhiStart = 0;
    if (fabs(fabs(fP1[0] - fCenter[0]) - fRadius) < 1.e-6) {
        if (fP1[0] > fCenter[0])
            fPhiStart = 0.;
        else
            fPhiStart = M_PI;
    }
    else
        fPhiStart = acos((fP1[0] - fCenter[0]) / fRadius);

    if (fP1[1] < fCenter[1])
        fPhiStart = 2. * M_PI - fPhiStart;

    fPhiEnd = 0;

    if (fabs(fabs(fP2[0] - fCenter[0]) - fRadius) < 1.e-6) {
        if (fP2[0] > fCenter[0])
            fPhiEnd = 0.;
        else
            fPhiEnd = M_PI;
    }
    else
        fPhiEnd = acos((fP2[0] - fCenter[0]) / fRadius);

    if (fP2[1] < fCenter[1])
        fPhiEnd = 2. * M_PI - fPhiEnd;

    if (fPositiveOrientation && fPhiStart > fPhiEnd)
        fPhiEnd += 2. * M_PI;

    if (!fPositiveOrientation && fPhiStart < fPhiEnd)
        fPhiStart += 2. * M_PI;

    fPhiBoundary = (fPhiStart + fPhiEnd) * .5 + M_PI;

    // Here we re-assign our angles to be between 0 and 2.*M_PI

    fPhiStart = NormalizeAngle(fPhiStart);
    fPhiEnd = NormalizeAngle(fPhiEnd);
    fPhiBoundary = NormalizeAngle(fPhiBoundary);
}

double KGRotatedObject::Arc::GetLength() const
{
    // Returns the length of the arc.

    double chord = sqrt((fP2[0] - fP1[0]) * (fP2[0] - fP1[0]) + (fP2[1] - fP1[1]) * (fP2[1] - fP1[1]));
    double theta = 2. * asin(chord / (2. * fRadius));

    return fRadius * theta;
}

double KGRotatedObject::Arc::NormalizeAngle(double angle)
{
    double normalized_angle = angle;
    while (normalized_angle > 2. * M_PI)
        normalized_angle -= 2. * M_PI;
    while (normalized_angle < 0)
        normalized_angle += 2. * M_PI;
    return normalized_angle;
}

double KGRotatedObject::Arc::GetRadius(double z) const
{
    double z1 = (fP1[1] < fP2[1] ? fP1[1] : fP2[1]);
    double z2 = (fP1[1] > fP2[1] ? fP1[1] : fP2[1]);

    if (z < z1 || z > z2)
        return 0.;

    z -= fCenter[1];

    double r = sqrt(fRadius * fRadius - z * z);

    return fCenter[0] + (fPositiveOrientation ? 1. : -1.) * r;
}

bool KGRotatedObject::Arc::AngleIsWithinRange(double phi_test, double phi_min, double phi_max, bool positiveOrientation)
{
    // determines whether or not <phi_test> is sandwiched by <phi_min> and
    // <phi_max>.

    bool result;

    if (phi_min < phi_max)
        result = (phi_min < phi_test && phi_test < phi_max);
    else
        result = (phi_test > phi_min || phi_test < phi_max);

    if (!positiveOrientation)
        result = !result;
    return result;
}
}  // namespace KGeoBag
