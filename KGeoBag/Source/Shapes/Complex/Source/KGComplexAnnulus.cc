#include "KGComplexAnnulus.hh"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace KGeoBag
{
KGComplexAnnulus::KGComplexAnnulus(double rmain) : fCoordTransform(nullptr)
{
    fRMain = rmain;
}

KGComplexAnnulus::~KGComplexAnnulus()
{
    if (fCoordTransform)
        delete fCoordTransform;

    for (unsigned int i = 0; i < fRings.size(); i++) {
        if (fRings.at(i))
            delete fRings.at(i);
    }
}

void KGComplexAnnulus::AddRing(double asub[2], double rsub)
{
    // Adds a Ring to the Annulus.
    auto* r = new KGComplexAnnulus::Ring(this, asub, rsub);
    fRings.push_back(r);
}

void KGComplexAnnulus::AddRing(KGComplexAnnulus::Ring* ring)
{
    // Adds a ring.

    ring->SetComplexAnnulus(this);

    fRings.push_back(ring);
}

KGComplexAnnulus* KGComplexAnnulus::Clone() const
{
    auto* a = new KGComplexAnnulus();
    a->fRMain = fRMain;
    a->fRadialMeshMain = fRadialMeshMain;
    a->fPolyMain = fPolyMain;

    a->fCoordTransform = new KGCoordinateTransform(*fCoordTransform);

    for (unsigned int i = 0; i < fRings.size(); i++)
        a->fRings.push_back(fRings.at(i)->Clone(a));
    return a;
}

void KGComplexAnnulus::Initialize() const
{
    fRadialMeshMain = 30;
    fPolyMain = 120;

    fCoordTransform = new KGCoordinateTransform();
    for (unsigned int i = 0; i < fRings.size(); i++)
        fRings.at(i)->Initialize();
}

bool KGComplexAnnulus::ContainsPoint(const double* P) const
{
    //Annulus is a 2D-Object, so the point must be on the same plane at the very least.
    //Annulus is initialized as an object rotated around the z-axis with a defined radius, so the z-coordinates must be equivalent.
    double epsilon = 1e-7;
    if (fabs(P[2]) > epsilon) {
        return false;
    }

    //Is it within radius of annulus?
    double pr = sqrt(P[0] * P[0] + P[1] * P[1]);
    if (pr > fRMain) {
        return false;
    }

    //Is the point in a hole?
    for (unsigned int i = 0; i < fRings.size(); i++) {
        if (fRings.at(i)->ContainsPoint(P))
            return false;
    }

    //Otherwise it must be contained.
    return true;
}

double KGComplexAnnulus::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    if (!P_in || !P_norm)
        return NAN;

    //Compute the closest point P_in to Point P as well as the norm vector pointing from point to closest point and returns the distance between them.

    double dist_main = 0;
    double Norm[3] = {0, 0, 1};  //Norm vector of any annulus plane...

    //Compute the distance from point to x,y-Plane that is lifted by ZMain off the Origin. (using Hessian Normal)

    double D = P[2] * Norm[2];

    //Compute the projected point on the plane and calculates it's distance towards the edge of the surface.

    double P_Plane[3];
    double PO[3];
    double PO_Norm_Sq = 0;

    for (unsigned int i = 0; i < 3; i++) {
        P_Plane[i] = P[i] - D * Norm[i];
        PO[i] = P_Plane[i];
        PO_Norm_Sq += PO[i] * PO[i];
    }

    double PO_Norm = sqrt(PO_Norm_Sq);

    //Get case when the projection is in the annulus.

    if (PO_Norm < fRMain) {
        for (unsigned int i = 0; i < 3; i++) {
            P_in[i] = P_Plane[i];
            if (D > 0) {
                P_norm[i] = -Norm[i];
            }
            else
                P_norm[i] = Norm[i];
        }
        dist_main = fabs(D);
        return dist_main;
    }

    //continuation of normal case.

    for (unsigned int i = 0; i < 3; i++) {
        PO[i] /= PO_Norm;
    }

    double t = fRMain;
    double NormSq = 0;

    for (unsigned int i = 0; i < 3; i++) {
        P_in[i] = t * PO[i];
        P_norm[i] = P_in[i] - P[i];
        NormSq += P_norm[i] * P_norm[i];
    }

    double Normalize = sqrt(NormSq);

    for (unsigned int i = 0; i < 3; i++) {
        P_norm[i] /= Normalize;
    }

    dist_main = Normalize;

    return dist_main;
}

KGComplexAnnulus::Ring::Ring(KGComplexAnnulus* complexAnnulus, double ASub[2], double RSub) : Ring(complexAnnulus)
{
    for (unsigned int i = 0; i < 2; i++) {
        fASub[i] = ASub[i];
    }
    fASub[2] = 0.;
    fRSub = RSub;

    Initialize();
}

KGComplexAnnulus::Ring::~Ring()
{
    if (fCoordTransform)
        delete fCoordTransform;
}

void KGComplexAnnulus::Ring::Initialize()
{
    fRadialMeshSub = 6;

    ComputeLocalFrame(fCen);

    fCoordTransform = new KGCoordinateTransform(fCen, fX_loc, fY_loc, fZ_loc);

    // for later calculations, we compute some parameters of the port here
    double aSub_loc[3];
    fCoordTransform->ConvertToLocalCoords(fASub, aSub_loc, false);

    double LengthSq = 0;

    for (int i = 0; i < 3; i++) {
        fNorm[i] = fASub[i] - fCen[i];
        LengthSq += fNorm[i] * fNorm[i];
    }

    double Length = sqrt(LengthSq);
    for (int i = 0; i < 3; i++)
        fNorm[i] /= Length;
}

KGComplexAnnulus::Ring* KGComplexAnnulus::Ring::Clone(KGComplexAnnulus* a) const
{
    auto* r = new Ring();

    r->fComplexAnnulus = a;

    for (unsigned int i = 0; i < 3; i++) {
        r->fCen[i] = fCen[i];
        r->KGComplexAnnulus::Ring::fNorm[i] = KGComplexAnnulus::Ring::fNorm[i];
        r->fNorm[i] = fNorm[i];
    }
    r->fCoordTransform = new KGCoordinateTransform(*fCoordTransform);

    r->fASub[0] = fASub[0];
    r->fASub[1] = fASub[1];
    r->fASub[2] = fASub[2];  // always 0
    r->fRSub = fRSub;
    r->fRadialMeshSub = fRadialMeshSub;
    r->fPolySub = fPolySub;

    return r;
}

void KGComplexAnnulus::Ring::ComputeLocalFrame(double* cen) const
{  //Shifts center of coordinate system towards center of ring.

    for (int i = 0; i < 2; i++) {
        cen[i] = fASub[i];
    }
    cen[2] = 0.;
}

bool KGComplexAnnulus::Ring::ContainsPoint(const double* P) const
{
    // Checks if point <P> is contained by the port.

    // first, we transform to local coordinates
    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P, P_loc, false);

    //We check again if the point is even in the same x,y-plane:
    double epsilon = 1e-7;
    if (fabs(P_loc[2]) > epsilon)
        return false;

    double r_p = sqrt(P_loc[0] * P_loc[0] + P_loc[1] * P_loc[1]);

    // if the point is outside of the radius of the ring, return false.
    if (r_p > fRSub)
        return false;

    // if all of the above conditions are satisfied, return true
    return true;
}

double KGComplexAnnulus::Ring::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    if (!P_in || !P_norm)
        return NAN;

    //Let's first transform to local coordinates:

    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P, P_loc, false);

    //Compute the closest point P_in to Point P as well as the norm vector pointing from point to closest point and returns the distance between them.

    double dist_main = 0;
    double Norm[3] = {0, 0, 1};  //Norm vector of any annulus plane...

    //Compute the distance from point to x,y-Plane that is lifted by ZMain off the Origin. (using Hessian Normal)

    double D = P_loc[2];

    //Compute the projected point on the plane and calculates it's distance towards the edge of the surface.

    double P_Plane[3];
    double PO[3];
    double PO_Norm_Sq = 0;

    for (unsigned int i = 0; i < 3; i++) {
        P_Plane[i] = P_loc[i] - D * Norm[i];
        PO[i] = P_Plane[i];
        PO_Norm_Sq += PO[i] * PO[i];
    }

    double PO_Norm = sqrt(PO_Norm_Sq);

    //Get case when the projection is in the annulus.

    if (PO_Norm < fRSub) {
        double P_in_loc[3];
        double P_in_glob[3];
        for (unsigned int i = 0; i < 3; i++) {
            P_in_loc[i] = P_Plane[i];
        }

        fCoordTransform->ConvertToGlobalCoords(P_in_loc, P_in_glob, false);

        for (unsigned int i = 0; i < 3; i++) {
            P_in[i] = P_in_glob[i];
            if (D > 0) {
                P_norm[i] = -Norm[i];
            }
            else
                P_norm[i] = Norm[i];
        }

        dist_main = fabs(D);
        return dist_main;
    }

    //continuation of normal case.

    for (unsigned int i = 0; i < 3; i++) {
        PO[i] /= PO_Norm;
    }

    double t = fRSub;
    double NormSq = 0;
    double P_in_loc[3];
    double P_norm_loc[3];

    for (unsigned int i = 0; i < 3; i++) {
        P_in_loc[i] = t * PO[i];
        P_norm_loc[i] = P_in_loc[i] - P_loc[i];
        NormSq += P_norm_loc[i] * P_norm_loc[i];
    }

    double Normalize = sqrt(NormSq);

    for (unsigned int i = 0; i < 3; i++) {
        P_norm_loc[i] /= Normalize;
    }

    fCoordTransform->ConvertToGlobalCoords(P_in_loc, P_in, false);
    fCoordTransform->ConvertToGlobalCoords(P_norm_loc, P_norm, true);

    dist_main = Normalize;

    return dist_main;
}
}  // namespace KGeoBag
