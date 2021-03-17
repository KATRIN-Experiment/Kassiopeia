#include "KGPortHousing.hh"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace KGeoBag
{
KGPortHousing::KGPortHousing(const double Amain[3], const double Bmain[3], double rmain) : fCoordTransform(nullptr)
{
    fRMain = rmain;
    for (int i = 0; i < 3; i++) {
        fAMain[i] = Amain[i];
        fBMain[i] = Bmain[i];
    }
}

KGPortHousing::~KGPortHousing()
{
    delete fCoordTransform;

    for (auto& port : fPorts) {
        delete port;
    }
}

void KGPortHousing::AddCircularPort(double asub[3], double rsub)
{
    // Adds a circular port to the housing.

    auto* p = new KGPortHousing::CircularPort(this, asub, rsub);
    fPorts.push_back(p);
}

void KGPortHousing::AddRectangularPort(double asub[3], double length, double width)
{
    // Adds a rectangular port housing.

    auto* p = new KGPortHousing::RectangularPort(this, asub, length, width);
    fPorts.push_back(p);
}

void KGPortHousing::AddPort(KGPortHousing::Port* port)
{
    // Adds a port housing.

    port->SetPortHousing(this);

    fPorts.push_back(port);
}

KGPortHousing* KGPortHousing::Clone() const
{
    auto* p = new KGPortHousing();
    for (unsigned int i = 0; i < 3; i++) {
        p->fAMain[i] = fAMain[i];
        p->fBMain[i] = fBMain[i];
        p->fD[i] = fD[i];
        p->fNorm[i] = fNorm[i];
    }
    p->fRMain = fRMain;
    p->fLength = fLength;
    p->fLengthSq = fLengthSq;
    p->fRSq = fRSq;
    p->fNumDiscMain = fNumDiscMain;
    p->fPolyMain = fPolyMain;

    p->fCoordTransform = new KGCoordinateTransform(*fCoordTransform);

    for (auto* port : fPorts)
        p->fPorts.push_back(port->Clone(p));
    return p;
}

void KGPortHousing::Initialize() const
{
    fLengthSq = 0.;

    for (int i = 0; i < 3; i++) {
        fD[i] = fNorm[i] = (fBMain[i] - fAMain[i]);
        fLengthSq += fD[i] * fD[i];
    }

    fLength = sqrt(fLengthSq);
    for (double& i : fNorm)
        i /= fLength;

    fRSq = (fRMain * fRMain);

    fNumDiscMain = 30;
    fPolyMain = 120;

    fCoordTransform = new KGCoordinateTransform();
    for (auto* port : fPorts)
        port->Initialize();
}

bool KGPortHousing::ContainsPoint(const double* P) const
{
    double pd[3] = {P[0] - fAMain[0], P[1] - fAMain[1], P[2] - fAMain[2]};

    double dot = fD[0] * pd[0] + fD[1] * pd[1] + fD[2] * pd[2];

    // if the point is beyond the cylinder in the direction of its axis, then the
    // point is definitely not contained
    if (dot < 0. || dot > fLengthSq)
        return false;

    double dsq = (pd[0] * pd[0] + pd[1] * pd[1] + pd[2] * pd[2]) - dot * dot / fLengthSq;

    if (dsq <= fRSq) {
        // if the point is within the port housing cylinder, it is definitely
        // contained
        return true;
    }
    else {
        // otherwise, we have to look in each of the ports
        for (auto port : fPorts) {
            if (port->ContainsPoint(P))
                return true;
        }
    }

    return false;
}

double KGPortHousing::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    if (P_norm && !P_in) {
        double p_in[3];
        return DistanceTo(P, p_in, P_norm);
    }

    double pd[3] = {P[0] - fAMain[0], P[1] - fAMain[1], P[2] - fAMain[2]};

    double dot = fNorm[0] * pd[0] + fNorm[1] * pd[1] + fNorm[2] * pd[2];

    double dist_par = 0;

    int below_in_above = 1;

    if (dot > fLength) {
        below_in_above = 2;
        dist_par = dot - fLength;
    }
    else if (dot < 0.) {
        below_in_above = 0;
        dist_par = fabs(dot);
    }

    for (unsigned int i = 0; i < 3; i++)
        pd[i] -= fNorm[i] * dot;

    double norm2[3];
    double len_perp = 0;
    for (unsigned int i = 0; i < 3; i++) {
        norm2[i] = pd[i];
        len_perp += pd[i] * pd[i];
    }

    len_perp = sqrt(len_perp);

    if (len_perp > 0.)
        for (double& i : norm2)
            i /= len_perp;
    else {
        for (unsigned int i = 0; i < 3; i++)
            norm2[i] = (i == 1 ? 1. : -1.) * fNorm[(i + 1) % 3];
    }

    double dist_perp = fabs(len_perp - fRMain);

    double dist_main = sqrt(dist_par * dist_par + dist_perp * dist_perp);
    double P_main[3];

    if (below_in_above == 0)
        for (unsigned int i = 0; i < 3; i++)
            P_main[i] = norm2[i] * fRMain + fAMain[i];
    else if (below_in_above == 2)
        for (unsigned int i = 0; i < 3; i++)
            P_main[i] = norm2[i] * fRMain + fBMain[i];
    else
        for (unsigned int i = 0; i < 3; i++)
            P_main[i] = norm2[i] * fRMain + fAMain[i] + fNorm[i] * dot;

    bool pointIsOnMainCylinder = true;

    if (P_norm) {
        double r = sqrt(pd[0] * pd[0] + pd[1] * pd[1]);
        double sign = (r > fRMain ? 1. : -1.);
        for (unsigned int i = 0; i < 3; i++)
            P_norm[i] = sign * norm2[i];
    }

    for (auto* port : fPorts) {
        double P_tmp[3];
        double P_tmp_norm[3];
        double dist_tmp = port->DistanceTo(P, P_tmp, P_tmp_norm);

        if (dist_tmp < dist_main || (port->ContainsPoint(P_main) && pointIsOnMainCylinder)) {
            pointIsOnMainCylinder = false;

            if (P_in)
                for (unsigned int j = 0; j < 3; j++)
                    P_in[j] = P_tmp[j];

            if (P_norm)
                for (unsigned int j = 0; j < 3; j++)
                    P_norm[j] = P_tmp_norm[j];

            dist_main = dist_tmp;
        }
    }
    return dist_main;
}

KGPortHousing::RectangularPort::RectangularPort(KGPortHousing* portHousing, const double asub[3], double length,
                                                double width) :
    Port(portHousing)
{
    for (int i = 0; i < 3; i++)
        fASub[i] = asub[i];

    fLength = length;
    fWidth = width;

    Initialize();
}

KGPortHousing::RectangularPort::~RectangularPort()
{

    delete fCoordTransform;
}

void KGPortHousing::RectangularPort::Initialize()
{
    fXDisc = 8;

    fLengthDisc = 10;
    fWidthDisc = 10;

    fNumDiscSub = 20;

    fBoxLength = 0;
    fBoxWidth = 0;

    // first, we find the intersection of the axes of the rectangular prism and
    // the main cylinder

    ComputeLocalFrame(fCen, fX_loc, fY_loc, fZ_loc);

    fCoordTransform = new KGCoordinateTransform(fCen, fX_loc, fY_loc, fZ_loc);
    // the length along the port to discretize in compensation for the
    // asymmetries associated with the intersection is the length of the port
    // from the edge of the main cylinder
    fPortLength = sqrt((fCen[0] - fASub[0]) * (fCen[0] - fASub[0]) + (fCen[1] - fASub[1]) * (fCen[1] - fASub[1]) +
                       (fCen[2] - fASub[2]) * (fCen[2] - fASub[2]));
}

KGPortHousing::RectangularPort* KGPortHousing::RectangularPort::Clone(KGPortHousing* p) const
{
    auto* r = new RectangularPort();

    r->fPortHousing = p;

    for (unsigned int i = 0; i < 3; i++) {
        r->fCen[i] = fCen[i];
        r->fX_loc[i] = fX_loc[i];
        r->fY_loc[i] = fY_loc[i];
        r->fZ_loc[i] = fZ_loc[i];
        r->fNorm[i] = fNorm[i];
    }
    r->fCoordTransform = new KGCoordinateTransform(*fCoordTransform);

    r->fASub[0] = fASub[0];
    r->fASub[1] = fASub[1];
    r->fASub[2] = fASub[2];
    r->fLength = fLength;
    r->fWidth = fWidth;
    r->fXDisc = fXDisc;
    r->fNumDiscSub = fNumDiscSub;
    r->fLengthDisc = fLengthDisc;
    r->fWidthDisc = fWidthDisc;
    r->fBoxLength = fBoxLength;
    r->fBoxWidth = fBoxWidth;
    r->fPortLength = fPortLength;

    return r;
}

void KGPortHousing::RectangularPort::ComputeLocalFrame(double* cen, double* x, double* y, double* z) const
{
    // This function computes the local center (0,0,0), x (1,0,0), y (0,1,0) and
    // z (0,0,0) for the port valve

    double u = 0;
    double len2 = 0;

    double aMain[3];
    double bMain[3];
    for (int i = 0; i < 3; i++) {
        aMain[i] = fPortHousing->GetAMain(i);
        bMain[i] = fPortHousing->GetBMain(i);
    }

    for (int i = 0; i < 3; i++) {
        u += (fASub[i] - aMain[i]) * (bMain[i] - aMain[i]);
        len2 += (bMain[i] - aMain[i]) * (bMain[i] - aMain[i]);
    }
    u /= len2;

    for (int i = 0; i < 3; i++) {
        cen[i] = aMain[i] + u * (bMain[i] - aMain[i]);
        x[i] = bMain[i] - aMain[i];
        z[i] = fASub[i] - cen[i];
    }

    double tmp = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    for (int i = 0; i < 3; i++)
        x[i] /= tmp;
    tmp = sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
    for (int i = 0; i < 3; i++)
        z[i] /= tmp;
    for (int i = 0; i < 3; i++)
        y[i] = z[(i + 1) % 3] * x[(i + 2) % 3] - z[(i + 2) % 3] * x[(i + 1) % 3];
}

bool KGPortHousing::RectangularPort::ContainsPoint(const double* P) const
{
    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P, P_loc, false);

    if (P_loc[2] < 0 || P_loc[2] > fPortLength)
        return false;

    if (fabs(P_loc[0]) > fWidth / 2)
        return false;

    if (fabs(P_loc[1]) > fLength / 2)
        return false;

    double r_p = sqrt(P_loc[1] * P_loc[1] + P_loc[2] * P_loc[2]) + 1.e-8;

    if (r_p < fPortHousing->GetRMain())
        return false;

    return true;
}

double KGPortHousing::RectangularPort::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P, P_loc, false);

    double rMain = fPortHousing->GetRMain();

    double h = P_loc[2] - rMain;

    double P_tmp[3];
    double P_n[3];

    if (h < 0.) {
        // if here, we may have to deal with the intersection...

        if (P_norm) {
            P_n[0] = P_n[1] = 0.;
            P_n[2] = -1.;
            fCoordTransform->ConvertToGlobalCoords(P_n, P_norm, true);
        }

        double theta_geom = asin(fLength / (2. * rMain));
        double theta = 0.;
        if (P_loc[2] > 1.e-14)
            theta = atan(P_loc[1] / P_loc[2]);
        else if (P_loc[2] * P_loc[2] > 0.)
            theta = M_PI / 2.;
        else
            theta = -M_PI / 2.;

        if (theta > M_PI / 2.)
            theta -= 2. * M_PI;

        if (fabs(theta) > theta_geom)
            theta = theta / fabs(theta) * theta_geom;

        if (fabs(fabs(P_loc[1]) - fLength / 2.) > fabs(fabs(P_loc[0]) - fWidth / 2.)) {
            // we cast to the curved edges of the intersecion

            P_tmp[0] = (P_loc[0] < 0.) ? -fWidth / 2. : fWidth / 2.;
            P_tmp[1] = P_loc[1];
            if (P_tmp[1] < -fLength / 2.)
                P_tmp[1] = -fLength / 2.;
            else if (P_tmp[1] > fLength / 2.)
                P_tmp[1] = fLength / 2.;
            P_tmp[2] = P_loc[2];
            if (P_tmp[2] < rMain * cos(theta))
                P_tmp[2] = rMain * cos(theta);

            if (P_in)
                fCoordTransform->ConvertToGlobalCoords(P_tmp, P_in, false);

            return sqrt((P_tmp[0] - P_loc[0]) * (P_tmp[0] - P_loc[0]) + (P_tmp[1] - P_loc[1]) * (P_tmp[1] - P_loc[1]) +
                        (P_tmp[2] - P_loc[2]) * (P_tmp[2] - P_loc[2]));
        }
        else {
            // we cast to the straight edges of the intersecion

            P_tmp[0] = P_loc[0];
            if (P_tmp[0] < -fWidth / 2.)
                P_tmp[0] = -fWidth / 2.;
            else if (P_tmp[0] > fWidth / 2.)
                P_tmp[0] = fWidth / 2.;
            P_tmp[1] = (P_loc[1] < 0.) ? -fLength / 2. : fLength / 2.;
            P_tmp[2] = P_loc[2];
            if (P_tmp[2] < rMain * cos(theta))
                P_tmp[2] = rMain * cos(theta);

            if (P_in)
                fCoordTransform->ConvertToGlobalCoords(P_tmp, P_in, false);

            return sqrt((P_tmp[0] - P_loc[0]) * (P_tmp[0] - P_loc[0]) + (P_tmp[1] - P_loc[1]) * (P_tmp[1] - P_loc[1]) +
                        (P_tmp[2] - P_loc[2]) * (P_tmp[2] - P_loc[2]));
        }
    }
    else {
        if (fabs(P_loc[0]) < fWidth / 2. && fabs(P_loc[1]) < fLength / 2.) {
            // we're inside the port
            if (fabs(fabs(P_loc[0]) - fWidth / 2.) > fabs(fabs(P_loc[1]) - fLength / 2.)) {
                P_n[0] = P_n[2] = 0.;
                P_n[1] = (P_loc[1] < 0.) ? 1. : -1.;
                P_tmp[0] = P_loc[0];
                if (P_loc[0] > fWidth / 2.)
                    P_tmp[0] = fWidth / 2.;
                else if (P_loc[0] < -fWidth / 2.)
                    P_tmp[0] = -fWidth / 2.;
                P_tmp[1] = (P_loc[1] < 0.) ? -fLength / 2. : fLength / 2.;
            }
            else {
                P_n[1] = P_n[2] = 0.;
                P_n[0] = (P_loc[0] < 0.) ? 1. : -1.;
                P_tmp[0] = (P_loc[0] < 0.) ? -fWidth / 2. : fWidth / 2.;
                P_tmp[1] = P_loc[1];
                if (P_loc[1] > fLength / 2.)
                    P_tmp[1] = fLength / 2.;
                else if (P_loc[1] < -fLength / 2.)
                    P_tmp[1] = -fLength / 2.;
            }
        }
        else {
            P_n[0] = P_n[1] = P_n[2] = 0.;

            P_tmp[0] = P_loc[0];
            if (P_loc[0] > fWidth / 2.) {
                P_n[0] = 1.;
                P_tmp[0] = fWidth / 2.;
            }
            else if (P_loc[0] < -fWidth / 2.) {
                P_n[0] = -1.;
                P_tmp[0] = -fWidth / 2.;
            }
            P_tmp[1] = P_loc[1];
            if (P_loc[1] > fLength / 2.) {
                if (fabs(P_n[0]) < .5)
                    P_n[1] = 1.;
                P_tmp[1] = fLength / 2.;
            }
            else if (P_loc[1] < -fLength / 2.) {
                if (fabs(P_n[0]) < .5)
                    P_n[1] = -1.;
                P_tmp[1] = -fLength / 2.;
            }
        }
        P_tmp[2] = (P_loc[2] > fPortLength) ? fPortLength : P_loc[2];

        if (P_in)
            fCoordTransform->ConvertToGlobalCoords(P_tmp, P_in, false);

        if (P_norm)
            fCoordTransform->ConvertToGlobalCoords(P_n, P_norm, true);

        return sqrt((P_tmp[0] - P_loc[0]) * (P_tmp[0] - P_loc[0]) + (P_tmp[1] - P_loc[1]) * (P_tmp[1] - P_loc[1]) +
                    (P_tmp[2] - P_loc[2]) * (P_tmp[2] - P_loc[2]));
    }
}

KGPortHousing::CircularPort::CircularPort(KGPortHousing* portHousing, const double asub[3], double rsub) :
    Port(portHousing)
{
    for (int i = 0; i < 3; i++)
        fASub[i] = asub[i];

    fRSub = rsub;

    Initialize();
}

KGPortHousing::CircularPort::~CircularPort()
{

    delete fCoordTransform;
}

void KGPortHousing::CircularPort::Initialize()
{
    fNumDiscSub = 6;
    fXDisc = 5;
    fBoxLength = 0;

    // first, we find the intersection of the axes of the two cylinders
    ComputeLocalFrame(fCen, fX_loc, fY_loc, fZ_loc);

    fCoordTransform = new KGCoordinateTransform(fCen, fX_loc, fY_loc, fZ_loc);

    // for later calculations, we compute some parameters of the port here
    double aSub_loc[3];
    fCoordTransform->ConvertToLocalCoords(fASub, aSub_loc, false);

    fLengthSq = 0;

    for (int i = 0; i < 3; i++) {
        fNorm[i] = fASub[i] - fCen[i];
        fLengthSq += fNorm[i] * fNorm[i];
    }

    fLength = sqrt(fLengthSq);
    for (double& i : fNorm)
        i /= fLength;
}

KGPortHousing::CircularPort* KGPortHousing::CircularPort::Clone(KGPortHousing* p) const
{
    auto* c = new CircularPort();

    c->fPortHousing = p;

    for (unsigned int i = 0; i < 3; i++) {
        c->fCen[i] = fCen[i];
        c->fX_loc[i] = fX_loc[i];
        c->fY_loc[i] = fY_loc[i];
        c->fZ_loc[i] = fZ_loc[i];
        c->KGPortHousing::Port::fNorm[i] = KGPortHousing::Port::fNorm[i];
        c->fNorm[i] = fNorm[i];
    }
    c->fCoordTransform = new KGCoordinateTransform(*fCoordTransform);

    c->fASub[0] = fASub[0];
    c->fASub[1] = fASub[1];
    c->fASub[2] = fASub[2];
    c->fRSub = fRSub;
    c->fLength = fLength;
    c->fLengthSq = fLengthSq;
    c->fXDisc = fXDisc;
    c->fNumDiscSub = fNumDiscSub;
    c->fPolySub = fPolySub;
    c->fBoxLength = fBoxLength;

    return c;
}

void KGPortHousing::CircularPort::ComputeLocalFrame(double* cen, double* x, double* y, double* z) const
{
    // This function computes the local center (0,0,0), x (1,0,0), y (0,1,0) and
    // z (0,0,1) for the port valve

    double u = 0;
    double len2 = 0;

    double aMain[3];
    double bMain[3];
    for (int i = 0; i < 3; i++) {
        aMain[i] = fPortHousing->GetAMain(i);
        bMain[i] = fPortHousing->GetBMain(i);
    }

    for (int i = 0; i < 3; i++) {
        u += (fASub[i] - aMain[i]) * (bMain[i] - aMain[i]);
        len2 += (bMain[i] - aMain[i]) * (bMain[i] - aMain[i]);
    }
    u /= len2;

    for (int i = 0; i < 3; i++) {
        cen[i] = aMain[i] + u * (bMain[i] - aMain[i]);
        x[i] = bMain[i] - aMain[i];
        z[i] = fASub[i] - cen[i];
    }

    double tmp = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    for (int i = 0; i < 3; i++)
        x[i] /= tmp;
    tmp = sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
    for (int i = 0; i < 3; i++)
        z[i] /= tmp;
    for (int i = 0; i < 3; i++)
        y[i] = z[(i + 1) % 3] * x[(i + 2) % 3] - z[(i + 2) % 3] * x[(i + 1) % 3];
}

bool KGPortHousing::CircularPort::ContainsPoint(const double* P) const
{
    // Checks if point <P> is contained by the port.

    // first, we transform to local coordinates
    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P, P_loc, false);

    // if the transformed point is lower than the intersection of the axis of
    // the two cylinders, or if it is higher than the length of the subordinate
    // cylinder, return false
    if (P_loc[2] < 0 || P_loc[2] > fLength)
        return false;

    double r_p = sqrt(P_loc[0] * P_loc[0] + P_loc[1] * P_loc[1]);

    // if the point is outside of the radius of the subordinate cylinder, return
    // false
    if (r_p > fRSub)
        return false;

    r_p = sqrt(P_loc[1] * P_loc[1] + P_loc[2] * P_loc[2]) + 1.e-8;

    // if the point is within the main cylinder, return false
    if (r_p < fPortHousing->GetRMain())
        return false;

    // if all of the above conditions are satisfied, return true
    return true;
}

//______________________________________________________________________________

double KGPortHousing::CircularPort::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    // Doing this right requires a numerical minimizer to solve elliptic
    // functions and is probably overkill for the task.  The following is
    // approximately correct in the limit that r_sub is much smaller than
    // r_main, and gives the same gross features, I think.
    //
    // Ansatz: 1. Convert to local coordinates
    //         2. Check if the point is in a region of the cylinder that's easy
    //            to compute (above the sheath of the main cylinder).  If it is,
    //            compute the distance and P_in, and exit.
    //         3. If the point is below the sheath ofthe main cylinder, compute
    //            theta, defined as the angle about the subordinate cylinder
    //            axis (theta = 0 points in the local x-direction).
    //         4. Compute the height of the saddle at the given theta.  If the
    //            height is smaller than the height of the local P, then return
    //            the distance and P_in using the same algorithm as in step 2.
    //         5. If the height of the saddle at theta is larger than the height
    //            of the local P (the x-coordinate), then set P_in to be the
    //            saddle point at theta (this is the approximation) and compute
    //            and return the distance between these two points.

    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P, P_loc, false);

    double rMain = fPortHousing->GetRMain();

    double h = P_loc[2] - rMain;

    double P_tmp[3];
    double P_n[3];

    if (h < 0.) {
        // if here, we may have to deal with the saddle...

        if (P_norm) {
            P_norm[0] = fNorm[0];
            P_norm[1] = fNorm[1];
            P_norm[2] = fNorm[2];
        }

        double theta = 0.;
        if (fabs(P_loc[0]) > 1.e-14)
            theta = atan(fabs(P_loc[1] / P_loc[0]));
        else
            theta = M_PI / 2.;

        if (P_loc[1] > 0. && P_loc[0] < -0.)
            theta = M_PI - theta;
        else if (P_loc[1] < -0. && P_loc[0] < -0.)
            theta += M_PI;
        else if (P_loc[1] < -0. && P_loc[0] > 0.)
            theta = 2. * M_PI - theta;

        double sine2 = sin(theta);
        sine2 *= sine2;

        if (P_loc[2] < sqrt(rMain * rMain - fRSub * fRSub * sine2)) {
            // we need to deal with the saddle!
            P_tmp[0] = fRSub * cos(theta);
            P_tmp[1] = fRSub * sin(theta);
            P_tmp[2] = sqrt(rMain * rMain - fRSub * fRSub * sine2);

            if (P_in)
                fCoordTransform->ConvertToGlobalCoords(P_tmp, P_in, false);

            return sqrt((P_tmp[0] - P_loc[0]) * (P_tmp[0] - P_loc[0]) + (P_tmp[1] - P_loc[1]) * (P_tmp[1] - P_loc[1]) +
                        (P_tmp[2] - P_loc[2]) * (P_tmp[2] - P_loc[2]));
        }
    }

    P_tmp[2] = (h > (fLength - rMain)) ? fLength : P_loc[2];
    P_n[2] = 0.;

    double r_loc = sqrt(P_loc[0] * P_loc[0] + P_loc[1] * P_loc[1]);
    if (r_loc > 1.e-14) {
        P_tmp[0] = P_loc[0] * fRSub / r_loc;
        P_tmp[1] = P_loc[1] * fRSub / r_loc;
        P_n[0] = P_loc[0] / r_loc;
        P_n[1] = P_loc[1] / r_loc;
    }
    else {
        P_tmp[0] = 0.;
        P_tmp[1] = fRSub;
        P_n[0] = 0.;
        P_n[1] = 1.;
    }

    if (P_norm) {
        if (r_loc < fRSub)
            for (double& i : P_n)
                i *= -1.;
        fCoordTransform->ConvertToGlobalCoords(P_n, P_norm, true);
    }

    if (P_in)
        fCoordTransform->ConvertToGlobalCoords(P_tmp, P_in, false);

    return sqrt((P_tmp[0] - P_loc[0]) * (P_tmp[0] - P_loc[0]) + (P_tmp[1] - P_loc[1]) * (P_tmp[1] - P_loc[1]) +
                (P_tmp[2] - P_loc[2]) * (P_tmp[2] - P_loc[2]));
}
}  // namespace KGeoBag
