#include "KGCoordinateTransform.hh"

namespace KGeoBag
{
KGCoordinateTransform::KGCoordinateTransform()
{
    for (int i = 0; i < 3; i++) {
        fP[i] = 0;
        fX[i] = 0;
        fY[i] = 0;
        fZ[i] = 0;
    }
    fX[0] = fY[1] = fZ[2] = 1.;
}

KGCoordinateTransform::KGCoordinateTransform(const double* p, const double* x, const double* y, const double* z)
{
    for (int i = 0; i < 3; i++) {
        fP[i] = p[i];
        fX[i] = x[i];
        fY[i] = y[i];
        fZ[i] = z[i];
    }
}

void KGCoordinateTransform::ConvertToLocalCoords(const double* global, double* local, const bool isVec) const
{
    /**
     * Transforms global x,y,z using the coordinates of the group into local
     * x,y,z.  If the conversion is for a vector, no translation occurs.
     */

    double pp[3] = {0, 0, 0};

    for (int i = 0; i < 3; i++) {
        local[i] = 0;
        if (!isVec)
            pp[i] = fP[i];
    }

    for (int i = 0; i < 3; i++) {
        local[0] += fX[i] * (global[i] - pp[i]);
        local[1] += fY[i] * (global[i] - pp[i]);
        local[2] += fZ[i] * (global[i] - pp[i]);
    }
}

void KGCoordinateTransform::ConvertToGlobalCoords(const double* local, double* global, const bool isVec) const
{
    /**
     * Transforms local x,y,z using the coordinates of the group into global
     * x,y,z.  If the conversion is for a vector, no translation occurs.
     */

    double pp[3] = {0, 0, 0};

    if (!isVec) {
        for (int i = 0; i < 3; i++) {
            pp[0] -= fX[i] * fP[i];
            pp[1] -= fY[i] * fP[i];
            pp[2] -= fZ[i] * fP[i];
        }
    }

    for (int i = 0; i < 3; i++) {
        global[i] = (fX[i] * (local[0] - pp[0]) + fY[i] * (local[1] - pp[1]) + fZ[i] * (local[2] - pp[2]));
    }
}

void KGCoordinateTransform::ConvertToLocalCoords(const KThreeVector& global, KThreeVector& local,
                                                 const bool isVec) const
{
    /**
     * Transforms global x,y,z using the coordinates of the group into local
     * x,y,z.  If the conversion is for a vector, no translation occurs.
     */

    double pp[3] = {0, 0, 0};

    for (int i = 0; i < 3; i++) {
        local[i] = 0;
        if (!isVec)
            pp[i] = fP[i];
    }

    for (int i = 0; i < 3; i++) {
        local[0] += fX[i] * (global[i] - pp[i]);
        local[1] += fY[i] * (global[i] - pp[i]);
        local[2] += fZ[i] * (global[i] - pp[i]);
    }
}

void KGCoordinateTransform::ConvertToGlobalCoords(const KThreeVector& local, KThreeVector& global,
                                                  const bool isVec) const
{
    /**
     * Transforms local x,y,z using the coordinates of the group into global
     * x,y,z.  If the conversion is for a vector, no translation occurs.
     */

    double pp[3] = {0, 0, 0};

    if (!isVec) {
        for (int i = 0; i < 3; i++) {
            pp[0] -= fX[i] * fP[i];
            pp[1] -= fY[i] * fP[i];
            pp[2] -= fZ[i] * fP[i];
        }
    }

    for (int i = 0; i < 3; i++) {
        global[i] = (fX[i] * (local[0] - pp[0]) + fY[i] * (local[1] - pp[1]) + fZ[i] * (local[2] - pp[2]));
    }
}
}  // namespace KGeoBag
