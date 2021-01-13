#include "KVMSpaceTriangle.hh"

namespace KEMField
{

KVMSpaceTriangle::KVMSpaceTriangle()
{
    fP.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN());

    fN1.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());

    fN2.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());

    fN3.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());
    fL1 = 0.0;
    fL2 = 0.0;
}


void KVMSpaceTriangle::SetCornerPoint(const double* point)
{
    fP.SetComponents(point[0], point[1], point[2]);
}

void KVMSpaceTriangle::SetBasisVectorN1(const double* vec)
{
    fN1.SetComponents(vec[0], vec[1], vec[2]);
    fN1 = fN1.Unit();
}

void KVMSpaceTriangle::SetBasisVectorN2(const double* vec)
{
    fN2.SetComponents(vec[0], vec[1], vec[2]);
    fN2 = fN2.Unit();
}

void KVMSpaceTriangle::SetAll(const double* point, const double* vec1, const double* vec2, double len1, double len2)
{
    SetCornerPoint(point);
    SetBasisVectorN1(vec1);
    SetBasisVectorN2(vec2);
    SetLengthSide1(len1);
    SetLengthSide2(len2);
    InitializeParameters();
}

void KVMSpaceTriangle::SetVertices(const double* point0, const double* point1, const double* point2)
{
    double v1[3];
    double v2[3];
    double l1 = 0;
    double l2 = 0;

    for (unsigned int i = 0; i < 3; i++) {
        v1[i] = point1[i] - point0[i];
        v2[i] = point2[i] - point0[i];
        l1 += v1[i] * v1[i];
        l2 += v2[i] * v2[i];
    }

    l1 = std::sqrt(l1);
    l2 = std::sqrt(l2);
    SetAll(point0, v1, v2, l1, l2);
}

void KVMSpaceTriangle::GetVertices(double* point0, double* point1, double* point2) const
{
    for (int i = 0; i < 3; i++) {
        point0[i] = fP[i];
        point1[i] = fP[i] + fL1 * fN1[i];
        point2[i] = fP[i] + fL2 * fN2[i];
    }
}


void KVMSpaceTriangle::InitializeParameters()
{
    fN3 = fN1.Cross(fN2);
    fSinTheta = fN3.Magnitude();
    fN3 = fN3.Unit();
}

}  // namespace KEMField
