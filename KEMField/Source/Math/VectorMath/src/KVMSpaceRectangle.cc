#include "KVMSpaceRectangle.hh"

namespace KEMField
{

KVMSpaceRectangle::KVMSpaceRectangle()
{
    //set all internal variables to stuff that will produce nonsense if this
    //class is used without initialization

    fP.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN());

    fN1.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());

    fN2.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());

    fL1 = 0.0;
    fL2 = 0.0;
}

void KVMSpaceRectangle::SetCornerPoint(const double* point)
{
    fP.SetComponents(point[0], point[1], point[2]);
}

void KVMSpaceRectangle::SetBasisVectorN1(const double* vec)
{
    fN1.SetComponents(vec[0], vec[1], vec[2]);
    fN1 = fN1.Unit();
}

void KVMSpaceRectangle::SetBasisVectorN2(const double* vec)
{
    fN2.SetComponents(vec[0], vec[1], vec[2]);
    fN2 = fN2.Unit();
}

void KVMSpaceRectangle::SetAll(const double* point, const double* vec1, const double* vec2, double len1, double len2)
{
    SetCornerPoint(point);
    SetBasisVectorN1(vec1);
    SetBasisVectorN2(vec2);
    SetLengthSide1(len1);
    SetLengthSide2(len2);
    InitializeParameters();
}

void KVMSpaceRectangle::SetVertices(const double* point0, const double* point1, const double* point2,
                                    const double* point3)
{
    //we have to split the rectangle into two triangles
    //so here we figure out which sets of points we need to

    double v1[3];
    double v2[3];
    double v3[3];
    double l1 = 0;
    double l2 = 0;
    double l3 = 0;

    for (unsigned int i = 0; i < 3; i++) {
        v1[i] = point1[i] - point0[i];
        v2[i] = point2[i] - point0[i];
        v3[i] = point3[i] - point0[i];

        l1 += v1[i] * v1[i];
        l2 += v2[i] * v2[i];
        l3 += v3[i] * v3[i];
    }

    l1 = std::sqrt(l1);
    l2 = std::sqrt(l2);
    l3 = std::sqrt(l3);

    int opposite_corner_index;
    double max_dist = l1;
    opposite_corner_index = 1;
    if (l2 > max_dist) {
        max_dist = l2;
        opposite_corner_index = 2;
    }
    if (l3 > max_dist) {
        max_dist = l3;
        opposite_corner_index = 3;
    }

    switch (opposite_corner_index) {
        case 1:
            SetAll(point0, v2, v3, l2, l3);
            break;
        case 2:
            SetAll(point0, v3, v1, l3, l1);
            break;
        case 3:
            SetAll(point0, v1, v2, l1, l2);
            break;
    }
}

void KVMSpaceRectangle::GetVertices(double* point0, double* point1, double* point2, double* point3) const
{
    for (unsigned int i = 0; i < 3; i++) {
        point0[i] = fP[i];
        point1[i] = fP[i] + fL1 * fN1[i];
        point2[i] = fP[i] + fL2 * fN2[i];
        point3[i] = fP[i] + fL1 * fN1[i] + fL2 * fN2[i];
    }
}


}  // namespace KEMField
