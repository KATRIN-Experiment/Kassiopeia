#include "KVMSpaceLineSegment.hh"

#include <limits>

namespace KEMField
{

KVMSpaceLineSegment::KVMSpaceLineSegment()
{
    //set all internal variables to stuff that will produce nonsense if this
    //class is used without initialization

    fP1.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());

    fP2.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());

    fN.SetComponents(std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN());

    fL = 0.0;
}

void KVMSpaceLineSegment::InitializeParameters()
{
    fL = (fP2 - fP1).Magnitude();
    fN = (fP2 - fP1).Unit();
}

void KVMSpaceLineSegment::SetPoint1(const double* point)
{
    fP1.SetComponents(point[0], point[1], point[2]);
}

void KVMSpaceLineSegment::SetPoint2(const double* point)
{
    fP2.SetComponents(point[0], point[1], point[2]);
}


void KVMSpaceLineSegment::SetAll(const double* point1, const double* point2)
{
    SetPoint1(point1);
    SetPoint2(point2);
    InitializeParameters();
}

void KVMSpaceLineSegment::SetVertices(const double* point1, const double* point2)
{
    SetAll(point1, point2);
}

void KVMSpaceLineSegment::GetVertices(double* point1, double* point2) const
{
    for (int i = 0; i < 3; i++) {
        point1[i] = fP1[i];
        point2[i] = fP2[i];
    }
}


}  // namespace KEMField
