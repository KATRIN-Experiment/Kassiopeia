#include "KAxis.hh"

#include <cmath>

namespace katrin
{

KAxis::KAxis() : fCenter(KThreeVector::sZero), fDirection(KThreeVector::sZUnit) {}
KAxis::KAxis(const KAxis&) = default;
KAxis::~KAxis() = default;

bool KAxis::EqualTo(const KAxis& anAxis) const
{
    if (fabs(fDirection.Dot(anAxis.fDirection)) < (1. - 1.e-5)) {
        return false;
    }

    if ((fCenter - anAxis.fCenter).Magnitude() > 1.e-5) {
        return false;
    }

    return true;
}
bool KAxis::ParallelTo(const KThreeVector& aDirection) const
{
    if (fabs(fDirection.Dot(aDirection)) < (1. - 1.e-5)) {
        return false;
    }

    return true;
}

void KAxis::SetPoints(const KThreeVector& aPointOne, const KThreeVector& aPointTwo)
{
    if (aPointOne.Z() > aPointTwo.Z()) {
        fDirection = (aPointOne - aPointTwo).Unit();
    }
    else if (aPointOne.Z() < aPointTwo.Z()) {
        fDirection = (aPointTwo - aPointOne).Unit();
    }
    else {
        if (aPointOne.Y() > aPointTwo.Y()) {
            fDirection = (aPointOne - aPointTwo).Unit();
        }
        else if (aPointOne.Y() < aPointTwo.Y()) {
            fDirection = (aPointTwo - aPointOne).Unit();
        }
        else {
            if (aPointOne.X() > aPointTwo.X()) {
                fDirection = (aPointOne - aPointTwo).Unit();
            }
            else if (aPointOne.X() < aPointTwo.X()) {
                fDirection = (aPointTwo - aPointOne).Unit();
            }
            else {
                fDirection.SetComponents(0., 0., 1.);
            }
        }
    }

    fCenter = aPointOne - aPointOne.Dot(fDirection) * fDirection;

    return;
}
const KThreeVector& KAxis::GetCenter() const
{
    return fCenter;
}
const KThreeVector& KAxis::GetDirection() const
{
    return fDirection;
}

}  // namespace katrin
