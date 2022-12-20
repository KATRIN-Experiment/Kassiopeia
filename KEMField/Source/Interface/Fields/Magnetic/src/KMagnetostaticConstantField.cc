#include "KMagnetostaticConstantField.hh"

namespace KEMField
{

KMagnetostaticConstantField::KMagnetostaticConstantField() :
    fFieldVector(katrin::KThreeVector::sInvalid),
    fLocation(katrin::KThreeVector::sZero)
{}

KMagnetostaticConstantField::KMagnetostaticConstantField(const KFieldVector& aField) : fFieldVector(aField) {}

/** We choose A(r) = 1/2 * B x r as the magnetic potential.
 * This is a viable choice for Coulomb gauge.*/
KFieldVector KMagnetostaticConstantField::MagneticPotentialCore(const KPosition& aSamplePoint) const
{
    KPosition FieldPoint = aSamplePoint - fLocation;
    return 0.5 * fFieldVector.Cross(FieldPoint);
}
KFieldVector KMagnetostaticConstantField::MagneticFieldCore(const KPosition& /*aSamplePoint*/) const
{
    return fFieldVector;
}
KGradient KMagnetostaticConstantField::MagneticGradientCore(const KPosition& /*aSamplePoint*/) const
{
    return katrin::KThreeMatrix::sZero;
}

void KMagnetostaticConstantField::SetField(const KFieldVector& aFieldVector)
{
    fFieldVector = aFieldVector;
}
KFieldVector KMagnetostaticConstantField::GetField() const
{
    return fFieldVector;
}

void KMagnetostaticConstantField::SetLocation(const KPosition& aLocation)
{
    fLocation = aLocation;
}
KFieldVector KMagnetostaticConstantField::GetLocation() const
{
    return fLocation;
}

}  // namespace KEMField
