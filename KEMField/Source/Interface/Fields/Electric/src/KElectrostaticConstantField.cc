#include "KElectrostaticConstantField.hh"

namespace KEMField
{

KElectrostaticConstantField::KElectrostaticConstantField() :
  fFieldVector(katrin::KThreeVector::sInvalid),
  fLocation(katrin::KThreeVector::sZero),
  fPotentialOffset(0)
{};

KElectrostaticConstantField::KElectrostaticConstantField(const KFieldVector& field) :
  fFieldVector(field),
  fLocation(katrin::KThreeVector::sZero),
  fPotentialOffset(0)
{}

double KElectrostaticConstantField::PotentialCore(const KPosition& aSamplePoint) const
{
    KPosition FieldPoint = aSamplePoint - fLocation;
    return fFieldVector.Dot(FieldPoint) + fPotentialOffset;
}

KFieldVector KElectrostaticConstantField::ElectricFieldCore(const KPosition& /*aSamplePoint*/) const
{
    return fFieldVector;
}

void KElectrostaticConstantField::SetField(const KFieldVector& field)
{
    fFieldVector = field;
}
KFieldVector KElectrostaticConstantField::GetField() const
{
    return fFieldVector;
}

void KElectrostaticConstantField::SetLocation(const KPosition& aLocation)
{
    fLocation = aLocation;
}
KFieldVector KElectrostaticConstantField::GetLocation() const
{
    return fLocation;
}

void KElectrostaticConstantField::SetPotentialOffset(const double& aPotential)
{
    fPotentialOffset = aPotential;
}
const double& KElectrostaticConstantField::GetPotentialOffset() const
{
    return fPotentialOffset;
}

}  // namespace KEMField
