#include "KElectrostaticConstantField.hh"

namespace KEMField
{

KElectrostaticConstantField::KElectrostaticConstantField() = default;

KElectrostaticConstantField::KElectrostaticConstantField(const KFieldVector& field) : fField(field) {}

double KElectrostaticConstantField::PotentialCore(const KPosition& aSamplePoint) const
{
    KPosition FieldPoint = aSamplePoint - fLocation;
    return fField.Dot(FieldPoint) + fPotentialOffset;
}

KFieldVector KElectrostaticConstantField::ElectricFieldCore(const KPosition& /*aSamplePoint*/) const
{
    return fField;
}

void KElectrostaticConstantField::SetField(const KFieldVector& field)
{
    fField = field;
}
KFieldVector KElectrostaticConstantField::GetField() const
{
    return fField;
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
