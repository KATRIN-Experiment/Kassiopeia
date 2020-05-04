#include "KElectrostaticConstantField.hh"

namespace KEMField
{

KElectrostaticConstantField::KElectrostaticConstantField() : KElectrostaticField(), fField(), fLocation() {}

KElectrostaticConstantField::KElectrostaticConstantField(const KThreeVector& field) :
    KElectrostaticField(),
    fField(field),
    fLocation()
{}

double KElectrostaticConstantField::PotentialCore(const KPosition& aSamplePoint) const
{
    KPosition FieldPoint = aSamplePoint - fLocation;
    return fField.Dot(FieldPoint) + fPotentialOffset;
}

KThreeVector KElectrostaticConstantField::ElectricFieldCore(const KPosition& /*aSamplePoint*/) const
{
    return fField;
}

void KElectrostaticConstantField::SetField(KThreeVector field)
{
    fField = field;
}
KThreeVector KElectrostaticConstantField::GetField() const
{
    return fField;
}

void KElectrostaticConstantField::SetLocation(const KPosition& aLocation)
{
    fLocation = aLocation;
}
KThreeVector KElectrostaticConstantField::GetLocation() const
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
