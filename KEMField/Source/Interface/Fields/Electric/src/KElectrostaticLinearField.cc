#include "KElectrostaticLinearField.hh"

#include "KGVolume.hh"

#include <limits>

namespace KEMField
{

KElectrostaticLinearField::KElectrostaticLinearField() :
    KElectrostaticField(),
    fU1(0),
    fU2(0),
    fZ1(0),
    fZ2(0),
    fSurface(nullptr)
{}

double KElectrostaticLinearField::PotentialCore(const KPosition& aSamplePoint) const
{
    // TODO: coordinate transform

    if ((aSamplePoint.Z() < fZ1) || (aSamplePoint.Z()) > fZ2)
        return 0;

    auto E = (fU2 - fU1) / (fZ2 - fZ1);

    return (aSamplePoint.Z() - fZ1) * E + fU1;
}

KThreeVector KElectrostaticLinearField::ElectricFieldCore(const KPosition& aSamplePoint) const
{
    // TODO: coordinate transform

    if ((aSamplePoint.Z() < fZ1) || (aSamplePoint.Z()) > fZ2)
        return KThreeVector::sZero;

    auto E = (fU2 - fU1) / (fZ2 - fZ1);

    return KThreeVector(0, 0, E);
}

void KElectrostaticLinearField::SetPotential1(double aPotential)
{
    fU1 = aPotential;
}
double KElectrostaticLinearField::GetPotential1() const
{
    return fU1;
}

void KElectrostaticLinearField::SetPotential2(double aPotential)
{
    fU2 = aPotential;
}
double KElectrostaticLinearField::GetPotential2() const
{
    return fU2;
}

void KElectrostaticLinearField::SetZ1(double aPosition)
{
    fZ1 = aPosition;
}
double KElectrostaticLinearField::GetZ1() const
{
    return fZ1;
}

void KElectrostaticLinearField::SetZ2(double aPosition)
{
    fZ2 = aPosition;
}
double KElectrostaticLinearField::GetZ2() const
{
    return fZ2;
}


void KElectrostaticLinearField::SetSurface(KGeoBag::KGSurface* aSurface)
{
    fSurface = aSurface;
}
const KGeoBag::KGSurface* KElectrostaticLinearField::GetSurface() const
{
    return fSurface;
}

}  // namespace KEMField
