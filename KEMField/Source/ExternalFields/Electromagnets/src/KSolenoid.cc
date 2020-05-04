#include "KSolenoid.hh"

#include "KElectromagnetVisitor.hh"

namespace KEMField
{
void KSolenoid::SetValues(const KPosition& p0, const KPosition& p1, double current)
{
    fP0 = p0;
    fP1 = p1;
    fCurrent = current;
}

void KSolenoid::SetValues(double r, double z0, double z1, double current)
{
    fP0[0] = r;
    fP0[1] = 0.;
    fP0[2] = z0;
    fP1[0] = r;
    fP1[1] = 0.;
    fP1[2] = z1;
    fCurrent = current;
}

void KSolenoid::Accept(KElectromagnetVisitor& visitor)
{
    visitor.Visit(*this);
}
}  // namespace KEMField
