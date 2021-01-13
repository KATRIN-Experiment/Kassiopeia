#include "KCoil.hh"

#include "KElectromagnetVisitor.hh"

namespace KEMField
{
void KCoil::SetValues(const KPosition& p0, const KPosition& p1, double current, int integrationScale)
{
    fP0 = p0;
    fP1 = p1;
    fCurrent = current;
    fIntegrationScale = integrationScale;
}

void KCoil::SetValues(double r0, double r1, double z0, double z1, double current, int integrationScale)
{
    fP0[0] = r0;
    fP0[1] = 0.;
    fP0[2] = z0;
    fP1[0] = r1;
    fP1[1] = 0.;
    fP1[2] = z1;
    fCurrent = current;
    fIntegrationScale = integrationScale;
}

void KCoil::Accept(KElectromagnetVisitor& visitor)
{
    visitor.Visit(*this);
}
}  // namespace KEMField
