#include "KCurrentLoop.hh"

#include "KElectromagnetVisitor.hh"

namespace KEMField
{
void KCurrentLoop::SetValues(const KPosition& p, double current)
{
    fP = p;
    fCurrent = current;
}

void KCurrentLoop::SetValues(double r, double z, double current)
{
    fP[0] = r;
    fP[1] = 0.;
    fP[2] = z;
    fCurrent = current;
}

void KCurrentLoop::Accept(KElectromagnetVisitor& visitor)
{
    visitor.Visit(*this);
}
}  // namespace KEMField
