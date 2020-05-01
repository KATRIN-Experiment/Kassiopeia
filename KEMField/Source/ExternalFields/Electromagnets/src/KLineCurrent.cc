#include "KLineCurrent.hh"

#include "KEMConstants.hh"
#include "KElectromagnetVisitor.hh"

namespace KEMField
{
void KLineCurrent::SetValues(const KPosition& p0, const KPosition& p1, double current)
{
    fP0 = p0;
    fP1 = p1;
    fCurrent = current;
}

void KLineCurrent::Accept(KElectromagnetVisitor& visitor)
{
    visitor.Visit(*this);
}
}  // namespace KEMField
