#include "KGElectromagnet.hh"

namespace KGeoBag
{
void KGElectromagnetData::SetCurrent(double d)
{
    fCurrent = d;
}
double KGElectromagnetData::GetCurrent() const
{
    return fCurrent;
}

}  // namespace KGeoBag
