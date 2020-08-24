#include "KGElectromagnet.hh"

namespace KGeoBag
{
void KGElectromagnetData::SetCurrent(double d)
{
    fLineCurrent = d;
    fCurrentTurns = 1;
}
double KGElectromagnetData::GetCurrent() const
{
    return fLineCurrent * fCurrentTurns;
}

void KGElectromagnetData::SetCurrentTurns(double d)
{
    fCurrentTurns = d;
}
double KGElectromagnetData::GetCurrentTurns() const
{
    return fCurrentTurns;
}

void KGElectromagnetData::SetLineCurrent(double d)
{
    fLineCurrent = d;
}
double KGElectromagnetData::GetLineCurrent() const
{
    return fLineCurrent;
}

}  // namespace KGeoBag
