#include "KGRGBColor.hh"

namespace KGeoBag
{

KGRGBColor::KGRGBColor() : fRed(127), fGreen(127), fBlue(127) {}
KGRGBColor::KGRGBColor(int aRed, int aGreen, int aBlue) : fRed(aRed), fGreen(aGreen), fBlue(aBlue) {}
KGRGBColor::KGRGBColor(const KGRGBColor& aColor) : fRed(aColor.fRed), fGreen(aColor.fGreen), fBlue(aColor.fBlue) {}
KGRGBColor::~KGRGBColor() {}

void KGRGBColor::SetRed(const unsigned char& aRed)
{
    fRed = aRed;
    return;
}
const unsigned char& KGRGBColor::GetRed() const
{
    return fRed;
}
void KGRGBColor::SetGreen(const unsigned char& aGreen)
{
    fGreen = aGreen;
    return;
}
const unsigned char& KGRGBColor::GetGreen() const
{
    return fGreen;
}
void KGRGBColor::SetBlue(const unsigned char& aBlue)
{
    fBlue = aBlue;
    return;
}
const unsigned char& KGRGBColor::GetBlue() const
{
    return fBlue;
}

}  // namespace KGeoBag
