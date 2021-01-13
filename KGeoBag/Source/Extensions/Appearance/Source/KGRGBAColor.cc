#include "KGRGBAColor.hh"

namespace KGeoBag
{

KGRGBAColor::KGRGBAColor() : fOpacity(255) {}
KGRGBAColor::KGRGBAColor(int aRed, int aGreen, int aBlue, int anOpacity) :
    KGRGBColor(aRed, aGreen, aBlue),
    fOpacity(anOpacity)
{}
KGRGBAColor::KGRGBAColor(const KGRGBAColor&) = default;
KGRGBAColor::~KGRGBAColor() = default;

void KGRGBAColor::SetOpacity(const unsigned char& anOpacity)
{
    fOpacity = anOpacity;
    return;
}
const unsigned char& KGRGBAColor::GetOpacity() const
{
    return fOpacity;
}

}  // namespace KGeoBag
