#include "KGAppearance.hh"

namespace KGeoBag
{
void KGAppearanceData::SetColor(const KGRGBAColor& aColor)
{
    fColor = aColor;
    return;
}
const KGRGBAColor& KGAppearanceData::GetColor() const
{
    return fColor;
}

void KGAppearanceData::SetArc(const unsigned int& anArc)
{
    fArc = anArc;
    return;
}
const unsigned int& KGAppearanceData::GetArc() const
{
    return fArc;
}


}  // namespace KGeoBag
