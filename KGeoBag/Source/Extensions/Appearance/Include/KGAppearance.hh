#ifndef KGAPPEARANCE_HH_
#define KGAPPEARANCE_HH_

#include "KGCore.hh"
#include "KGRGBAColor.hh"

namespace KGeoBag
{

class KGAppearanceData
{
  public:
    KGAppearanceData() : fColor(), fArc(120) {}
    KGAppearanceData(KGSpace*) : fColor(), fArc(120) {}
    KGAppearanceData(KGSurface*) : fColor(), fArc(120) {}
    KGAppearanceData(KGSpace*, const KGAppearanceData& aCopy) : fColor(aCopy.fColor), fArc(aCopy.fArc) {}
    KGAppearanceData(KGSurface*, const KGAppearanceData& aCopy) : fColor(aCopy.fColor), fArc(aCopy.fArc) {}
    virtual ~KGAppearanceData() = default;

  public:
    void SetColor(const KGRGBAColor& aColor);
    const KGRGBAColor& GetColor() const;

    void SetArc(const unsigned int& anArc);
    const unsigned int& GetArc() const;

  private:
    KGRGBAColor fColor;
    unsigned int fArc;
};

class KGAppearance
{
  public:
    typedef KGAppearanceData Surface;
    using Space = KGAppearanceData;
};

using KGAppearanceSurface = KGExtendedSurface<KGAppearance>;
using KGAppearanceSpace = KGExtendedSpace<KGAppearance>;

}  // namespace KGeoBag

#endif
