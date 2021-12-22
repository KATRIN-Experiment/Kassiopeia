#ifndef KGELECTROMAGNET_DEF
#define KGELECTROMAGNET_DEF

#include "KElectromagnetContainer.hh"
#include "KSurfaceContainer.hh"

#include "KGCore.hh"

namespace KGeoBag
{
class KGElectromagnetData
{
  public:
    KGElectromagnetData() : fLineCurrent(0.), fCurrentTurns(1) {}
    KGElectromagnetData(KGSpace*) : fLineCurrent(0.), fCurrentTurns(1) {}
    KGElectromagnetData(KGSurface*) : fLineCurrent(0.), fCurrentTurns(1) {}
    KGElectromagnetData(KGSpace*, const KGElectromagnetData& aCopy) :
        fLineCurrent(aCopy.fLineCurrent),
        fCurrentTurns(aCopy.fCurrentTurns)
    {}
    KGElectromagnetData(KGSurface*, const KGElectromagnetData& aCopy) :
        fLineCurrent(aCopy.fLineCurrent),
        fCurrentTurns(aCopy.fCurrentTurns)
    {}

    virtual ~KGElectromagnetData() = default;

    void SetCurrent(double d);
    double GetCurrent() const;

    void SetCurrentTurns(double d);
    double GetCurrentTurns() const;

    void SetLineCurrent(double d);
    double GetLineCurrent() const;

  private:
    double fLineCurrent;
    double fCurrentTurns;
};

class KGElectromagnet
{
  public:
    using Surface = KGElectromagnetData;
    using Space = KGElectromagnetData;
};

using KGElectromagnetSurface = KGExtendedSurface<KGElectromagnet>;
using KGElectromagnetSpace = KGExtendedSpace<KGElectromagnet>;

}  // namespace KGeoBag

#endif /* KGELECTROMAGNETDATA_DEF */
