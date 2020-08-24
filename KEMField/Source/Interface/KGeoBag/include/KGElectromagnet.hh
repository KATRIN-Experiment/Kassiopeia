#ifndef KGELECTROMAGNET_DEF
#define KGELECTROMAGNET_DEF

#include "KElectromagnetContainer.hh"
#include "KSurfaceContainer.hh"

using KEMField::KMagnetostaticBasis;

using KEMField::KDirection;
using KEMField::KGradient;
using KEMField::KPosition;
using KGeoBag::KThreeVector;

using KEMField::KCoil;
using KEMField::KLineCurrent;
using KEMField::KSolenoid;

using KEMField::KElectromagnetContainer;

#include "KGCore.hh"

namespace KGeoBag
{
class KGElectromagnetData
{
  public:
    KGElectromagnetData() : fLineCurrent(0.), fCurrentTurns(1) {}
    KGElectromagnetData(KGSpace*) : fLineCurrent(0.), fCurrentTurns(1) {}
    KGElectromagnetData(KGSurface*) : fLineCurrent(0.), fCurrentTurns(1) {}
    KGElectromagnetData(KGSpace*, const KGElectromagnetData& aCopy) : fLineCurrent(aCopy.fLineCurrent), fCurrentTurns(aCopy.fCurrentTurns) {}
    KGElectromagnetData(KGSurface*, const KGElectromagnetData& aCopy) : fLineCurrent(aCopy.fLineCurrent), fCurrentTurns(aCopy.fCurrentTurns) {}

    virtual ~KGElectromagnetData() {}

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
    typedef KGElectromagnetData Surface;
    typedef KGElectromagnetData Space;
};

typedef KGExtendedSurface<KGElectromagnet> KGElectromagnetSurface;
typedef KGExtendedSpace<KGElectromagnet> KGElectromagnetSpace;

}  // namespace KGeoBag

#endif /* KGELECTROMAGNETDATA_DEF */
