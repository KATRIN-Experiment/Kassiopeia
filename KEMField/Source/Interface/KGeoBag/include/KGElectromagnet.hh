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
    KGElectromagnetData() : fCurrent(0.) {}
    KGElectromagnetData(KGSpace*) : fCurrent(0.) {}
    KGElectromagnetData(KGSurface*) : fCurrent(0.) {}
    KGElectromagnetData(KGSpace*, const KGElectromagnetData& aCopy) : fCurrent(aCopy.fCurrent) {}
    KGElectromagnetData(KGSurface*, const KGElectromagnetData& aCopy) : fCurrent(aCopy.fCurrent) {}

    virtual ~KGElectromagnetData() {}

    void SetCurrent(double d);
    double GetCurrent() const;

  private:
    double fCurrent;
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
