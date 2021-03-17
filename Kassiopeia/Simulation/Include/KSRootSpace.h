#ifndef Kassiopeia_KSRootSpace_h_
#define Kassiopeia_KSRootSpace_h_

#include "KSList.h"
#include "KSSide.h"
#include "KSSpace.h"
#include "KSSurface.h"

namespace Kassiopeia
{

class KSRootSpace : public KSComponentTemplate<KSRootSpace, KSSpace>
{
  public:
    KSRootSpace();
    KSRootSpace(const KSRootSpace& aCopy);
    KSRootSpace* Clone() const override;
    ~KSRootSpace() override;

  public:
    void Enter() const override;
    void Exit() const override;

    bool Outside(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const override;

  public:
    void AddSpace(KSSpace* aSpace);
    void RemoveSpace(KSSpace* aSpace);

    void AddSurface(KSSurface* aSurface);
    void RemoveSurface(KSSurface* aSurface);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif
