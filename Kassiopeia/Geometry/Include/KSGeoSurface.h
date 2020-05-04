#ifndef Kassiopeia_KSGeoSurface_h_
#define Kassiopeia_KSGeoSurface_h_

#include "KGCore.hh"
#include "KSSurface.h"
using namespace KGeoBag;

namespace Kassiopeia
{

class KSGeoSpace;

class KSGeoSurface : public KSComponentTemplate<KSGeoSurface, KSSurface>
{
  public:
    friend class KSGeoSpace;

  public:
    KSGeoSurface();
    KSGeoSurface(const KSGeoSurface& aCopy);
    KSGeoSurface* Clone() const override;
    ~KSGeoSurface() override;

  public:
    void On() const override;
    void Off() const override;

    KThreeVector Point(const KThreeVector& aPoint) const override;
    KThreeVector Normal(const KThreeVector& aPoint) const override;

  public:
    void AddContent(KGSurface* aSurface);
    void RemoveContent(KGSurface* aSurface);
    std::vector<KGSurface*> GetContent();

    void AddCommand(KSCommand* anCommand);
    void RemoveCommand(KSCommand* anCommand);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    KSGeoSpace* fParent;

    mutable std::vector<KGSurface*> fContents;
    mutable std::vector<KSCommand*> fCommands;
};

}  // namespace Kassiopeia

#endif
