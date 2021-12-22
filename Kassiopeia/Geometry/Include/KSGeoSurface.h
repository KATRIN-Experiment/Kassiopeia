#ifndef Kassiopeia_KSGeoSurface_h_
#define Kassiopeia_KSGeoSurface_h_

#include "KGCore.hh"
#include "KSSurface.h"

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

    katrin::KThreeVector Point(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector Normal(const katrin::KThreeVector& aPoint) const override;

  public:
    void AddContent(KGeoBag::KGSurface* aSurface);
    void RemoveContent(KGeoBag::KGSurface* aSurface);
    std::vector<KGeoBag::KGSurface*> GetContent();

    void AddCommand(KSCommand* anCommand);
    void RemoveCommand(KSCommand* anCommand);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    KSGeoSpace* fParent;

    mutable std::vector<KGeoBag::KGSurface*> fContents;
    mutable std::vector<KSCommand*> fCommands;
};

}  // namespace Kassiopeia

#endif
