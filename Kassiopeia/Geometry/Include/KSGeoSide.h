#ifndef Kassiopeia_KSGeoSide_h_
#define Kassiopeia_KSGeoSide_h_

#include "KGCore.hh"
#include "KSSide.h"

namespace Kassiopeia
{

class KSGeoSpace;

class KSGeoSide : public KSComponentTemplate<KSGeoSide, KSSide>
{
  public:
    friend class KSGeoSpace;

  public:
    KSGeoSide();
    KSGeoSide(const KSGeoSide& aCopy);
    KSGeoSide* Clone() const override;
    ~KSGeoSide() override;

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
    KSGeoSpace* fOutsideParent;
    KSGeoSpace* fInsideParent;

    mutable std::vector<KGeoBag::KGSurface*> fContents;
    mutable std::vector<KSCommand*> fCommands;
};

}  // namespace Kassiopeia

#endif
