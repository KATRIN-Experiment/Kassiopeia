#ifndef Kassiopeia_KSGeoSide_h_
#define Kassiopeia_KSGeoSide_h_

#include "KGCore.hh"
#include "KSSide.h"
using namespace KGeoBag;

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
    KSGeoSpace* fOutsideParent;
    KSGeoSpace* fInsideParent;

    mutable std::vector<KGSurface*> fContents;
    mutable std::vector<KSCommand*> fCommands;
};

}  // namespace Kassiopeia

#endif
