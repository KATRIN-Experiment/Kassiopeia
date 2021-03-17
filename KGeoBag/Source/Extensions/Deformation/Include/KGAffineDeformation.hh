#ifndef KGAFFINEDEFORMATION_HH_
#define KGAFFINEDEFORMATION_HH_

#include "KGDeformation.hh"
#include "KThreeMatrix.hh"

namespace KGeoBag
{

class KGAffineDeformation : public KGDeformation
{
  public:
    KGAffineDeformation();
    KGAffineDeformation(const KGAffineDeformation& affine);

    ~KGAffineDeformation() override = default;

    void SetLinearMap(const KThreeMatrix& map)
    {
        fLinearMap = map;
    }
    void SetTranslation(const KGeoBag::KThreeVector& t)
    {
        fTranslation = t;
    }

    void Apply(KThreeVector& point) const override;

  private:
    KThreeMatrix fLinearMap;
    KGeoBag::KThreeVector fTranslation;
};
}  // namespace KGeoBag

#endif
