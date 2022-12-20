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

    void SetLinearMap(const katrin::KThreeMatrix& map)
    {
        fLinearMap = map;
    }
    void SetTranslation(const katrin::KThreeVector& t)
    {
        fTranslation = t;
    }

    void Apply(katrin::KThreeVector& point) const override;

  private:
    katrin::KThreeMatrix fLinearMap;
    katrin::KThreeVector fTranslation;
};
}  // namespace KGeoBag

#endif
