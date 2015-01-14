#ifndef KGAFFINEDEFORMATION_HH_
#define KGAFFINEDEFORMATION_HH_

#include "KThreeMatrix.hh"

#include "KGDeformation.hh"

namespace KGeoBag
{

  class KGAffineDeformation : public KGDeformation
  {
  public:
    KGAffineDeformation();
    KGAffineDeformation(const KGAffineDeformation& affine);

    virtual ~KGAffineDeformation() {}

    void SetLinearMap(const KThreeMatrix& map) { fLinearMap = map; }
    void SetTranslation(const KThreeVector& t) { fTranslation = t; }

    virtual void Apply(KThreeVector& point) const;

  private:
    KThreeMatrix fLinearMap;
    KThreeVector fTranslation;
  };
}

#endif
