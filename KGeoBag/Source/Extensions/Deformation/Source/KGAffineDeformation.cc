#include "KGAffineDeformation.hh"

namespace KGeoBag
{
  KGAffineDeformation::KGAffineDeformation() : KGDeformation()
  {
    fLinearMap[0] = 1.; fLinearMap[1] = 0.; fLinearMap[2] = 0.;
    fLinearMap[3] = 0.; fLinearMap[4] = 1.; fLinearMap[5] = 0.;
    fLinearMap[6] = 0.; fLinearMap[7] = 0.; fLinearMap[8] = 1.;

    fTranslation[0] = fTranslation[1] = fTranslation[2] = 0.;
  }

  KGAffineDeformation::KGAffineDeformation(const KGAffineDeformation& affine)
  {
    fLinearMap = affine.fLinearMap;
    fTranslation = affine.fTranslation;
  }

  void KGAffineDeformation::Apply(KThreeVector& point) const
  {
    point = fLinearMap*point + fTranslation;
  }
}
