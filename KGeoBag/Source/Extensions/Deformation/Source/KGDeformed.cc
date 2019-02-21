#include "KGDeformed.hh"

namespace KGeoBag
{
  void KGDeformedObject::SetDeformation(std::shared_ptr<KGDeformation> deformation)
  {
    fDeformation = deformation;
  }

  std::shared_ptr<KGDeformation> KGDeformedObject::GetDeformation() const
  {
    return fDeformation;
  }
}
