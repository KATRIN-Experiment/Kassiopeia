#include "KGDeformed.hh"

namespace KGeoBag
{
  void KGDeformedObject::SetDeformation(KSmartPointer<KGDeformation> deformation)
  {
    fDeformation = deformation;
  }

  KSmartPointer<KGDeformation> KGDeformedObject::GetDeformation() const
  {
    return fDeformation;
  }
}
