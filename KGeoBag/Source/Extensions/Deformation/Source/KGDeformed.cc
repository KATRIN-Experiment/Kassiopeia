#include "KGDeformed.hh"

#include <utility>

namespace KGeoBag
{
void KGDeformedObject::SetDeformation(std::shared_ptr<KGDeformation> deformation)
{
    fDeformation = std::move(deformation);
}

std::shared_ptr<KGDeformation> KGDeformedObject::GetDeformation() const
{
    return fDeformation;
}
}  // namespace KGeoBag
