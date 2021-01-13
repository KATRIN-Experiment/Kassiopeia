#include "KGDiscreteRotationalMesherBase.hh"

namespace KGeoBag
{

KGDiscreteRotationalMesherBase::KGDiscreteRotationalMesherBase() : fCurrentElements(nullptr) {}
KGDiscreteRotationalMesherBase::~KGDiscreteRotationalMesherBase() = default;

void KGDiscreteRotationalMesherBase::SetMeshElementOutput(KGDiscreteRotationalMeshElementVector* aMeshElementVector)

{
    fCurrentElements = aMeshElementVector;
    return;
}

}  // namespace KGeoBag
