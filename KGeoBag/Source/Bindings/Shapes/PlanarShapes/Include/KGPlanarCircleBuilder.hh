#ifndef KGPLANARCIRCLEBUILDER_HH_
#define KGPLANARCIRCLEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGPlanarCircle.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGPlanarCircle> KGPlanarCircleBuilder;

template<> inline bool KGPlanarCircleBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x") {
        anAttribute->CopyTo(fObject, &KGPlanarCircle::X);
        return true;
    }
    if (anAttribute->GetName() == "y") {
        anAttribute->CopyTo(fObject, &KGPlanarCircle::Y);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGPlanarCircle::Radius);
        return true;
    }
    if (anAttribute->GetName() == "circle_mesh_count") {
        anAttribute->CopyTo(fObject, &KGPlanarCircle::MeshCount);
        return true;
    }
    return true;
}

}  // namespace katrin

#endif
