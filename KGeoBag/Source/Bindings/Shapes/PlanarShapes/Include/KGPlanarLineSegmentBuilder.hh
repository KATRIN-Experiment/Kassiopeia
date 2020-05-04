#ifndef KGPLANARLINESEGMENTBUILDER_HH_
#define KGPLANARLINESEGMENTBUILDER_HH_

#include "KComplexElement.hh"
#include "KGPlanarLineSegment.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGPlanarLineSegment> KGPlanarLineSegmentBuilder;

template<> inline bool KGPlanarLineSegmentBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x1") {
        anAttribute->CopyTo(fObject, &KGPlanarLineSegment::X1);
        return true;
    }
    if (anAttribute->GetName() == "y1") {
        anAttribute->CopyTo(fObject, &KGPlanarLineSegment::Y1);
        return true;
    }
    if (anAttribute->GetName() == "x2") {
        anAttribute->CopyTo(fObject, &KGPlanarLineSegment::X2);
        return true;
    }
    if (anAttribute->GetName() == "y2") {
        anAttribute->CopyTo(fObject, &KGPlanarLineSegment::Y2);
        return true;
    }
    if (anAttribute->GetName() == "line_mesh_count") {
        anAttribute->CopyTo(fObject, &KGPlanarLineSegment::MeshCount);
        return true;
    }
    if (anAttribute->GetName() == "line_mesh_power") {
        anAttribute->CopyTo(fObject, &KGPlanarLineSegment::MeshPower);
        return true;
    }
    return true;
}

}  // namespace katrin

#endif
