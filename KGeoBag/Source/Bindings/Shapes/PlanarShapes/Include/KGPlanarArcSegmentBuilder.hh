#ifndef KGPLANARARCSEGMENTBUILDER_HH_
#define KGPLANARARCSEGMENTBUILDER_HH_

#include "KComplexElement.hh"
#include "KGPlanarArcSegment.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGPlanarArcSegment> KGPlanarArcSegmentBuilder;

template<> inline bool KGPlanarArcSegmentBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x1") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::X1);
        return true;
    }
    if (anAttribute->GetName() == "y1") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::Y1);
        return true;
    }
    if (anAttribute->GetName() == "x2") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::X2);
        return true;
    }
    if (anAttribute->GetName() == "y2") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::Y2);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::Radius);
        return true;
    }
    if (anAttribute->GetName() == "right") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::Right);
        return true;
    }
    if (anAttribute->GetName() == "short") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::Short);
        return true;
    }
    if (anAttribute->GetName() == "arc_mesh_count") {
        anAttribute->CopyTo(fObject, &KGPlanarArcSegment::MeshCount);
        return true;
    }
    return true;
}

}  // namespace katrin

#endif
