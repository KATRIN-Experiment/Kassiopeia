#ifndef KGROTATEDARCSEGMENTSURFACEBUILDER_HH_
#define KGROTATEDARCSEGMENTSURFACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGRotatedArcSegmentSurface.hh"

namespace katrin
{

typedef KComplexElement<KGRotatedArcSegmentSurface> KGRotatedArcSegmentSurfaceBuilder;

template<> inline bool KGRotatedArcSegmentSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedArcSegmentSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        fObject->RotatedMeshCount(anAttribute->AsReference<unsigned int>());
        return true;
    }
    return false;
}

template<> inline bool KGRotatedArcSegmentSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "arc_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarArcSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
