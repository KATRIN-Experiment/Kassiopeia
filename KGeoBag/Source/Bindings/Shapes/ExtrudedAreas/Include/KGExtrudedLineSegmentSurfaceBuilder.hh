#ifndef KGEXTRUDEDLINESEGMENTSURFACEBUILDER_HH_
#define KGEXTRUDEDLINESEGMENTSURFACEBUILDER_HH_

#include "KGExtrudedLineSegmentSurface.hh"
#include "KGPlanarLineSegmentBuilder.hh"

namespace katrin
{

typedef KComplexElement<KGExtrudedLineSegmentSurface> KGExtrudedLineSegmentSurfaceBuilder;

template<> inline bool KGExtrudedLineSegmentSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedLineSegmentSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedLineSegmentSurface::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedLineSegmentSurface::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedLineSegmentSurface::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedLineSegmentSurface::ExtrudedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedLineSegmentSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "line_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarLineSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
