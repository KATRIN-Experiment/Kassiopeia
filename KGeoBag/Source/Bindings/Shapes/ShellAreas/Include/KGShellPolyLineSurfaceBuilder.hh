#ifndef KGSHELLPOLYLINESURFACEBUILDER_HH_
#define KGSHELLPOLYLINESURFACEBUILDER_HH_

#include "KGPlanarPolyLineBuilder.hh"
#include "KGShellPolyLineSurface.hh"

namespace katrin
{

typedef KComplexElement<KGShellPolyLineSurface> KGShellPolyLineSurfaceBuilder;

template<> inline bool KGShellPolyLineSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_count") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::ShellMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "angle_start") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::AngleStart);
        return true;
    }
    if (anAttribute->GetName() == "angle_stop") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::AngleStop);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_power") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::ShellMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGShellPolyLineSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_line") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLine::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
