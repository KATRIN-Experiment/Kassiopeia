#ifndef KGSHELLCIRCLESURFACEBUILDER_HH_
#define KGSHELLCIRCLESURFACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGShellCircleSurface.hh"

namespace katrin
{

typedef KComplexElement<KGShellCircleSurface> KGShellCircleSurfaceBuilder;

template<> inline bool KGShellCircleSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGShellCircleSurface::SetName);
        return true;
    }

    if (anAttribute->GetName() == "angle_start") {
        anAttribute->CopyTo(fObject, &KGShellCircleSurface::AngleStart);
        return true;
    }
    if (anAttribute->GetName() == "angle_stop") {
        anAttribute->CopyTo(fObject, &KGShellCircleSurface::AngleStop);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_count") {
        anAttribute->CopyTo(fObject, &KGShellCircleSurface::ShellMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_power") {
        anAttribute->CopyTo(fObject, &KGShellCircleSurface::ShellMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGShellCircleSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
