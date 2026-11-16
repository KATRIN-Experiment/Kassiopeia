#ifndef KGSHELLCIRCLESURFACEBUILDER_HH_
#define KGSHELLCIRCLESURFACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGShellCircleSurface.hh"

namespace katrin
{

//! This produces an angularly limited portion of a surface of revolution from a circle that is rotated around the local
//! x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_shell_circle_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <shell_circle_surface name="shell_circle_surface" angle_start="200" angle_stop="130" shell_mesh_count="32" shell_mesh_power="6">
//!     <circle x="0.1" y="0.5" radius="0.3" circle_mesh_count="32"/>
//! </shell_circle_surface>
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
