#include "KVTKPainter.h"

#include "KUtilityMessage.h"
#include "KVTKWindow.h"

namespace katrin
{

KVTKPainter::KVTKPainter() : fWindow(nullptr), fDisplayEnabled(true), fWriteEnabled(true) {}

KVTKPainter::~KVTKPainter() = default;

void KVTKPainter::SetWindow(KWindow* aWindow)
{
    auto* tWindow = dynamic_cast<KVTKWindow*>(aWindow);
    if (tWindow != nullptr) {
        if (fWindow == nullptr) {
            fWindow = tWindow;
            return;
        }
        utilmsg(eError) << "cannot use vtk window <" << tWindow->GetName() << "> with vtk painter <" << GetName() << ">"
                        << eom;
    }
    utilmsg(eError) << "cannot use non-vtk window <" << aWindow->GetName() << "> with vtk painter <" << GetName() << ">"
                    << eom;
    return;
}

void KVTKPainter::ClearWindow(KWindow* aWindow)
{
    auto* tWindow = dynamic_cast<KVTKWindow*>(aWindow);
    if (tWindow != nullptr) {
        if (fWindow == tWindow) {
            fWindow = nullptr;
            return;
        }
        utilmsg(eError) << "cannot use vtk window <" << tWindow->GetName() << "> with vtk painter <" << GetName() << ">"
                        << eom;
    }
    utilmsg(eError) << "cannot use non-vtk window <" << aWindow->GetName() << "> with vtk painter <" << GetName() << ">"
                    << eom;
    return;
}

void KVTKPainter::SetDisplayMode(bool aMode)
{
    fDisplayEnabled = aMode;
    return;
}

void KVTKPainter::SetWriteMode(bool aMode)
{
    fWriteEnabled = aMode;
    return;
}

std::string KVTKPainter::HelpText()
{
    return "";
}

void KVTKPainter::OnKeyPress(vtkObject* /*caller*/, long unsigned int /*eventId*/, void* /*client*/, void* /*callData*/)
{
    return;
}

}  // namespace katrin
