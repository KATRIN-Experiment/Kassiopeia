#include "KVTKPainter.h"

#include "KUtilityMessage.h"
#include "KVTKWindow.h"

namespace katrin
{

KVTKPainter::KVTKPainter() : fWindow(NULL), fDisplayEnabled(true), fWriteEnabled(true) {}

KVTKPainter::~KVTKPainter() {}

void KVTKPainter::SetWindow(KWindow* aWindow)
{
    KVTKWindow* tWindow = dynamic_cast<KVTKWindow*>(aWindow);
    if (tWindow != NULL) {
        if (fWindow == NULL) {
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
    KVTKWindow* tWindow = dynamic_cast<KVTKWindow*>(aWindow);
    if (tWindow != NULL) {
        if (fWindow == tWindow) {
            fWindow = NULL;
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

}  // namespace katrin
