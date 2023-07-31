#include "KROOTPainter.h"

#include "KROOTWindow.h"
#include "KUtilityMessage.h"

namespace katrin
{

KROOTPainter::KROOTPainter() : fWindow(nullptr), fDisplayEnabled(true), fWriteEnabled(true) {}

KROOTPainter::~KROOTPainter() = default;

void KROOTPainter::SetWindow(KWindow* aWindow)
{
    auto* tWindow = dynamic_cast<KROOTWindow*>(aWindow);
    if (tWindow != nullptr) {
        if (fWindow == nullptr) {
            fWindow = tWindow;
            return;
        }
        utilmsg(eError) << "cannot use root window <" << tWindow->GetName() << "> with root painter <" << GetName()
                        << ">" << eom;
    }
    utilmsg(eError) << "cannot use non-root window <" << aWindow->GetName() << "> with root painter <" << GetName()
                    << ">" << eom;
    return;
}

void KROOTPainter::ClearWindow(KWindow* aWindow)
{
    auto* tWindow = dynamic_cast<KROOTWindow*>(aWindow);
    if (tWindow != nullptr) {
        if (fWindow == tWindow) {
            fWindow = nullptr;
            return;
        }
        utilmsg(eError) << "cannot use root window <" << tWindow->GetName() << "> with root painter <" << GetName()
                        << ">" << eom;
    }
    return;
    utilmsg(eError) << "cannot use non-root window <" << aWindow->GetName() << "> with root painter <" << GetName()
                    << ">" << eom;
}

void KROOTPainter::SetDisplayMode(bool aMode)
{
    fDisplayEnabled = aMode;
    return;
}

void KROOTPainter::SetWriteMode(bool aMode)
{
    fWriteEnabled = aMode;
    return;
}

KROOTWindow* KROOTPainter::GetWindow()
{
    return fWindow;
}

}  // namespace katrin
