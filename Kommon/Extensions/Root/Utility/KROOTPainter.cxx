#include "KROOTPainter.h"
#include "KROOTWindow.h"
#include "KUtilityMessage.h"

namespace katrin
{

    KROOTPainter::KROOTPainter() :
        fWindow( NULL ),
        fDisplayEnabled( true ),
        fWriteEnabled( true )
    {
    }

    KROOTPainter::~KROOTPainter()
    {
    }

    void KROOTPainter::SetWindow( KWindow* aWindow )
    {
    	KROOTWindow* tWindow = dynamic_cast< KROOTWindow* >( aWindow );
        if( tWindow != NULL )
        {
            if( fWindow == NULL )
            {
                fWindow = tWindow;
                return;
            }
            utilmsg( eError ) << "cannot use root window <" << tWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
        }
        utilmsg( eError ) << "cannot use non-root window <" << aWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
        return;
    }

    void KROOTPainter::ClearWindow( KWindow* aWindow )
    {
    	KROOTWindow* tWindow = dynamic_cast< KROOTWindow* >( aWindow );
        if( tWindow != NULL )
        {
            if( fWindow == tWindow )
            {
                fWindow = NULL;
                return;
            }
            utilmsg( eError ) << "cannot use root window <" << tWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
        }
        return;
        utilmsg( eError ) << "cannot use non-root window <" << aWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
    }

    void KROOTPainter::SetDisplayMode( bool aMode )
    {
        fDisplayEnabled = aMode;
        return;
    }

    void KROOTPainter::SetWriteMode( bool aMode )
    {
        fWriteEnabled = aMode;
        return;
    }

}
