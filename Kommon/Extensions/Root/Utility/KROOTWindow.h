#ifndef _katrin_KROOTWindow_h_
#define _katrin_KROOTWindow_h_

#include "KWindow.h"
#include "KField.h"
#include "KFile.h"

#include <vector>
using std::vector;

#include "TH2F.h"
#include "TApplication.h"
#include "TCanvas.h"
#include <cstdlib>
#include <iostream>

namespace katrin
{

    class KROOTPainter;
    class KROOTPad;

    class KROOTWindow :
        public KWindow
    {

        public:
    		KROOTWindow();
            virtual ~KROOTWindow();

            void Render();
            void Display();
            void Write();

            void AddPainter( KPainter* aPainter );
            void RemovePainter( KPainter* aPainter );

            void AddWindow( KWindow* aWindow );
            void RemoveWindow( KWindow* aWindow );

            TPad* GetPad();

        private:
            typedef vector< KROOTPainter* > PainterVector;
            typedef PainterVector::iterator PainterIt;
            PainterVector fPainters;
            typedef vector< KROOTPad* > PadVector;
            typedef PadVector::iterator PadIt;
            PadVector fPads;
            TH2F* fFrame;
            TApplication* fApplication;
            TCanvas* fCanvas;

            //settings
            ;K_SET( unsigned int, CanvasWidth );
            ;K_SET( unsigned int, CanvasHeight );
            ;K_SET( bool, Active );
            ;K_SET( bool, WriteEnabled );
            ;K_SET( std::string, Path );

    };


}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KComplexElement.hh"

namespace katrin
{

    typedef KComplexElement< KROOTWindow > KROOTWindowBuilder;

    template< >
    inline bool KROOTWindowBuilder::Begin()
    {
        fObject = new KROOTWindow();
        return true;
    }

    template< >
    inline bool KROOTWindowBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "canvas_width" )
        {
            aContainer->CopyTo( fObject, &KROOTWindow::SetCanvasWidth );
            return true;
        }
        if( aContainer->GetName() == "canvas_height" )
        {
            aContainer->CopyTo( fObject, &KROOTWindow::SetCanvasHeight );
            return true;
        }
        if( aContainer->GetName() == "active" )
        {
            aContainer->CopyTo( fObject, &KROOTWindow::SetActive );
            return true;
        }
        if( aContainer->GetName() == "write_enabled" )
        {
            aContainer->CopyTo( fObject, &KROOTWindow::SetWriteEnabled );
            return true;
        }
        if( aContainer->GetName() == "path" )
        {
            aContainer->CopyTo( fObject, &KROOTWindow::SetPath );
            return true;
        }
        return false;
    }

    template< >
    inline bool KROOTWindowBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KPainter >() == true )
        {
            aContainer->ReleaseTo( fObject, &KROOTWindow::AddPainter );
            return true;
        }
        if( aContainer->Is< KWindow >() == true )
        {
            aContainer->ReleaseTo( fObject, &KROOTWindow::AddWindow );
            return true;
        }
        return false;
    }

    template< >
    inline bool KROOTWindowBuilder::End()
    {
        fObject->Render();
        fObject->Display();
        fObject->Write();
        delete fObject;
        return true;
    }

}

#endif
