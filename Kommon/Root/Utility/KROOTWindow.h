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

#endif
