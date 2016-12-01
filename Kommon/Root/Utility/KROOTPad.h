#ifndef _katrin_KROOTPad_h_
#define _katrin_KROOTPad_h_

#include "KField.h"
#include "KROOTWindow.h"
#include "KPainter.h"

#include <vector>
using std::vector;

#include "TH2F.h"
#include "TPad.h"
#include "TCanvas.h"
#include <cstdlib>
#include <iostream>

namespace katrin
{

    class KROOTPainter;

    class KROOTPad :
		public KROOTWindow
    {

        public:
    		KROOTPad();
            virtual ~KROOTPad();

            void Render();
            void Display();
            void Write();

            void AddPainter( KPainter* aPainter );
            void RemovePainter( KPainter* aPainter );

            void SetWindow( KWindow* aWindow );
            void ClearWindow( KWindow* aWindow );

        private:
            typedef vector< KROOTPainter* > PainterVector;
            typedef PainterVector::iterator PainterIt;
            PainterVector fPainters;
            TH2F* fFrame;
            TPad* fPad;
            KROOTWindow* fWindow;

            //settings
            ;K_SET( double, xlow );
            ;K_SET( double, ylow );
            ;K_SET( double, xup );
            ;K_SET( double, yup );

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



#endif
