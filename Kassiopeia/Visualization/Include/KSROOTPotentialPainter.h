#ifndef KSROOTPOTENTIALPAINTER_H
#define KSROOTPOTENTIALPAINTER_H

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KField.h"

namespace Kassiopeia
{

    class KSROOTPotentialPainter :
        public KROOTPainter
    {
        public:
    	KSROOTPotentialPainter();
            ~KSROOTPotentialPainter();

            virtual void Render();
            virtual void Display();
            virtual void Write();

            virtual double GetXMin();
            virtual double GetXMax();
            virtual double GetYMin();
            virtual double GetYMax();

            virtual std::string GetXAxisLabel();
            virtual std::string GetYAxisLabel();

        private:
            ;K_SET( string, XAxis );
            ;K_SET( string, YAxis );
            ;K_SET( double, Zmin );
            ;K_SET( double, Zmax );
            ;K_SET( double, Rmax );
            ;K_SET( int, Zsteps  );
            ;K_SET( int, Rsteps  );
            ;K_SET( string, ElectricFieldName );
            ;K_SET( bool, CalcPot);
            TH2D* fMap;

    };

}

#endif // KSROOTPOTENTIALPAINTER_H
