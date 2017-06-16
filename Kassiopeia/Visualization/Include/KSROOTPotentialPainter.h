#ifndef KSROOTPOTENTIALPAINTER_H
#define KSROOTPOTENTIALPAINTER_H

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KGCore.hh"

#include "KField.h"
#include "KSElectricField.h"

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

        public:
            bool CheckPosition( const KThreeVector& aPosition ) const;

        private:
            ;K_SET( std::string, XAxis );
            ;K_SET( std::string, YAxis );
            ;K_SET( double, Zmin );
            ;K_SET( double, Zmax );
            ;K_SET( double, Rmax );
            ;K_SET( int, Zsteps  );
            ;K_SET( int, Rsteps  );
            ;K_SET( std::string, ElectricFieldName );
            ;K_SET( bool, CalcPot);
            TH2D* fMap;
            ;K_SET( bool, Comparison );
            ;K_SET( std::string, ReferenceFieldName );
    };

}

#endif // KSROOTPOTENTIALPAINTER_H
