#ifndef KSROOTMAGFIELDPAINTER_H
#define KSROOTMAGFIELDPAINTER_H

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KField.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

    class KSROOTMagFieldPainter :
        public KROOTPainter
    {
        public:
        KSROOTMagFieldPainter();
            ~KSROOTMagFieldPainter();

            virtual void Render();
            virtual void Display();
            virtual void Write();
            virtual void FieldMapX(KSMagneticField* tMagField, double tDeltaZ, double tDeltaR);
            virtual void FieldMapZ(KSMagneticField* tMagField, double tDeltaZ, double tDeltaR);

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
            ;K_SET( double, Zfix );
            ;K_SET( double, Rmax );
            ;K_SET( int, Zsteps  );
            ;K_SET( int, Rsteps  );
            ;K_SET( string, MagneticFieldName );
            ;K_SET( bool, AxialSymmetry);
            ;K_SET( string, Plot );
            ;K_SET( bool, UseLogZ );
            ;K_SET( bool, GradNumerical);
            ;K_SET( string, Draw);
            TH2D* fMap;

    };

}

#endif // KSROOTMAGFIELDPAINTER_H
