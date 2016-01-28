#ifndef _Kassiopeia_KSROOTTrackPainter_h_
#define _Kassiopeia_KSROOTTrackPainter_h_

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KField.h"

#include "TMultiGraph.h"
#include "TColor.h"

#include "KSVisualizationMessage.h"

namespace Kassiopeia
{

    class KSROOTTrackPainter :
        public KROOTPainter
    {
        public:
    		KSROOTTrackPainter();
            ~KSROOTTrackPainter();

            virtual void Render();
            virtual void Display();
            virtual void Write();

            virtual double GetXMin();
            virtual double GetXMax();
            virtual double GetYMin();
            virtual double GetYMax();

            virtual std::string GetXAxisLabel();
            virtual std::string GetYAxisLabel();

            void AddBaseColor(TColor aColor, double aFraction );

            typedef enum
            {
            	eColorFix, eColorStep, eColorTrack
            } ColorMode;

            typedef enum
            {
                eColorFPDRings, eColorDefault, eColorCustom
            } ColorPalette;

            typedef enum
            {
                ePlotStep, ePlotTrack
            } PlotMode;

        private:
            void CreateColors();

        private:
            ;K_SET( string, Path );
            ;K_SET( string, Base );
            ;K_SET( string, XAxis );
            ;K_SET( string, YAxis );
            ;K_SET( string, StepOutputGroupName );
            ;K_SET( string, PositionName );
    	    ;K_SET( string, TrackOutputGroupName );
			;K_SET( string, ColorVariable );
			;K_SET( ColorMode, ColorMode );
			;K_SET( ColorPalette, ColorPalette );
            ;K_SET( string, DrawOptions );
            ;K_SET( PlotMode, PlotMode );
            ;K_SET( bool, AxialMirror );
            TMultiGraph* fMultigraph;
            vector<std::pair< TColor, double > > fBaseColors;
            vector<Color_t> fColorVector;

    };

    inline void KSROOTTrackPainter::AddBaseColor(TColor aColor, double aFraction = -1.0 )
    {
    	vismsg( eNormal ) <<"ROOTTrackPainter adding color " <<aColor.GetRed()<<","<<aColor.GetGreen()<<","<<aColor.GetBlue()<<" with fraction "<<aFraction<<eom;
    	fBaseColors.push_back( std::pair< TColor, double > ( aColor, aFraction) );
    }

}

#endif
