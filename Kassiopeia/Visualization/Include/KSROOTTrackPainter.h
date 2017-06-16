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
            ;K_SET( std::string, Path );
            ;K_SET( std::string, Base );
            ;K_SET( std::string, XAxis );
            ;K_SET( std::string, YAxis );
            ;K_SET( std::string, StepOutputGroupName );
            ;K_SET( std::string, PositionName );
    	    ;K_SET( std::string, TrackOutputGroupName );
			;K_SET( std::string, ColorVariable );
			;K_SET( ColorMode, ColorMode );
			;K_SET( ColorPalette, ColorPalette );
            ;K_SET( std::string, DrawOptions );
            ;K_SET( PlotMode, PlotMode );
            ;K_SET( bool, AxialMirror );
            TMultiGraph* fMultigraph;
            std::vector<std::pair< TColor, double > > fBaseColors;
            std::vector<Color_t> fColorVector;

    };

    inline void KSROOTTrackPainter::AddBaseColor(TColor aColor, double aFraction = -1.0 )
    {
    	vismsg( eNormal ) <<"ROOTTrackPainter adding color " <<aColor.GetRed()<<","<<aColor.GetGreen()<<","<<aColor.GetBlue()<<" with fraction "<<aFraction<<eom;
    	fBaseColors.push_back( std::pair< TColor, double > ( aColor, aFraction) );
    }

}

#endif
