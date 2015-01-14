#ifndef _Kassiopeia_KSROOTTrackPainter_h_
#define _Kassiopeia_KSROOTTrackPainter_h_

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KField.h"

#include "TMultiGraph.h"

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

            typedef enum
            {
                eColorFix, eColorStep, eColorTrack, eColorFPDRings
            } ColorMode;

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
            ;K_SET( int, Color );
            ;K_SET( string, DrawOptions );
            ;K_SET( PlotMode, PlotMode );
            ;K_SET( bool, AxialMirror );
            TMultiGraph* fMultigraph;
            vector<Color_t> fColorVector;

    };

}

#endif
