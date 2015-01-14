#ifndef Kassiopeia_KSROOTTrackPainterBuilder_h_
#define Kassiopeia_KSROOTTrackPainterBuilder_h_

#include "KComplexElement.hh"
#include "KSROOTTrackPainter.h"
#include <stdlib.h>

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSROOTTrackPainter > KSROOTTrackPainterBuilder;

    template< >
    inline bool KSROOTTrackPainterBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "base" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetBase );
            return true;
        }
        if( aContainer->GetName() == "path" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetPath );
            return true;
        }
        if( aContainer->GetName() == "x_axis" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetXAxis );
            return true;
        }
        if( aContainer->GetName() == "y_axis" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetYAxis );
            return true;
        }
        if( aContainer->GetName() == "step_output_group_name" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetStepOutputGroupName );
            return true;
        }
        if( aContainer->GetName() == "position_name" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetPositionName );
            return true;
        }
        if( aContainer->GetName() == "track_output_group_name" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetTrackOutputGroupName );
            return true;
        }
        if( aContainer->GetName() == "color_variable" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetColorVariable );
            return true;
        }
        if( aContainer->GetName() == "color_mode" )
        {
            if( aContainer->AsReference< string >() == string( "fix" ) )
            {
                fObject->SetColorMode( KSROOTTrackPainter::eColorFix );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "step" ) )
            {
                fObject->SetColorMode( KSROOTTrackPainter::eColorStep );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "track" ) )
            {
                fObject->SetColorMode( KSROOTTrackPainter::eColorTrack );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "fpd_rings" ) )
            {
                fObject->SetColorMode( KSROOTTrackPainter::eColorFPDRings );
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "color" )
        {
			if( aContainer->AsReference< string >() == string( "kWhite" ) )
			{
				fObject->SetColor( kWhite );
				return true;
			}
			if( aContainer->AsReference< string >() == string( "kGray" ) )
			{
				fObject->SetColor( kGray );
				return true;
			}
            if( aContainer->AsReference< string >() == string( "kBlack" ) )
            {
                fObject->SetColor( kBlack );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kRed" ) )
            {
                fObject->SetColor( kRed );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kGreen" ) )
            {
                fObject->SetColor( kGreen );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kBlue" ) )
            {
                fObject->SetColor( kBlue );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kYellow" ) )
            {
                fObject->SetColor( kYellow );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kMagenta" ) )
            {
                fObject->SetColor( kMagenta );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kCyan" ) )
            {
                fObject->SetColor( kCyan );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kOrange" ) )
            {
                fObject->SetColor( kOrange );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kSpring" ) )
            {
                fObject->SetColor( kSpring );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kTeal" ) )
            {
                fObject->SetColor( kTeal );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kAzure" ) )
            {
                fObject->SetColor( kAzure );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kViolet" ) )
            {
                fObject->SetColor( kViolet );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "kPink" ) )
            {
                fObject->SetColor( kPink );
                return true;
            }
            int tColor = std::strtol( aContainer->AsReference< string >().c_str(), 0 , 0 );
            fObject->SetColor ( tColor );
            return true;
        }
        if( aContainer->GetName() == "draw_options" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetDrawOptions );
            return true;
        }
        if( aContainer->GetName() == "plot_mode" )
        {
            if( aContainer->AsReference< string >() == string( "step" ) )
            {
                fObject->SetPlotMode( KSROOTTrackPainter::ePlotStep );
                return true;
            }
            if( aContainer->AsReference< string >() == string( "track" ) )
            {
                fObject->SetPlotMode( KSROOTTrackPainter::ePlotTrack );
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "axial_mirror" )
        {
            aContainer->CopyTo( fObject, &KSROOTTrackPainter::SetAxialMirror );
            return true;
        }
        return false;
    }

}

#endif
