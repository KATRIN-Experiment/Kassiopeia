#include "KSROOTTrackPainter.h"

#include "KSObject.h"
#include "KSReadFileROOT.h"
#include "KSVisualizationMessage.h"

#include "TGraph.h"
#include "TColor.h"
#include "TStyle.h"

#include <limits>

namespace Kassiopeia
{
    KSROOTTrackPainter::KSROOTTrackPainter() :
            fPath( "" ),
            fBase( "" ),
            fXAxis( "z" ),
            fYAxis( "y" ),
            fStepOutputGroupName( "output_step_world" ),
            fPositionName( "position" ),
            fTrackOutputGroupName( "output_track_world" ),
            fColorVariable( "color_variable" ),
            fColorMode( eColorFix ),
            fColorPalette( eColorDefault ),
            fDrawOptions( "" ),
            fPlotMode( ePlotStep ),
            fAxialMirror( false ),
            fMultigraph(),
            fBaseColors(),
            fColorVector()
    {
    }
    KSROOTTrackPainter::~KSROOTTrackPainter()
    {
    }

    void KSROOTTrackPainter::Render()
    {
        fMultigraph = new TMultiGraph();

        KRootFile* tRootFile = KRootFile::CreateOutputRootFile( fBase );
        if( !fPath.empty() )
        {
            tRootFile->AddToPaths( fPath );
        }

        KSReadFileROOT tReader;
        if (! tReader.TryFile( tRootFile ))
        {
            vismsg( eWarning ) << "Could not open file <" << tRootFile->GetName() << ">" << eom;
            return;
        }

        tReader.OpenFile( tRootFile );

        CreateColors( tReader );
        vector<Color_t>::iterator tColorIterator = fColorVector.begin();

        KSReadRunROOT& tRunReader = tReader.GetRun();
        KSReadEventROOT& tEventReader = tReader.GetEvent();
        KSReadTrackROOT& tTrackReader = tReader.GetTrack();
        KSReadStepROOT& tStepReader = tReader.GetStep();

        if ( fPlotMode == ePlotStep )
        {
            KSReadObjectROOT& tStepGroup = tStepReader.GetObject( fStepOutputGroupName );
            KSThreeVector& tPosition = tStepGroup.Get< KSThreeVector >( fPositionName );

            for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
            {
                vismsg( eDebug ) << "Analyzing run <" << tRunReader.GetRunIndex() << "> with events from <" << tRunReader.GetFirstEventIndex() << "> to <"<<tRunReader.GetLastEventIndex()<<">"<< eom;
                for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
                {
                    vismsg( eDebug ) << "Analyzing event <" << tEventReader.GetEventIndex() << "> with tracks from <" << tEventReader.GetFirstTrackIndex() << "> to <"<<tEventReader.GetLastTrackIndex()<<">"<< eom;
                    for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                    {
                        vismsg( eDebug ) << "Analyzing track <" << tTrackReader.GetTrackIndex() << "> with steps from <" << tTrackReader.GetFirstStepIndex() << "> to <"<<tTrackReader.GetLastStepIndex()<<">"<< eom;
                        TGraph* myGraph = NULL;
                        if ( fColorMode == eColorTrack || fColorMode == eColorFix  )
                        {
                            myGraph = new TGraph();
                            if ( tColorIterator == fColorVector.end() )
                            {
                                vismsg( eError ) <<"color vector has to less entries, something is wrong!"<<eom;
                            }
                            myGraph->SetLineColor( *tColorIterator );
                            tColorIterator++;
                        }

                        for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
                        {
                            if( tStepGroup.Valid() )
                            {
                                double tXValue = 0.;
                                if( fXAxis == string( "x" ) || fXAxis == string( "X" ) )
                                {
                                    tXValue = tPosition.Value().X();
                                }
                                if( fXAxis == string( "y" ) || fXAxis == string( "Y" ) )
                                {
                                    tXValue = tPosition.Value().Y();
                                }
                                if( fXAxis == string( "z" ) || fXAxis == string( "Z" ) )
                                {
                                    tXValue = tPosition.Value().Z();
                                }
                                double tYValue = 0.;
                                if( fYAxis == string( "x" ) || fYAxis == string( "X" ) )
                                {
                                    tYValue = tPosition.Value().X();
                                }
                                if( fYAxis == string( "y" ) || fYAxis == string( "Y" ) )
                                {
                                    tYValue = tPosition.Value().Y();
                                }
                                if( fYAxis == string( "z" ) || fYAxis == string( "Z" ) )
                                {
                                    tYValue = tPosition.Value().Z();
                                }
                                if( fYAxis == string( "r" ) || fYAxis == string( "R" ) )
                                {
                                    tYValue = tPosition.Value().Perp();
                                }

                                if ( fColorMode == eColorStep )
                                {
                                    //create one graph for each point (one graph can only have one color)
                                    myGraph = new TGraph();
                                    myGraph->SetPoint( myGraph->GetN(), tXValue, tYValue );
                                    if ( tColorIterator == fColorVector.end() )
                                    {
                                        vismsg( eError ) <<"color vector has to less entries, something is wrong!"<<eom;
                                    }
                                    myGraph->SetMarkerColor( *tColorIterator );
                                    tColorIterator++;
                                    if (myGraph->GetN() > 0)
                                    {
                                        fMultigraph->Add( myGraph );
                                    }
                                }
                                if ( fColorMode == eColorTrack || fColorMode == eColorFix )
                                {
                                    myGraph->SetPoint( myGraph->GetN(), tXValue, tYValue );
                                }
                            }
                        }

                        if ( fColorMode == eColorTrack || fColorMode == eColorFix )
                        {
                            if (myGraph->GetN() > 0)
                            {
                                fMultigraph->Add( myGraph );
                            }

                            //if axial mirror is set, another graph is created with the same points, put y has a changed sign
                            if ( fAxialMirror )
                            {
                                TGraph* myMirroredGraph = new TGraph();
                                myMirroredGraph->SetLineColor( myGraph->GetLineColor() );
                                double tX,tY;
                                for ( int tIndex = 0; tIndex < myGraph->GetN(); tIndex++ )
                                {
                                    myGraph->GetPoint( tIndex, tX, tY );
                                    myMirroredGraph->SetPoint( tIndex, tX, -1.0*tY );
                                }
                                if (myMirroredGraph->GetN() > 0)
                                {
                                    fMultigraph->Add( myMirroredGraph );
                                }
                            }
                        }
                    }
                }
            }
        }

        if ( fPlotMode == ePlotTrack )
        {
            KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject( fTrackOutputGroupName );
            KSThreeVector& tPosition = tTrackGroup.Get< KSThreeVector >( fPositionName );
            for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
            {
                for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
                {

                    for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                    {
                        TGraph* myGraph;
                        myGraph = new TGraph();

                        if ( fColorMode == eColorTrack || fColorMode == eColorFix )
                        {
                            if ( tColorIterator == fColorVector.end() )
                            {
                                vismsg( eError ) <<"color vector has to less entries, something is wrong!"<<eom;
                            }
                            myGraph->SetMarkerColor( *tColorIterator );
                            tColorIterator++;
                        }

                        double tXValue = 0.;
                        if( fXAxis == string( "x" ) || fXAxis == string( "X" ) )
                        {
                            tXValue = tPosition.Value().X();
                        }
                        if( fXAxis == string( "y" ) || fXAxis == string( "Y" ) )
                        {
                            tXValue = tPosition.Value().Y();
                        }
                        if( fXAxis == string( "z" ) || fXAxis == string( "Z" ) )
                        {
                            tXValue = tPosition.Value().Z();
                        }
                        double tYValue  = 0.;
                        if( fYAxis == string( "x" ) || fYAxis == string( "X" ) )
                        {
                            tYValue = tPosition.Value().X();
                        }
                        if( fYAxis == string( "y" ) || fYAxis == string( "Y" ) )
                        {
                            tYValue = tPosition.Value().Y();
                        }
                        if( fYAxis == string( "z" ) || fYAxis == string( "Z" ) )
                        {
                            tYValue = tPosition.Value().Z();
                        }
                        if( fYAxis == string( "r" ) || fYAxis == string( "R" ) )
                        {
                            tYValue = tPosition.Value().Perp();
                        }

                        myGraph->SetPoint( myGraph->GetN(), tXValue, tYValue );
                        if (myGraph->GetN() > 0)
                        {
                            fMultigraph->Add( myGraph );
                        }
                    }

                }
            }
        }

        tReader.CloseFile();
        delete tRootFile;

        return;
    }

    void KSROOTTrackPainter::Display()
    {
        if( (fDisplayEnabled == true) && (fMultigraph->GetListOfGraphs() != nullptr) )
        {
            if ( fPlotMode == ePlotStep )
            {
                if ( fColorMode == eColorStep )
                {
                    fMultigraph->Draw( ( string("P") + fDrawOptions ).c_str() );
                }
                else
                {
                    fMultigraph->Draw( ( string("L") + fDrawOptions ).c_str() );
                }
            }
            if ( fPlotMode == ePlotTrack )
            {
                fMultigraph->Draw( ( string("P") + fDrawOptions ).c_str() );
            }
        }

        return;
    }

    void KSROOTTrackPainter::Write()
    {
        if( fWriteEnabled == true )
        {
            return;
        }
        return;
    }

    void KSROOTTrackPainter::CreateColors( KSReadFileROOT& aReader )
    {
        KSReadRunROOT& tRunReader = aReader.GetRun();
        KSReadEventROOT& tEventReader = aReader.GetEvent();
        KSReadTrackROOT& tTrackReader = aReader.GetTrack();
        KSReadStepROOT& tStepReader = aReader.GetStep();

        //getting number of tracks/steps in file
        size_t tNumberOfTracks = 0;
        size_t tNumberOfSteps = 0;
        for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
        {
            for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
            {
                for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                {
                    tNumberOfTracks++;

                    for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
                    {
                        tNumberOfSteps++;
                    }
                }
            }
        }

        if ( fColorPalette == eColorFPDRings )
        {
            while ( ( fColorVector.size() < tNumberOfTracks && fColorMode == eColorTrack )
                    || ( fColorVector.size() < tNumberOfSteps && fColorMode == eColorStep ) )
            {
                //rainbow scheme
                fColorVector.push_back( kBlack );
                fColorVector.push_back( kViolet + 7 );
                fColorVector.push_back( kBlue + 2 );
                fColorVector.push_back( kAzure + 2 );
                fColorVector.push_back( kAzure + 10 );
                fColorVector.push_back( kTeal + 7 );
                fColorVector.push_back( kGreen + 1);
                fColorVector.push_back( kSpring - 3 );
                fColorVector.push_back( kSpring + 10 );
                fColorVector.push_back( kYellow );
                fColorVector.push_back( kOrange - 3 );
                fColorVector.push_back( kOrange + 7 );
                fColorVector.push_back( kRed );
                fColorVector.push_back( kRed + 2);
            }
        }

        if ( fColorMode == eColorFix )
        {
            Color_t tFixColor(kRed);
            if ( fBaseColors.size() > 0 )
            {
                TColor tTempColor;
                tFixColor = tTempColor.GetColor(fBaseColors.at(0).first.GetRed(),fBaseColors.at(0).first.GetGreen(),fBaseColors.at(0).first.GetBlue());
            }
            while ( fColorVector.size() < tNumberOfTracks )
            {
                fColorVector.push_back( tFixColor );
            }
        }

        if ( fColorPalette == eColorCustom || fColorPalette == eColorDefault )
        {

            int tColorBins=100;
            size_t tNumberBaseColors = fBaseColors.size();

            double tRed[tNumberBaseColors];
            double tGreen[tNumberBaseColors];
            double tBlue[tNumberBaseColors];
            double tFraction[tNumberBaseColors];

            for ( size_t tIndex = 0; tIndex < tNumberBaseColors; tIndex++ )
            {
                tRed[tIndex] = fBaseColors.at( tIndex ).first.GetRed();
                tGreen[tIndex] = fBaseColors.at( tIndex ).first.GetGreen();
                tBlue[tIndex] = fBaseColors.at( tIndex ).first.GetBlue();
                tFraction[tIndex] = fBaseColors.at( tIndex ).second;
                if ( tFraction[tIndex] == -1.0 )
                {
                    tFraction[tIndex] = tIndex / (double)(tNumberBaseColors - 1);
                }
            }

            int tMinColor = TColor::CreateGradientColorTable(tNumberBaseColors,tFraction,tRed,tGreen,tBlue,tColorBins);
            int tMaxColor = tMinColor + tColorBins - 1;

            if ( fColorPalette == eColorDefault )
            {
                tMinColor = 51; //purple
                tMaxColor = 100; //red
            }

    //        int tPalette[tColorBins];
    //        for ( int i = 0; i<tColorBins; i++)
    //        {
    //          tPalette[i] = tMinColor + i;
    //        }
    //        gStyle->SetPalette( tColorBins, tPalette );


            double tColorVariableMax( -1.0 * std::numeric_limits< double >::max());
            double tColorVariableMin( std::numeric_limits< double >::max());

            if ( fColorMode == eColorTrack )
            {
                if ( tNumberOfTracks == 1 )
                {
                    fColorVector.push_back( tMaxColor );
                    return;
                }
                //get track group and color variable
                KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject( fTrackOutputGroupName );

                //find min and max of color variable
                for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
                {
                    for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
                    {
                        for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                        {
                            double tColorVariable = 0.;
                            if ( tTrackGroup.Exists< KSDouble >( fColorVariable ) )
                            {
                                tColorVariable = tTrackGroup.Get< KSDouble >( fColorVariable ).Value();
                            }
                            else if ( tTrackGroup.Exists< KSInt >( fColorVariable ) )
                            {
                                tColorVariable = tTrackGroup.Get< KSInt >( fColorVariable ).Value();
                            }
                            else if ( tTrackGroup.Exists< KSUInt >( fColorVariable ) )
                            {
                                tColorVariable = tTrackGroup.Get< KSUInt >( fColorVariable ).Value();
                            }
                            else
                            {
                                vismsg( eError ) <<"Color variable is of unsupported type"<<eom;
                            }


                            if ( tColorVariable > tColorVariableMax )
                            {
                                tColorVariableMax =  tColorVariable;
                            }
                            if ( tColorVariable < tColorVariableMin )
                            {
                                tColorVariableMin =  tColorVariable;
                            }
                        }
                    }
                }
                vismsg( eInfo ) << "Range of track color variable is from < "<<tColorVariableMin <<" > to < "<<tColorVariableMax<<" >"<<eom;

                for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
                {
                    for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
                    {
                        for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                        {
                            double tColorVariable = 0.;
                            if ( tTrackGroup.Exists< KSDouble >( fColorVariable ) )
                            {
                                tColorVariable = tTrackGroup.Get< KSDouble >( fColorVariable ).Value();
                            }
                            else if ( tTrackGroup.Exists< KSInt >( fColorVariable ) )
                            {
                                tColorVariable = tTrackGroup.Get< KSInt >( fColorVariable ).Value();
                            }
                            else if ( tTrackGroup.Exists< KSUInt >( fColorVariable ) )
                            {
                                tColorVariable = tTrackGroup.Get< KSUInt >( fColorVariable ).Value();
                            }
                            else
                            {
                                vismsg( eError ) <<"Color variable is of unsupported type"<<eom;
                            }
                            double tCurrentColor = tMinColor+((tMaxColor-tMinColor)*(tColorVariable-tColorVariableMin)/(tColorVariableMax-tColorVariableMin));
                            fColorVector.push_back( tCurrentColor );
                        }
                    }
                }
            }

            if ( fColorMode == eColorStep )
            {
                //get step group and color variable
                KSReadObjectROOT& tStepGroup = tStepReader.GetObject( fStepOutputGroupName );

                //find min and max of color variable
                for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
                {
                    for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
                    {
                        for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                        {
                            for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
                            {
                                if ( tStepGroup.Valid() )
                                {
                                    double tColorVariable  = 0.;
                                    if ( tStepGroup.Exists< KSDouble >( fColorVariable ) )
                                    {
                                        tColorVariable = tStepGroup.Get< KSDouble >( fColorVariable ).Value();
                                    }
                                    else if ( tStepGroup.Exists< KSInt >( fColorVariable ) )
                                    {
                                        tColorVariable = tStepGroup.Get< KSInt >( fColorVariable ).Value();
                                    }
                                    else if ( tStepGroup.Exists< KSUInt >( fColorVariable ) )
                                    {
                                        tColorVariable = tStepGroup.Get< KSUInt >( fColorVariable ).Value();
                                    }
                                    else
                                    {
                                        vismsg( eError ) <<"Color variable is of unsupported type"<<eom;
                                    }

                                    if ( tColorVariable > tColorVariableMax )
                                    {
                                        tColorVariableMax =  tColorVariable;
                                    }
                                    if ( tColorVariable < tColorVariableMin )
                                    {
                                        tColorVariableMin =  tColorVariable;
                                    }
                                }
                            }
                        }
                    }
                }

                vismsg( eInfo ) << "Range of step color variable is from < "<<tColorVariableMin <<" > to < "<<tColorVariableMax<<" >"<<eom;

                for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
                {
                    for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
                    {
                        for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
                        {
                            for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
                            {
                                if ( tStepGroup.Valid() )
                                {
                                    double tColorVariable = 0.;
                                    if ( tStepGroup.Exists< KSDouble >( fColorVariable ) )
                                    {
                                        tColorVariable = tStepGroup.Get< KSDouble >( fColorVariable ).Value();
                                    }
                                    else if ( tStepGroup.Exists< KSInt >( fColorVariable ) )
                                    {
                                        tColorVariable = tStepGroup.Get< KSInt >( fColorVariable ).Value();
                                    }
                                    else if ( tStepGroup.Exists< KSUInt >( fColorVariable ) )
                                    {
                                        tColorVariable = tStepGroup.Get< KSUInt >( fColorVariable ).Value();
                                    }
                                    else
                                    {
                                        vismsg( eError ) <<"Color variable is of unsupported type"<<eom;
                                    }
                                    double tCurrentColor = tMinColor+((tMaxColor-tMinColor)*(tColorVariable-tColorVariableMin)/(tColorVariableMax-tColorVariableMin));
                                    fColorVector.push_back( tCurrentColor );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double KSROOTTrackPainter::GetXMin()
    {
        double tMin( std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
        if (tGraphList != nullptr)
        {
            for ( int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++ )
            {
                TGraph* tGraph = dynamic_cast<TGraph*> ( tGraphList->At( tIndex ) );
                double* tX = tGraph->GetX();
                for ( int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++ )
                {
                    if ( tX[tIndexArray] < tMin )
                    {
                        tMin = tX[tIndexArray];
                    }
                }
            }
        }
        return tMin;
    }
    double KSROOTTrackPainter::GetXMax()
    {
        double tMax( -1.0 * std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
        if (tGraphList != nullptr)
        {
            for ( int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++ )
            {
                TGraph* tGraph = dynamic_cast<TGraph*> ( tGraphList->At( tIndex ) );
                double* tX = tGraph->GetX();
                for ( int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++ )
                {
                    if ( tX[tIndexArray] > tMax )
                    {
                        tMax = tX[tIndexArray];
                    }
                }
            }
        }
        return tMax;
    }

    double KSROOTTrackPainter::GetYMin()
    {
        double tMin( std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
        if (tGraphList != nullptr)
        {
            for ( int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++ )
            {
                TGraph* tGraph = dynamic_cast<TGraph*> ( tGraphList->At( tIndex ) );
                double* tY = tGraph->GetY();
                for ( int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++ )
                {
                    if ( tY[tIndexArray] < tMin )
                    {
                        tMin = tY[tIndexArray];
                    }
                }
            }
        }
        return tMin;
    }
    double KSROOTTrackPainter::GetYMax()
    {
        double tMax( -1.0 * std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
        if (tGraphList != nullptr)
        {
            for ( int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++ )
            {
                TGraph* tGraph = dynamic_cast<TGraph*> ( tGraphList->At( tIndex ) );
                double* tY = tGraph->GetY();
                for ( int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++ )
                {
                    if ( tY[tIndexArray] > tMax )
                    {
                        tMax = tY[tIndexArray];
                    }
                }
            }
        }
        return tMax;
    }

    std::string KSROOTTrackPainter::GetXAxisLabel()
    {
        return fXAxis;
    }

    std::string KSROOTTrackPainter::GetYAxisLabel()
    {
        return fYAxis;
    }

}
