#include "KSROOTTrackPainter.h"

#include "KSObject.h"
#include "KSReadFileROOT.h"
#include "KSVisualizationMessage.h"

#include "TGraph.h"

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
            fColor( kRed ),
            fDrawOptions( "" ),
            fPlotMode( ePlotStep ),
            fAxialMirror( false ),
            fMultigraph(),
            fColorVector()
    {
    }
    KSROOTTrackPainter::~KSROOTTrackPainter()
    {
    }

    void KSROOTTrackPainter::Render()
    {
    	CreateColors();
        vector<Color_t>::iterator tColorIterator = fColorVector.begin();

        fMultigraph = new TMultiGraph();

        KRootFile* tRootFile = katrin::CreateOutputRootFile( fBase );
        if( !fPath.empty() )
        {
            tRootFile->AddToPaths( fPath );
        }

        KSReadFileROOT tReader;
        tReader.OpenFile( tRootFile );

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
						TGraph* myGraph;
						if ( fColorMode == eColorTrack || fColorMode == eColorFix || fColorMode == eColorFPDRings )
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
								double tXValue;
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
								double tYValue;
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
									fMultigraph->Add( myGraph );
									tColorIterator++;
								}
								if ( fColorMode == eColorTrack || fColorMode == eColorFix || fColorMode == eColorFPDRings )
								{
									myGraph->SetPoint( myGraph->GetN(), tXValue, tYValue );
								}
							}
						}

						if ( fColorMode == eColorTrack || fColorMode == eColorFix || fColorMode == eColorFPDRings )
						{
							fMultigraph->Add( myGraph );

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
								fMultigraph->Add( myMirroredGraph );
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
						if ( fColorMode == eColorFix || fColorMode == eColorTrack )
						{
							myGraph = new TGraph();
							if ( tColorIterator == fColorVector.end() )
							{
								vismsg( eError ) <<"color vector has to less entries, something is wrong!"<<eom;
							}
							myGraph->SetMarkerColor( *tColorIterator );
							tColorIterator++;
						}

						double tXValue;
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
						double tYValue;
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
						fMultigraph->Add( myGraph );
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
        if( fDisplayEnabled == true )
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

    void KSROOTTrackPainter::CreateColors()
    {
        KRootFile* tRootFile = katrin::CreateOutputRootFile( fBase );
        if( !fPath.empty() )
        {
            tRootFile->AddToPaths( fPath );
        }

        KSReadFileROOT tReader;
        tReader.OpenFile( tRootFile );

        KSReadRunROOT& tRunReader = tReader.GetRun();
        KSReadEventROOT& tEventReader = tReader.GetEvent();
        KSReadTrackROOT& tTrackReader = tReader.GetTrack();
        KSReadStepROOT& tStepReader = tReader.GetStep();

        //getting number of tracks in file
        size_t tNumberOfTracks = 0;
	    for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
	    {
			for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
			{
				for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
				{
					tNumberOfTracks++;
				}
			}
	    }

    	if ( fColorMode == eColorFPDRings )
    	{
    		while ( fColorVector.size() < tNumberOfTracks )
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
    		while ( fColorVector.size() < tNumberOfTracks )
    		{
    			fColorVector.push_back( fColor );
    		}
    	}


        double tColorVariableMax( -1.0 * std::numeric_limits< double >::max());
        double tColorVariableMin( std::numeric_limits< double >::max());

    	double tMinColor = 51; // =^=Purple
    	double tMaxColor = 100; // =^=Red

    	if ( fColorMode == eColorTrack )
    	{
    		if ( tNumberOfTracks == 1 )
    		{
    			fColorVector.push_back( tMaxColor );
    			return;
    		}
        	//get track group and color variable
			KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject( fTrackOutputGroupName );
	        KSDouble& tColorVariable = tTrackGroup.Get< KSDouble >( fColorVariable );

	        //find min and max of color variable
		    for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
		    {
				for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
				{
					for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
					{
						if ( tColorVariable.Value() > tColorVariableMax )
						{
							tColorVariableMax =  tColorVariable.Value();
						}
						if ( tColorVariable.Value() < tColorVariableMin )
						{
							tColorVariableMin =  tColorVariable.Value();
						}
					}
				}
		    }
            vismsg( eNormal ) << "Range of track color variable is from < "<<tColorVariableMin <<" > to < "<<tColorVariableMax<<" >"<<eom;

		    for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
		    {
				for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
				{
					for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
					{
						KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject( fTrackOutputGroupName );
						KSDouble& tColorVariable = tTrackGroup.Get< KSDouble >( fColorVariable );
						double tCurrentColor = tMinColor+((tMaxColor-tMinColor)*(tColorVariable.Value()-tColorVariableMin)/(tColorVariableMax-tColorVariableMin));
						fColorVector.push_back( tCurrentColor );
					}
				}
		    }
    	}


        if ( fColorMode == eColorStep )
        {
        	//get step group and color variable
	        KSReadObjectROOT& tStepGroup = tStepReader.GetObject( fStepOutputGroupName );
	        KSDouble& tColorVariable = tStepGroup.Get< KSDouble >( fColorVariable );

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
								if ( tColorVariable.Value() > tColorVariableMax )
								{
									tColorVariableMax =  tColorVariable.Value();
								}
								if ( tColorVariable.Value() < tColorVariableMin )
								{
									tColorVariableMin =  tColorVariable.Value();
								}
							}
						}
					}
				}
		    }

            vismsg( eNormal ) << "Range of step color variable is from < "<<tColorVariableMin <<" > to < "<<tColorVariableMax<<" >"<<eom;

		    for( tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++ )
		    {
				for( tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex(); tEventReader++ )
				{
					for( tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex(); tTrackReader++ )
					{
						for( tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex(); tStepReader++ )
						{
							KSDouble& tColorVariable = tStepGroup.Get< KSDouble >( fColorVariable );
							double tCurrentColor = tMinColor+((tMaxColor-tMinColor)*(tColorVariable.Value()-tColorVariableMin)/(tColorVariableMax-tColorVariableMin));
							fColorVector.push_back( tCurrentColor );
						}
					}
				}
			}

        }
        tReader.CloseFile();
        delete tRootFile;

    }

    double KSROOTTrackPainter::GetXMin()
    {
        double tMin( std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
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
        return tMin;
    }
    double KSROOTTrackPainter::GetXMax()
    {
        double tMax( -1.0 * std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
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
        return tMax;
    }

    double KSROOTTrackPainter::GetYMin()
    {
        double tMin( std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
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
        return tMin;
    }
    double KSROOTTrackPainter::GetYMax()
    {
        double tMax( -1.0 * std::numeric_limits< double >::max() );
        TList* tGraphList = fMultigraph->GetListOfGraphs();
        for ( int tIndex = 0; tIndex < tGraphList->GetSize(); tIndex++ )
        {
        	TGraph* tGraph = dynamic_cast<TGraph*> ( tGraphList->At( tIndex ) );
        	double* tY = tGraph->GetY();
        	for ( int tIndexArray = 0; tIndexArray < tGraph->GetN(); tIndexArray++ )
        	{
				if ( tY[tIndexArray] < tMax )
				{
					tMax = tY[tIndexArray];
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
