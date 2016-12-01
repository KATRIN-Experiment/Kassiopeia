#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KTagProcessor.hh"
#include "KElementProcessor.hh"

#include "KToolbox.h"
#include "KSStep.h"
#include "KSRootSpaceInteraction.h"
#include "KSMainMessage.h"

#include "KSParticle.h"
#include "KSParticleFactory.h"

#include "KSMagneticFieldConstant.h"
#include "KSRootMagneticField.h"

#include "KSElectricFieldConstant.h"
#include "KSRootElectricField.h"

#include "KSTrajTrajectoryExact.h"
#include "KSTrajIntegratorRK8.h"
#include "KSTrajInterpolatorFast.h"
#include "KSTrajTermPropagation.h"
#include "KSTrajControlCyclotron.h"
#include "KSRootTrajectory.h"

#include "KConst.h"

#include "TApplication.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPolyMarker3D.h"
#include "TGraph.h"
#include "TAxis.h"

#include <sstream>
using std::stringstream;

using namespace Kassiopeia;
using namespace katrin;

int main( int anArgc, char** anArgv )
{
    // read in xml file
    KXMLTokenizer tXMLTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter( &tXMLTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tIncludeProcessor.InsertAfter( &tFormulaProcessor );
    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tTagProcessor.InsertAfter( &tConditionProcessor );
    tElementProcessor.InsertAfter( &tTagProcessor );

    KTextFile tInputFile;
    tInputFile.AddToBases( "TestSpaceInteraction.xml" );
    tInputFile.AddToPaths( string( CONFIG_DEFAULT_DIR ) + string( "/Validation" ) );
    tXMLTokenizer.ProcessFile( &tInputFile );

    // read in command line parameters
    if( anArgc <= 1 )
    {
        mainmsg( eWarning ) << ret;
        mainmsg << "usage:" << ret;
        mainmsg << "TestSpaceInteraction <count=UNSIGNED INT> <command=PARENT.FIELD:CHILD>" << ret;
        mainmsg << ret;
        mainmsg << "available trajectory components found in " << CONFIG_DEFAULT_DIR << "/Validation/TestSpaceInteraction.xml" << ret;
        mainmsg << "link objects there to the root trajectory using the command interface above to test them" << ret;
        mainmsg << eom;
        return -1;
    }

    unsigned int tCount;
    KSObject* tParent;
    KSObject* tChild;
    KSCommand* tCommand;
    vector< KSCommand* > tCommands;
    vector< KSCommand* >::iterator tCommandIt;
    vector< KSCommand* >::reverse_iterator tCommandRIt;
    size_t tEqualsPos;
    size_t tPeriodPos;
    size_t tColonPos;
    string tArgument;
    string tArgumentLabel;
    string tArgumentValue;
    string tCommandParent;
    string tCommandField;
    string tCommandChild;
    stringstream tArgumentConverter;
    for( int tArgumentIndex = 1; tArgumentIndex < anArgc; tArgumentIndex++ )
    {
        tArgument.assign( anArgv[ tArgumentIndex ] );

        tEqualsPos = tArgument.find( '=' );
        if( tEqualsPos == string::npos )
        {
            mainmsg( eWarning ) << ret;
            mainmsg << "malformed argument <" << tArgument << ">" << ret;
            mainmsg << "arguments must be of form <NAME=VALUE>" << ret;
            mainmsg << eom;
            return -1;
        }

        tArgumentLabel = tArgument.substr( 0, tEqualsPos );
        tArgumentValue = tArgument.substr( tEqualsPos + 1 );

        if( tArgumentLabel == "count" )
        {
            tArgumentConverter.clear();
            tArgumentConverter.str( tArgumentValue );
            tArgumentConverter >> tCount;
            continue;
        }

        if( tArgumentLabel == "command" )
        {
            tPeriodPos = tArgumentValue.find( '.' );
            tColonPos = tArgumentValue.find( ':' );

            if( (tPeriodPos == string::npos) || (tColonPos == string::npos) || (tColonPos < tPeriodPos) )
            {
                mainmsg( eWarning ) << ret;
                mainmsg << "malformed argument <" << tArgument << ">" << ret;
                mainmsg << "command arguments must be of form <command=PARENT.FIELD:CHILD>" << ret;
                mainmsg << eom;
                return -1;
            }

            tCommandParent = tArgumentValue.substr( 0, tPeriodPos );
            tCommandField = tArgumentValue.substr( tPeriodPos + 1, tColonPos - tPeriodPos - 1 );
            tCommandChild = tArgumentValue.substr( tColonPos + 1 );

            tParent = KToolbox::GetInstance().Get( tCommandParent );
            tChild = KToolbox::GetInstance().Get( tCommandChild );
            tCommand = tParent->CreateCommand( tCommandField );
            tCommand->BindParent( tParent );
            tCommand->BindChild( tChild );
            tCommands.push_back( tCommand );
        }
    }

    // initialize root
    TApplication tApplication( "Test Space Interaction", 0, NULL );

    TCanvas tParticleXPositionCanvas( "particle_x_position_canvas", "Particle X Position" );
    TGraph tParticleXPositionGraph;

    TCanvas tParticleYPositionCanvas( "particle_y_position_canvas", "Particle Y Position" );
    TGraph tParticleYPositionGraph;

    TCanvas tParticleZPositionCanvas( "particle_z_position_canvas", "Particle Z Position" );
    TGraph tParticleZPositionGraph;

    TCanvas tParticleLengthCanvas( "particle_length_canvas", "Particle Length" );
    TGraph tParticleLengthGraph;

    TCanvas tParticleKineticEnergyCanvas( "particle_kinetic_energy_canvas", "Particle Kinetic Energy" );
    TGraph tParticleKineticEnergyGraph;

    TCanvas tParticleTotalEnergyCanvas( "particle_total_energy_canvas", "Particle Total Energy" );
    TGraph tParticleTotalEnergyGraph;

    // create simulation object
    KSStep tStep;

    // initialize fields
    KSMagneticFieldConstant tMagneticFieldConstant;
    tMagneticFieldConstant.SetField( KThreeVector( 0., 0., 5. ) );

    KSRootMagneticField tRootMagneticField;
    tRootMagneticField.AddMagneticField( &tMagneticFieldConstant );
    KSParticleFactory::GetInstance().SetMagneticField( &tRootMagneticField );

    KSElectricFieldConstant tElectricFieldConstant;
    tElectricFieldConstant.SetField( KThreeVector( 0., 0., -1000. ) );

    KSRootElectricField tRootElectricField;
    tRootElectricField.AddElectricField( &tElectricFieldConstant );
    KSParticleFactory::GetInstance().SetElectricField( &tRootElectricField );

    // initialize trajectories
    KSTrajTrajectoryExact tTrajectoryExact;
    KSTrajIntegratorRK8 tIntegratorRK8;
    KSTrajInterpolatorFast tInterpolatorFast;
    KSTrajTermPropagation tTermPropagation;
    KSTrajControlCyclotron tControlCyclotron;
    tControlCyclotron.SetFraction( 1. / 32. );
    tTrajectoryExact.SetIntegrator( &tIntegratorRK8 );
    tTrajectoryExact.SetInterpolator( &tInterpolatorFast );
    tTrajectoryExact.AddTerm( &tTermPropagation );
    tTrajectoryExact.AddControl( &tControlCyclotron );

    KSRootTrajectory tRootTrajectory;
    tRootTrajectory.SetTrajectory( &tTrajectoryExact );
    tRootTrajectory.SetStep( &tStep );

    // initialize space interaction
    KSRootSpaceInteraction* tRootSpaceInteraction = KToolbox::GetInstance().Get< KSRootSpaceInteraction >( "root_space_interaction" );
    tRootSpaceInteraction->SetStep( &tStep );
    tRootSpaceInteraction->SetRootTrajectory( &tRootTrajectory );

    tRootSpaceInteraction->Initialize();
    for( tCommandIt = tCommands.begin(); tCommandIt != tCommands.end(); tCommandIt++ )
    {
        tCommand = *tCommandIt;
        tCommand->Initialize();
    }

    tRootSpaceInteraction->Activate();
    for( tCommandIt = tCommands.begin(); tCommandIt != tCommands.end(); tCommandIt++ )
    {
        tCommand = *tCommandIt;
        tCommand->Activate();
    }

    // initialize primary
    KSParticleFactory::GetInstance().SetMagneticField( &tRootMagneticField );
    KSParticleFactory::GetInstance().SetElectricField( &tRootElectricField );
    KSParticle* tPrimary = KSParticleFactory::GetInstance().Create( 11 );
    tPrimary->SetTime( 0. );
    tPrimary->SetLength( 0. );
    tPrimary->SetPosition( 0., 0., 0. );
    tPrimary->SetKineticEnergy_eV( 1000. );
    tPrimary->SetPolarAngleToZ( 90. );
    tPrimary->SetAzimuthalAngleToX( 0. );

    mainmsg( eNormal ) << "starting track calculation..." << eom;
    unsigned int tStepIndex;
    for( unsigned int tTrackIndex = 0; tTrackIndex < tCount; tTrackIndex++ )
    {
        tStep.InitialParticle() = *tPrimary;
        tStep.IntegrationParticle() = *tPrimary;
        tStep.InterpolationParticle() = *tPrimary;
        tStep.InteractionParticle() = *tPrimary;
        tStep.FinalParticle() = *tPrimary;

        mainmsg( eNormal ) << "  starting step calculation..." << eom;

        tStepIndex = 0;
        while( true )
        {
            tRootTrajectory.ExecuteIntegration( 1. );
            tRootSpaceInteraction->ExecuteInteraction();

            tStep.InitialParticle() = tStep.GetInteractionParticle();

            if( tStepIndex % 100000 == 0 )
            {
                mainmsg( eNormal ) << "   calculated " << tStepIndex << " steps" << reom;
            }
            tStepIndex++;

            if( tStep.GetSpaceInteractionFlag() == true )
            {
                break;
            }
        }
        mainmsg( eNormal ) << "    calculated " << tStepIndex << " steps" << eom;
        mainmsg( eNormal ) << "  ...finished step calculation" << eom;

        tParticleXPositionGraph.SetPoint( tParticleXPositionGraph.GetN(), tStep.InterpolationParticle().GetTime(), tStep.InterpolationParticle().GetX() );
        tParticleYPositionGraph.SetPoint( tParticleYPositionGraph.GetN(), tStep.InterpolationParticle().GetTime(), tStep.InterpolationParticle().GetY() );
        tParticleZPositionGraph.SetPoint( tParticleZPositionGraph.GetN(), tStep.InterpolationParticle().GetTime(), tStep.InterpolationParticle().GetZ() );
        tParticleLengthGraph.SetPoint( tParticleLengthGraph.GetN(), tStep.InterpolationParticle().GetTime(), tStep.InterpolationParticle().GetLength() );
        tParticleKineticEnergyGraph.SetPoint( tParticleKineticEnergyGraph.GetN(), tStep.InterpolationParticle().GetTime(), tStep.InterpolationParticle().GetKineticEnergy_eV() );
        tParticleTotalEnergyGraph.SetPoint( tParticleTotalEnergyGraph.GetN(), tStep.InterpolationParticle().GetTime(), tStep.InterpolationParticle().GetKineticEnergy_eV() - tStep.InterpolationParticle().GetElectricPotential() );


        mainmsg( eNormal ) << "  calculated " << tTrackIndex << " tracks" << eom;

    }
    mainmsg( eNormal ) << "...finished track calculation" << eom;


    // show plots
    tParticleXPositionCanvas.cd( 0 );
    tParticleXPositionGraph.SetMarkerColor( kRed );
    tParticleXPositionGraph.SetMarkerStyle( 20 );
    tParticleXPositionGraph.SetMarkerSize( 0.5 );
    tParticleXPositionGraph.SetLineWidth( 1 );
    tParticleXPositionGraph.SetTitle( "Particle X Position vs Time" );
    tParticleXPositionGraph.GetXaxis()->SetTitle( "T [sec]" );
    tParticleXPositionGraph.GetYaxis()->SetTitle( "X [m]" );
    tParticleXPositionGraph.Draw( "AP" );

    tParticleYPositionCanvas.cd( 0 );
    tParticleYPositionGraph.SetMarkerColor( kRed );
    tParticleYPositionGraph.SetMarkerStyle( 20 );
    tParticleYPositionGraph.SetMarkerSize( 0.5 );
    tParticleYPositionGraph.SetLineWidth( 1 );
    tParticleYPositionGraph.SetTitle( "Particle Y Position vs Time" );
    tParticleYPositionGraph.GetXaxis()->SetTitle( "T [sec]" );
    tParticleYPositionGraph.GetYaxis()->SetTitle( "Y [m]" );
    tParticleYPositionGraph.Draw( "AP" );

    tParticleZPositionCanvas.cd( 0 );
    tParticleZPositionGraph.SetMarkerColor( kRed );
    tParticleZPositionGraph.SetMarkerStyle( 20 );
    tParticleZPositionGraph.SetMarkerSize( 0.5 );
    tParticleZPositionGraph.SetLineWidth( 1 );
    tParticleZPositionGraph.SetTitle( "Particle Z Position vs Time" );
    tParticleZPositionGraph.GetXaxis()->SetTitle( "T [sec]" );
    tParticleZPositionGraph.GetYaxis()->SetTitle( "Z [m]" );
    tParticleZPositionGraph.Draw( "AP" );

    tParticleLengthCanvas.cd( 0 );
    tParticleLengthGraph.SetMarkerColor( kRed );
    tParticleLengthGraph.SetMarkerStyle( 20 );
    tParticleLengthGraph.SetMarkerSize( 0.5 );
    tParticleLengthGraph.SetLineWidth( 1 );
    tParticleLengthGraph.SetTitle( "Particle Length vs Time" );
    tParticleLengthGraph.GetXaxis()->SetTitle( "T [sec]" );
    tParticleLengthGraph.GetYaxis()->SetTitle( "Length [m]" );
    tParticleLengthGraph.Draw( "AP" );

    tParticleKineticEnergyCanvas.cd( 0 );
    tParticleKineticEnergyGraph.SetMarkerColor( kRed );
    tParticleKineticEnergyGraph.SetMarkerStyle( 20 );
    tParticleKineticEnergyGraph.SetMarkerSize( 0.5 );
    tParticleKineticEnergyGraph.SetLineWidth( 1 );
    tParticleKineticEnergyGraph.SetTitle( "Particle Kinetic Energy vs Time" );
    tParticleKineticEnergyGraph.GetXaxis()->SetTitle( "T [sec]" );
    tParticleKineticEnergyGraph.GetYaxis()->SetTitle( "K [eV]" );
    tParticleKineticEnergyGraph.Draw( "AP" );

    tParticleTotalEnergyCanvas.cd( 0 );
    tParticleTotalEnergyGraph.SetMarkerColor( kRed );
    tParticleTotalEnergyGraph.SetMarkerStyle( 20 );
    tParticleTotalEnergyGraph.SetMarkerSize( 0.5 );
    tParticleTotalEnergyGraph.SetLineWidth( 1 );
    tParticleTotalEnergyGraph.SetTitle( "Particle Total Energy vs Time" );
    tParticleTotalEnergyGraph.GetXaxis()->SetTitle( "T [sec]" );
    tParticleTotalEnergyGraph.GetYaxis()->SetTitle( "E [eV]" );
    tParticleTotalEnergyGraph.Draw( "AP" );

    tApplication.Run();

    // deinitialize kassiopeia
    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        tCommand->Deactivate();
    }
    tRootSpaceInteraction->Deactivate();

    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        tCommand->Deinitialize();
    }
    tRootSpaceInteraction->Deinitialize();

    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        delete tCommand;
    }


    return 0;
}
