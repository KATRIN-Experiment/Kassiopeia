#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KTagProcessor.hh"
#include "KElementProcessor.hh"

#include "KSToolbox.h"
#include "KSStep.h"
#include "KSRootTrajectory.h"
#include "KSMainMessage.h"

#include "KSParticle.h"
#include "KSParticleFactory.h"

#include "KSMagneticFieldDipole.h"
#include "KSRootMagneticField.h"

#include "KSElectricFieldConstant.h"
#include "KSRootElectricField.h"

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
    tInputFile.AddToBases( "TestTrajectory.xml" );
    tInputFile.AddToPaths( string( CONFIG_DEFAULT_DIR ) + string( "/Validation" ) );
    tXMLTokenizer.ProcessFile( &tInputFile );

    // read in command line parameters
    if( anArgc <= 1 )
    {
        mainmsg( eWarning ) << ret;
        mainmsg << "usage:" << ret;
        mainmsg << "TestTrajectory <count=UNSIGNED INT> <command=PARENT.FIELD:CHILD>" << ret;
        mainmsg << ret;
        mainmsg << "available trajectory components found in " << CONFIG_DEFAULT_DIR << "/Validation/TestTrajectory.xml" << ret;
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

            tParent = KSToolbox::GetInstance()->GetObject( tCommandParent );
            tChild = KSToolbox::GetInstance()->GetObject( tCommandChild );
            tCommand = tParent->CreateCommand( tCommandField );
            tCommand->BindParent( tParent );
            tCommand->BindChild( tChild );
            tCommands.push_back( tCommand );
        }
    }

    // initialize root
    TApplication tApplication( "Test Trajectory", 0, NULL );

    TCanvas tParticlePositionCanvas( "particle_position_canvas", "Particle Position" );
    TPolyMarker3D tParticlePositionGraph;

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

    // initialize root trajectory
    KSRootTrajectory* tRootTrajectory = KSToolbox::GetInstance()->GetObjectAs< KSRootTrajectory >( "root_trajectory" );
    tRootTrajectory->SetStep( &tStep );

    tRootTrajectory->Initialize();
    for( tCommandIt = tCommands.begin(); tCommandIt != tCommands.end(); tCommandIt++ )
    {
        tCommand = *tCommandIt;
        tCommand->Initialize();
    }

    tRootTrajectory->Activate();
    for( tCommandIt = tCommands.begin(); tCommandIt != tCommands.end(); tCommandIt++ )
    {
        tCommand = *tCommandIt;
        tCommand->Activate();
    }

    // initialize fields
    KSMagneticFieldDipole tMagneticFieldDipolePlus;
    tMagneticFieldDipolePlus.SetLocation( KThreeVector( 0., 0., 1. ) );
    tMagneticFieldDipolePlus.SetMoment( KThreeVector( 0., 0., 15000. ) );

    KSMagneticFieldDipole tMagneticFieldDipoleMinus;
    tMagneticFieldDipoleMinus.SetLocation( KThreeVector( 0., 0., -1. ) );
    tMagneticFieldDipoleMinus.SetMoment( KThreeVector( 0., 0., 15000. ) );

    KSRootMagneticField tRootMagneticField;
    tRootMagneticField.AddMagneticField( &tMagneticFieldDipolePlus );
    tRootMagneticField.AddMagneticField( &tMagneticFieldDipoleMinus );

    KSElectricFieldConstant tElectricFieldConstant;
    tElectricFieldConstant.SetField( KThreeVector( 0., 0., 0. ) );

    KSRootElectricField tRootElectricField;
    tRootElectricField.AddElectricField( &tElectricFieldConstant );

    // initialize primary
    KSParticleFactory::GetInstance()->SetMagneticField( &tRootMagneticField );
    KSParticleFactory::GetInstance()->SetElectricField( &tRootElectricField );
    KSParticle* tPrimary = KSParticleFactory::GetInstance()->Create( 11 );
    tPrimary->SetTime( 0. );
    tPrimary->SetLength( 0. );
    tPrimary->SetPosition( 1., 0., 0. );
    tPrimary->SetKineticEnergy_eV( 5. );
    tPrimary->SetPolarAngleToZ( 1. );
    tPrimary->SetAzimuthalAngleToX( 0. );

    // calculate steps
    mainmsg( eNormal ) << "starting calculation..." << eom;

    tStep.InitialParticle() = *tPrimary;
    tStep.IntegrationParticle() = *tPrimary;
    tStep.InterpolationParticle() = *tPrimary;
    tStep.InteractionParticle() = *tPrimary;
    tStep.FinalParticle() = *tPrimary;
    for( unsigned int tIndex = 0; tIndex < tCount; tIndex++ )
    {
        tRootTrajectory->ExecuteIntegration( 1. );

        tParticlePositionGraph.SetPoint( tParticleXPositionGraph.GetN(), tStep.IntegrationParticle().GetX(), tStep.IntegrationParticle().GetY(), tStep.IntegrationParticle().GetZ() );
        tParticleXPositionGraph.SetPoint( tParticleXPositionGraph.GetN(), tStep.IntegrationParticle().GetTime(), tStep.IntegrationParticle().GetX() );
        tParticleYPositionGraph.SetPoint( tParticleYPositionGraph.GetN(), tStep.IntegrationParticle().GetTime(), tStep.IntegrationParticle().GetY() );
        tParticleZPositionGraph.SetPoint( tParticleZPositionGraph.GetN(), tStep.IntegrationParticle().GetTime(), tStep.IntegrationParticle().GetZ() );
        tParticleLengthGraph.SetPoint( tParticleLengthGraph.GetN(), tStep.IntegrationParticle().GetTime(), tStep.IntegrationParticle().GetLength() );
        tParticleKineticEnergyGraph.SetPoint( tParticleKineticEnergyGraph.GetN(), tStep.IntegrationParticle().GetTime(), tStep.IntegrationParticle().GetKineticEnergy_eV() );
        tParticleTotalEnergyGraph.SetPoint( tParticleTotalEnergyGraph.GetN(), tStep.IntegrationParticle().GetTime(), tStep.IntegrationParticle().GetKineticEnergy_eV() - tStep.IntegrationParticle().GetElectricPotential() );

        tStep.InitialParticle() = tStep.IntegrationParticle();

        if( tIndex % 100000 == 0 )
        {
            mainmsg( eNormal ) << "  calculated " << tIndex << " steps" << reom;
        }

    }
    mainmsg( eNormal ) << "  calculated " << tCount << " steps" << eom;

    mainmsg( eNormal ) << "...finished calculation" << eom;

    // show plots
    tParticlePositionCanvas.cd( 0 );
    tParticlePositionGraph.SetMarkerColor( kRed );
    tParticlePositionGraph.SetMarkerStyle( 20 );
    tParticlePositionGraph.SetMarkerSize( 0.5 );
    tParticlePositionGraph.Draw( "AP" );

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
    tRootTrajectory->Deactivate();

    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        tCommand->Deinitialize();
    }
    tRootTrajectory->Deinitialize();

    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        delete tCommand;
    }

    KSToolbox::DeleteInstance();

    return 0;
}
