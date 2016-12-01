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
#include "KSEvent.h"
#include "KSRootGenerator.h"
#include "KSMainMessage.h"

#include "KConst.h"

#include "TApplication.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH3D.h"

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
    tInputFile.AddToBases( "TestGenerator.xml" );
    tInputFile.AddToPaths( string( CONFIG_DEFAULT_DIR ) + string( "/Validation" ) );
    tXMLTokenizer.ProcessFile( &tInputFile );

    // read in command line parameters
    if( anArgc <= 1 )
    {
        mainmsg( eWarning ) << ret;
        mainmsg << "usage:" << ret;
        mainmsg << "TestGenerator <count=UNSIGNED INT> <command=PARENT.FIELD:CHILD>" << ret;
        mainmsg << ret;
        mainmsg << "available generator components found in " << CONFIG_DEFAULT_DIR << "/Validation/TestGenerator.xml" << ret;
        mainmsg << "link objects there to the root generator using the command interface above to test them" << ret;
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

    // initialize kassiopeia
    KSRootGenerator* tRootGenerator = KToolbox::GetInstance().Get< KSRootGenerator >( "root_generator" );

    tRootGenerator->Initialize();
    for( tCommandIt = tCommands.begin(); tCommandIt != tCommands.end(); tCommandIt++ )
    {
        tCommand = *tCommandIt;
        tCommand->Initialize();
    }

    tRootGenerator->Activate();
    for( tCommandIt = tCommands.begin(); tCommandIt != tCommands.end(); tCommandIt++ )
    {
        tCommand = *tCommandIt;
        tCommand->Activate();
    }

    // initialize root
    TApplication tApplication( "Test Generator", 0, NULL );

    TCanvas tGeneratorEnergyCanvas( "generator_energy_canvas", "Generator Energy" );
    TH1D tGeneratorEnergyHistogram( "generator_energy_histogram", "generator_energy_histogram", 600, 700., 1300. );

    TCanvas tGeneratorMinTimeCanvas( "generator_min_time_canvas", "Generator MinTime" );
    TH1D tGeneratorMinTimeHistogram( "generator_min_time_histogram", "generator_min_time_histogram", 100, 0., 1. );

    TCanvas tGeneratorMaxTimeCanvas( "generator_max_time_canvas", "Generator MaxTime" );
    TH1D tGeneratorMaxTimeHistogram( "generator_max_time_histogram", "generator_max_time_histogram", 100, 0., 1. );

    TCanvas tGeneratorLocationCanvas( "generator_location_canvas", "Generator Location" );
    TH3D tGeneratorLocationHistogram( "generator_location_histogram", "generator_location_histogram", 100, -1.1, 1.1, 100, -1.1, 1.1, 100, -1.1, 1.1 );

    TCanvas tGeneratorRadiusCanvas( "generator_radius_canvas", "Generator Radius" );
    TH1D tGeneratorRadiusHistogram( "generator_radius_histogram", "generator_radius_histogram", 100, 0., 1. );

    TCanvas tParticlePositionCanvas( "particle_position_canvas", "Particle Position" );
    TH3D tParticlePositionHistogram( "particle_position_histogram", "particle_position_histogram", 100, -1.1, 1.1, 100, -1.1, 1.1, 100, -1.1, 1.1 );

    TCanvas tParticleDirectionCanvas( "particle_direction_canvas", "Particle Direction" );
    TH3D tParticleDirectionHistogram( "particle_direction_histogram", "particle_direction_histogram", 100, -1.1, 1.1, 100, -1.1, 1.1, 100, -1.1, 1.1 );

    TCanvas tParticleDirectionThetaCanvas( "particle_direction_theta_canvas", "Particle Direction Polar Angle" );
    TH1D tParticleDirectionThetaHistogram( "particle_direction_theta_histogram", "particle_direction_theta_histogram", 90, 0., 180. );

    TCanvas tParticleDirectionPhiCanvas( "particle_direction_phi_canvas", "Particle Direction Azimuthal Angle" );
    TH1D tParticleDirectionPhiHistogram( "particle_direction_phi_histogram", "particle_direction_phi_histogram", 180, 0., 360. );

    TCanvas tParticleEnergyCanvas( "particle_energy_canvas", "Particle Energy" );
    TH1D tParticleEnergyHistogram( "particle_energy_histogram", "particle_energy_histogram", 600, 700., 1300. );

    TCanvas tParticleTimeCanvas( "particle_time_canvas", "Particle Time" );
    TH1D tParticleTimeHistogram( "particle_time_histogram", "particle_time_histogram", 100, 0., 1. );

    // generate events
    KSEvent tEvent;
    KSParticle* tParticle;
    KSParticleIt tParticleIt;

    tRootGenerator->SetEvent( &tEvent );

    for( unsigned int tIndex = 0; tIndex < tCount; tIndex++ )
    {
        tRootGenerator->ExecuteGeneration();

        tGeneratorEnergyHistogram.Fill( tEvent.GeneratorEnergy() );
        tGeneratorMinTimeHistogram.Fill( tEvent.GeneratorMinTime() );
        tGeneratorMaxTimeHistogram.Fill( tEvent.GeneratorMaxTime() );
        tGeneratorLocationHistogram.Fill( tEvent.GeneratorLocation().X(), tEvent.GeneratorLocation().Y(), tEvent.GeneratorLocation().Z() );
        tGeneratorRadiusHistogram.Fill( tEvent.GeneratorRadius() );

        for( tParticleIt = tEvent.ParticleQueue().begin(); tParticleIt != tEvent.ParticleQueue().end(); tParticleIt++ )
        {
            tParticle = (*tParticleIt);

            tParticlePositionHistogram.Fill( tParticle->GetPosition().X(), tParticle->GetPosition().Y(), tParticle->GetPosition().Z() );
            tParticleDirectionHistogram.Fill( tParticle->GetMomentum().Unit().X(), tParticle->GetMomentum().Unit().Y(), tParticle->GetMomentum().Unit().Z() );
            tParticleDirectionThetaHistogram.Fill( (180. / KConst::Pi()) * tParticle->GetMomentum().PolarAngle() );
            tParticleDirectionPhiHistogram.Fill( (180. / KConst::Pi()) * tParticle->GetMomentum().AzimuthalAngle() );
            tParticleEnergyHistogram.Fill( tParticle->GetKineticEnergy_eV() );
            tParticleTimeHistogram.Fill( tParticle->GetTime() );

            delete tParticle;
        }
        tEvent.ParticleQueue().clear();

    }

    // show plots
    tGeneratorEnergyCanvas.cd( 0 );
    tGeneratorEnergyHistogram.SetFillColor( kBlue );
    tGeneratorEnergyHistogram.SetTitle( "Generator Energy" );
    tGeneratorEnergyHistogram.GetXaxis()->SetTitle( "Energy [eV]" );
    tGeneratorEnergyHistogram.Write();
    tGeneratorEnergyHistogram.Draw( "" );

    tGeneratorMinTimeCanvas.cd( 0 );
    tGeneratorMinTimeHistogram.SetFillColor( kBlue );
    tGeneratorMinTimeHistogram.SetTitle( "Generator Minimum Time" );
    tGeneratorMinTimeHistogram.GetXaxis()->SetTitle( "Minimum Time [sec]" );
    tGeneratorMinTimeHistogram.Write();
    tGeneratorMinTimeHistogram.Draw( "" );

    tGeneratorMaxTimeCanvas.cd( 0 );
    tGeneratorMaxTimeHistogram.SetFillColor( kBlue );
    tGeneratorMaxTimeHistogram.SetTitle( "Generator Maximum Time" );
    tGeneratorMaxTimeHistogram.GetXaxis()->SetTitle( "Maximum Time [sec]" );
    tGeneratorMaxTimeHistogram.Write();
    tGeneratorMaxTimeHistogram.Draw( "" );

    tGeneratorLocationCanvas.cd( 0 );
    tGeneratorLocationHistogram.SetMarkerColor( kRed );
    tGeneratorLocationHistogram.SetMarkerStyle( 20 );
    tGeneratorLocationHistogram.SetMarkerSize( 0.5 );
    tGeneratorLocationHistogram.SetTitle( "Generator Location" );
    tGeneratorLocationHistogram.GetXaxis()->SetTitle( "X [m]" );
    tGeneratorLocationHistogram.GetYaxis()->SetTitle( "Y [m]" );
    tGeneratorLocationHistogram.GetZaxis()->SetTitle( "Z [m]" );
    tGeneratorLocationHistogram.Write();
    tGeneratorLocationHistogram.Draw( "P" );

    tGeneratorRadiusCanvas.cd( 0 );
    tGeneratorRadiusHistogram.SetFillColor( kBlue );
    tGeneratorRadiusHistogram.SetTitle( "Generator Radius" );
    tGeneratorRadiusHistogram.GetXaxis()->SetTitle( "Radius [m]" );
    tGeneratorRadiusHistogram.Write();
    tGeneratorRadiusHistogram.Draw( "" );

    tParticlePositionCanvas.cd( 0 );
    tParticlePositionHistogram.SetMarkerColor( kRed );
    tParticlePositionHistogram.SetMarkerStyle( 20 );
    tParticlePositionHistogram.SetMarkerSize( 0.5 );
    tParticlePositionHistogram.SetTitle( "Particle Position" );
    tParticlePositionHistogram.GetXaxis()->SetTitle( "X [m]" );
    tParticlePositionHistogram.GetYaxis()->SetTitle( "Y [m]" );
    tParticlePositionHistogram.GetZaxis()->SetTitle( "Z [m]" );
    tParticlePositionHistogram.Write();
    tParticlePositionHistogram.Draw( "P" );

    tParticleDirectionCanvas.cd( 0 );
    tParticleDirectionHistogram.SetMarkerColor( kRed );
    tParticleDirectionHistogram.SetMarkerStyle( 20 );
    tParticleDirectionHistogram.SetMarkerSize( 0.5 );
    tParticleDirectionHistogram.SetTitle( "Particle Direction" );
    tParticleDirectionHistogram.GetXaxis()->SetTitle( "X [none]" );
    tParticleDirectionHistogram.GetYaxis()->SetTitle( "Y [none]" );
    tParticleDirectionHistogram.GetZaxis()->SetTitle( "Z [none]" );
    tParticleDirectionHistogram.Write();
    tParticleDirectionHistogram.Draw( "P" );

    tParticleDirectionThetaCanvas.cd( 0 );
    tParticleDirectionThetaHistogram.SetFillColor( kBlue );
    tParticleDirectionThetaHistogram.SetTitle( "Particle Polar Angle" );
    tParticleDirectionThetaHistogram.GetXaxis()->SetTitle( "Theta [degrees]" );
    tParticleDirectionThetaHistogram.Write();
    tParticleDirectionThetaHistogram.Draw( "" );

    tParticleDirectionPhiCanvas.cd( 0 );
    tParticleDirectionPhiHistogram.SetFillColor( kBlue );
    tParticleDirectionPhiHistogram.SetTitle( "Particle Azimuthal Angle" );
    tParticleDirectionPhiHistogram.GetXaxis()->SetTitle( "Phi [degrees]" );
    tParticleDirectionPhiHistogram.Write();
    tParticleDirectionPhiHistogram.Draw( "" );

    tParticleEnergyCanvas.cd( 0 );
    tParticleEnergyHistogram.SetFillColor( kBlue );
    tParticleEnergyHistogram.SetTitle( "Particle Energy" );
    tParticleEnergyHistogram.GetXaxis()->SetTitle( "Energy [eV]" );
    tParticleEnergyHistogram.Write();
    tParticleEnergyHistogram.Draw( "" );

    tParticleTimeCanvas.cd( 0 );
    tParticleTimeHistogram.SetFillColor( kBlue );
    tParticleTimeHistogram.SetTitle( "Particle Time" );
    tParticleTimeHistogram.GetXaxis()->SetTitle( "Time [sec]" );
    tParticleTimeHistogram.Write();
    tParticleTimeHistogram.Draw( "" );

    tApplication.Run();

    // deinitialize kassiopeia
    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        tCommand->Deactivate();
    }
    tRootGenerator->Deactivate();

    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        tCommand->Deinitialize();
    }
    tRootGenerator->Deinitialize();

    for( tCommandRIt = tCommands.rbegin(); tCommandRIt != tCommands.rend(); tCommandRIt++ )
    {
        tCommand = *tCommandRIt;
        delete tCommand;
    }


    return 0;
}

