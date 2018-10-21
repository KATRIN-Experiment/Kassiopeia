#include <stdlib.h>
#include <math.h>

#include "KMessage.h"
#include "KTextFile.h"

#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"
#include "KPrintProcessor.hh"

#ifdef Kommon_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

#ifdef KSC_USE_KALI
#include "KKaliProcessor.hh"
#endif

#include "KThreeVector.hh"

#include "KSToolbox.h"
#include "KSMainMessage.h"

#include "KSRootMagneticField.h"

#include "TGraph.h"
#include "TH2.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TPaletteAxis.h"
#include "TStyle.h"

#include <vector>


using namespace Kassiopeia;
using namespace katrin;
using namespace KGeoBag;

int main( int argc, char** argv )
{
    if( argc < 9 )
    {
        cout << "usage: ./SimpleMagneticFieldCalculator <config_file.xml> <x_min> <x_max> <y_min> <y_max> <z> <n> <magnetic_field_name1> [<magnetic_field_name2> <...>] " << endl;
        // ./CrossSectionMagneticFieldCalculator ~/Work/kasper/install/config/Kassiopeia/Examples/TestProject8Simulation.xml -1.0 1.0 -1.0 1.0 0.0 1000 field_electromagnet
        // ./SimpleMagneticFieldCalculator ~/Work/kasper/install/config/Kassiopeia/Examples/TTestProject8Simulation.xml 0.6143 0.0972 0.0 field_electromagnet
        // ./CrossSectionMagneticFieldCalculator ~/Work/kasper/install/config/Kassiopeia/Examples/TestProject8Simulation.xml -0.1 0.1 -0.1 0.1 2.45 1000 field_electromagnet
        exit( -1 );
    }

    string tFileName( argv[ 1 ] );

	// string X( argv[ 2 ] );
	// string Y( argv[ 3 ] );
	// string Z( argv[ 4 ] );
	// string tSpaceString(" ");
	// string tCombine = X+tSpaceString+Y+tSpaceString+Z;
	// istringstream Converter( tCombine );
	// KThreeVector tPosition;
	// Converter >> tPosition;

    double x_min = strtod( argv[2], NULL );
    double x_max = strtod( argv[3], NULL );
    double y_min = strtod( argv[4], NULL );
    double y_max = strtod( argv[5], NULL );
    double z = strtod( argv[6], NULL );
    int n = strtol( argv[7], NULL, 10 );

    //std::cout << x_min << " " << x_max << " " <<y_min << " " << y_max << " " << z << " " << n << "\n";

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine( argc, argv );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor( tCommandLine.GetVariables() );
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

	tVariableProcessor.InsertAfter( &tTokenizer );
	tIncludeProcessor.InsertAfter( &tVariableProcessor );

#ifdef Kommon_USE_ROOT
	KFormulaProcessor tFormulaProcessor;
	tFormulaProcessor.InsertAfter( &tVariableProcessor );
	tIncludeProcessor.InsertAfter( &tFormulaProcessor );
#ifdef KSC_USE_KALI
	KKaliProcessor tKaliProcessor;
	tKaliProcessor.InsertAfter( &tVariableProcessor );
	tFormulaProcessor.InsertAfter( &tKaliProcessor );
#endif
#endif

    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tPrintProcessor.InsertAfter( &tConditionProcessor );
    tTagProcessor.InsertAfter( &tPrintProcessor );
    tElementProcessor.InsertAfter( &tTagProcessor );

    mainmsg( eNormal ) << "starting initialization..." << eom;

    KTextFile tFile;
    tFile.SetDefaultPath( CONFIG_DEFAULT_DIR );
    tFile.AddToNames( tFileName );
    tTokenizer.ProcessFile( &tFile );

    mainmsg( eNormal ) << "...initialization finished" << eom;

    // initialize magnetic field
    KSRootMagneticField tRootMagneticField;

    for ( int tIndex = 8; tIndex < argc; tIndex++ )
    {
        KSMagneticField* tMagneticFieldObject = KSToolbox::GetInstance()->GetObjectAs< KSMagneticField >( argv[tIndex] );
        tMagneticFieldObject->Initialize();
        tRootMagneticField.AddMagneticField( tMagneticFieldObject  );
    }

    KThreeVector tMagneticField;

    double magnetic_field_magnitudes[n+1][n+1];

    for( int i=0; i <= n; i++ )
    {
      for ( int j=0; j <= n; j++ )
      {
        KThreeVector tPosition ( x_min + (x_max-x_min)*i/n, y_min + (y_max-y_min)*j/n, z );
        tRootMagneticField.CalculateField( tPosition, 0.0, tMagneticField );
        // magnetic_field_magnitudes[i][j] = tMagneticField.Dot( tPosition ) / tPosition.Magnitude() ;
        // if (magnetic_field_magnitudes[i][j] > 0.1 ){
        //   magnetic_field_magnitudes[i][j] = 0.1;
        // }
        // else if (magnetic_field_magnitudes[i][j] < -0.1 ){
        //   magnetic_field_magnitudes[i][j] = -0.1;
        // }
        magnetic_field_magnitudes[i][j] = log10( tMagneticField.Magnitude() );
        //std::cout << tPosition << "\n";
      }
    }

    TApplication* tApplication = new TApplication("app", 0, 0 );

    TCanvas* c1 = new TCanvas("c1","Track Comparison",200,10,1000,700);

    TH2* h2 = new TH2D("h2", "Central Cross-Section Magnetic Field", n+1, x_min, x_max, n+1, y_min, y_max);

    for( int i=0; i <= n; i++ )
    {
      for ( int j=0; j <= n; j++ )
      {
        h2->Fill( x_min + (x_max-x_min)*i/n, y_min + (y_max-y_min)*j/n, magnetic_field_magnitudes[i][j] );
      }
    }
    h2->GetXaxis()->SetTitle("x");
    h2->GetYaxis()->SetTitle("y");
    h2->GetZaxis()->SetTitle("log10 of B");

    h2->GetXaxis()->CenterTitle();
    h2->GetYaxis()->CenterTitle();
    h2->GetZaxis()->CenterTitle();

    TPaletteAxis *palette=(TPaletteAxis*)h2->FindObject("palette");
    //palette->SetTitle("log10 of B");

    TStyle *style  = new TStyle("Style", "Style_Description");
    style->SetPalette(1, 0);
    h2->Draw("COLZ");

    c1->Update();

    tApplication->Run();

    // mainmsg( eNormal ) <<"Magnetic Field at position "<<tPosition<<" is "<<tMagneticField<<eom;

    for ( int tIndex = 8; tIndex < argc; tIndex++ )
    {
        KSMagneticField* tMagneticFieldObject = KSToolbox::GetInstance()->GetObjectAs< KSMagneticField >( argv[tIndex] );
        tMagneticFieldObject->Deinitialize();
    }
    KSToolbox::DeleteInstance();

    return 0;
}
