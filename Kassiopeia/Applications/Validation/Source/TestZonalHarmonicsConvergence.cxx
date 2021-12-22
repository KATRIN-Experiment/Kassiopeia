#include "KCommandLineTokenizer.hh"
#include "KConditionProcessor.hh"
#include "KConst.h"
#include "KElectricZHFieldSolver.hh"
#include "KElectromagnetContainer.hh"
#include "KElementProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KGElectrostaticBoundaryField.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KSMainMessage.h"
#include "KTagProcessor.hh"
#include "KThreeVector.hh"
#include "KToolbox.h"
#include "KVariableProcessor.hh"
#include "KXMLTokenizer.hh"
#include "TApplication.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1D.h"

#include <sstream>

using namespace KEMField;
using namespace Kassiopeia;
using namespace katrin;
using namespace std;

int main(int anArgc, char** anArgv)
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

    tVariableProcessor.InsertAfter(&tXMLTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tIncludeProcessor.InsertAfter(&tFormulaProcessor);
    tLoopProcessor.InsertAfter(&tIncludeProcessor);
    tConditionProcessor.InsertAfter(&tLoopProcessor);
    tTagProcessor.InsertAfter(&tConditionProcessor);
    tElementProcessor.InsertAfter(&tTagProcessor);

    KTextFile tInputFile;
    tInputFile.AddToBases("TestZonalHarmonics.xml");
    tInputFile.AddToPaths(string(CONFIG_DEFAULT_DIR) + "/Validation");
    tXMLTokenizer.ProcessFile(&tInputFile);

    // read in command line parameters
    if (anArgc != 2) {
        mainmsg(eWarning) << "usage:" << ret;
        mainmsg << "TestZonalHarmonicsConvergence <FIELDNAME>" << eom;
        return -1;
    }

    string tElFieldName = anArgv[1];
    string tMagFieldName = anArgv[2];
    mainmsg(eNormal) << "Using electric Field with name <" << tElFieldName << ">" << eom;
    mainmsg(eNormal) << "Using magentic Field with name <" << tMagFieldName << ">" << eom;

    // initialize root
    TApplication tApplication("Test ZonalHarmonics", nullptr, nullptr);

    TCanvas tConvergenceCanvas("convergence_canvas", "convergence radius");
    TGraph tElCentralExpansionGraph;
    //    TGraph tMagCentralExpansionGraph;

    //    TCanvas tMagFieldCanvas( "magnetic_field_canvas", "magnetic field strength" );
    //    TGraph tMagStrengthGraph;


    auto* tElField = KToolbox::GetInstance().Get<KGElectrostaticBoundaryField>(tElFieldName);
    tElField->Initialize();

    //    KStaticElectromagnetField* tMagField =
    //          KToolbox::GetInstance().Get<KStaticElectromagnetField>( tMagFieldName );
    //    tMagField->Initialize();

    KEMField::KSmartPointer<KGeoBag::KGBEMConverter> tElConverter = tElField->GetConverter();
    //    KGElectromagnetConverter* tMagConverter = tMagField->GetConverter();

    KElectricZHFieldSolver* tElZHSolver = dynamic_cast<KElectricZHFieldSolver*>(&(*tElField->GetFieldSolver()));
    //    KZonalHarmonicMagnetostaticFieldSolver* tMagZHSolver = dynamic_cast<KZonalHarmonicMagnetostaticFieldSolver*>(&(*tMagField->GetFieldSolver()));

    double tZ = -12.2;
    double tR = 0.;
    KThreeVector tPosition;

    for (int i = 0; i < 5000; i++) {
        tZ = -12.2 + i * 2. * 12.2 / 5000.;
        mainmsg(eNormal) << "Electric Field: Z Position: " << i << "/5000" << reom;
        for (int j = 0; j < 5000; j++) {
            tR = j * 0.001;

            tPosition.SetComponents(0., tR, tZ);
            if (!tElZHSolver->UseCentralExpansion(tElConverter->GlobalToInternalPosition(tPosition))) {
                tElCentralExpansionGraph.SetPoint(i, tZ, tR - 0.001);
                break;
            }
        }
    }

    //    KThreeVector tCheckMagfield;

    //    for( int i=0; i<5000; i++)
    //    {
    //        tZ = -12.2 + i*2.*12.2/5000.;
    //        mainmsg( eNormal ) << "Magnetic Field: Z Position: " << i <<"/5000" << reom;

    //        tMagField->CalculateField(KThreeVector(0.0,0.0,tZ),0.0,tCheckMagfield);
    //        tMagStrengthGraph.SetPoint(i, tZ, tCheckMagfield.Magnitude() );
    //        for ( int j=0; j<7000; j++ )
    //        {
    //            tR = j*0.001;

    //            tPosition.SetComponents(0.,tR,tZ);
    //            KPosition tMagneticPosition =  tMagConverter->GlobalToInternalPosition( tPosition ) + KPosition(0.0,0.0,12.2);
    //            if (!tMagZHSolver->UseCentralExpansion( tMagneticPosition  ) )
    //            {
    //                tMagCentralExpansionGraph.SetPoint(i, tZ, tR-0.001);
    //                break;
    //            }
    //        }
    //    }

    tConvergenceCanvas.cd(0);
    tElCentralExpansionGraph.SetMarkerColor(kRed);
    tElCentralExpansionGraph.SetMarkerStyle(20);
    tElCentralExpansionGraph.SetMarkerSize(0.5);
    tElCentralExpansionGraph.SetLineWidth(1);
    tElCentralExpansionGraph.SetTitle("Convergence Graph");
    tElCentralExpansionGraph.GetXaxis()->SetTitle("Z Position");
    tElCentralExpansionGraph.GetYaxis()->SetTitle("Convergence Radius");
    tElCentralExpansionGraph.Draw("AP");

    //    tConvergenceCanvas.cd( 0 );
    //    tMagCentralExpansionGraph.SetMarkerColor( kRed );
    //    tMagCentralExpansionGraph.SetMarkerStyle( 20 );
    //    tMagCentralExpansionGraph.SetMarkerSize( 0.5 );
    //    tMagCentralExpansionGraph.SetLineWidth( 0.1 );
    //    tMagCentralExpansionGraph.SetTitle( "Convergence Graph" );
    //    tMagCentralExpansionGraph.GetXaxis()->SetTitle(  "Z Position" );
    //    tMagCentralExpansionGraph.GetYaxis()->SetTitle( "Convergence Radius" );
    //    tMagCentralExpansionGraph.Draw( "AP" );

    //    tMagFieldCanvas.cd( 0 );
    //    tMagStrengthGraph.SetMarkerColor( kRed );
    //    tMagStrengthGraph.SetMarkerStyle( 20 );
    //    tMagStrengthGraph.SetMarkerSize( 0.5 );
    //    tMagStrengthGraph.SetLineWidth( 0.1 );
    //    tMagStrengthGraph.SetTitle( "Fieldstrength" );
    //    tMagStrengthGraph.GetXaxis()->SetTitle(  "Z Position" );
    //    tMagStrengthGraph.GetYaxis()->SetTitle( "Field" );
    //    tMagStrengthGraph.Draw( "AP" );

    tApplication.Run();

    // deinitialize kassiopeia

    return 0;
}
