#include "KCommandLineTokenizer.hh"
#include "KConditionProcessor.hh"
#include "KConst.h"
#include "KEMVTKFieldCanvas.hh"
#include "KElementProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KRandom.h"
#include "KSEvent.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSObject.h"
#include "KSRootMagneticField.h"
#include "KTagProcessor.hh"
#include "KToolbox.h"
#include "KVariableProcessor.hh"
#include "KXMLTokenizer.hh"
#include "TApplication.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TH3D.h"
#include "TMultiGraph.h"

#include <sstream>

using namespace Kassiopeia;
using namespace KGeoBag;
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
    tInputFile.AddToBases("TestField.xml");
    tInputFile.AddToPaths(string(CONFIG_DEFAULT_DIR) + string("/Validation"));
    tXMLTokenizer.ProcessFile(&tInputFile);

    // read in command line parameters
    if (anArgc <= 1) {
        mainmsg(eWarning) << ret;
        mainmsg << "usage:" << ret;
        mainmsg << "TestField <field_component>" << ret;
        mainmsg << ret;
        mainmsg << "available field components found in " << CONFIG_DEFAULT_DIR << "/Validation/TestField.xml" << ret;
        mainmsg << eom;
        return -1;
    }

    string tPath(anArgv[1]);

    unsigned int tCount = 100;

    // initialize magnetic field
    KSMagneticField* tRootMagneticField = getMagneticField(tPath);

    if (tRootMagneticField) {
        tRootMagneticField->Initialize();
        tRootMagneticField->Activate();
    }

    // initialize electric field
    KSElectricField* tRootElectricField = getElectricField(tPath);

    if (tRootElectricField) {
        tRootElectricField->Initialize();
        tRootElectricField->Activate();
    }

    KThreeVector tPoint, tMagneticPotential, tElectricField, tMagneticField;
    double tElectricPotential;
    KThreeMatrix tMagneticGradient;

    for (unsigned int i = 0; i < tCount; i++) {
        tPoint[0] = KRandom::GetInstance().Uniform(-10., 10.);
        tPoint[1] = KRandom::GetInstance().Uniform(-10., 10.);
        tPoint[2] = KRandom::GetInstance().Uniform(-10., 10.);

        if (tRootMagneticField) {
            tRootMagneticField->CalculatePotential(tPoint, 0, tMagneticPotential);
            tRootMagneticField->CalculateField(tPoint, 0, tMagneticField);
            tRootMagneticField->CalculateGradient(tPoint, 0, tMagneticGradient);
        }

        if (tRootElectricField) {
            tRootElectricField->CalculatePotential(tPoint, 0, tElectricPotential);
            tRootElectricField->CalculateField(tPoint, 0, tElectricField);
        }

        std::cout << "Point: " << tPoint << std::endl;
        if (tRootMagneticField) {
            std::cout << "  Magnetic Potential: " << tMagneticPotential << std::endl;
            std::cout << "  Magnetic Field:     " << tMagneticField << std::endl;
            std::cout << "  Magnetic Gradient:  " << tMagneticGradient << std::endl;
        }
        if (tRootElectricField) {
            std::cout << "  Electric Potential: " << tElectricPotential << std::endl;
            std::cout << "  Electric Field:     " << tElectricField << std::endl;
        }
    }

    if (tRootElectricField) {
        double z1 = -7.e-3;
        double z2 = 7.e-3;
        double x1 = 0.;
        double x2 = 1.e-2;

        double dx = 1.e-4;
        double dz = 1.e-4;

        KEMField::KEMFieldCanvas* fieldCanvas = nullptr;

        // fieldCanvas = new KEMField::KEMRootFieldCanvas(z1,z2,x1,x2,1.e30,true);
        fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);

        if (fieldCanvas) {
            int N_x = (int) ((x2 - x1) / dx);
            int N_z = (int) ((z2 - z1) / dz);

            int counter = 0;
            int countermax = N_z * N_x;

            std::vector<double> x_;
            std::vector<double> y_;
            std::vector<double> V_;

            double spacing[2] = {dz, dx};

            double phi = 0.;

            std::cout << "Computing potential field on a " << N_x << " by " << N_z << " grid" << std::endl;

            for (int g = 0; g < N_z; g++)
                x_.push_back(z1 + g * spacing[0] + spacing[0] / 2);

            for (int h = 0; h < N_x; h++)
                y_.push_back(x1 + h * spacing[1] + spacing[1] / 2);

            for (int g = 0; g < N_z; g++) {
                for (int h = 0; h < N_x; h++) {
                    double P[3] = {y_[h], 0., x_[g]};

                    tRootElectricField->CalculatePotential(P, 0, phi);

                    counter++;
                    if (counter * 100 % countermax == 0) {
                        std::cout << "\r";
                        std::cout << int((float) counter / countermax * 100) << " %";
                        std::cout.flush();
                    }

                    V_.push_back(phi);
                }
            }
            std::cout << "\r";
            std::cout.flush();

            fieldCanvas->DrawFieldMap(x_, y_, V_, false, .5);
            fieldCanvas->LabelAxes("z (m)", "r (m)", "#Phi (V)");
            fieldCanvas->SaveAs("VFieldMap_rz.gif");
        }
    }

    /*
// initialize root
TApplication tApplication( "Test Field", 0, NULL );

TCanvas tMagneticFieldCanvas( "magnetic_field_canvas", "MagneticField" );
TH1D tMagneticFieldXHistogram( "magnetic_field_x_histogram", "magnetic_field_x_histogram", 600,0.,6.);
TH1D tMagneticFieldYHistogram( "magnetic_field_y_histogram", "magnetic_field_y_histogram", 600,0.,6.);
TH1D tMagneticFieldZHistogram( "magnetic_field_z_histogram", "magnetic_field_z_histogram", 600,0.,6.);



// show plots
tMagneticFieldCanvas.cd( 0 );
tMagneticFieldXHistogram.SetFillColor( kBlue );
tMagneticFieldXHistogram.SetTitle( "Magnetic Field X" );
tMagneticFieldXHistogram.GetXaxis()->SetTitle( "Magnetic Field X [T]" );
tMagneticFieldXHistogram.Write();
tMagneticFieldXHistogram.Draw( "" );
tMagneticFieldYHistogram.SetTitle( "Magnetic Field Y" );
tMagneticFieldYHistogram.GetXaxis()->SetTitle( "Magnetic Field Y [T]" );
tMagneticFieldYHistogram.Write();
tMagneticFieldYHistogram.Draw( "" );
tMagneticFieldZHistogram.SetTitle( "Magnetic Field Z" );
tMagneticFieldZHistogram.GetXaxis()->SetTitle( "Magnetic Field Z [T]" );
tMagneticFieldZHistogram.Write();
tMagneticFieldZHistogram.Draw( "" );

tApplication.Run();
*/

    // deinitialize magnetic field
    if (tRootMagneticField) {
        tRootMagneticField->Deactivate();
        tRootMagneticField->Deinitialize();
    }

    // deinitialize electric field
    if (tRootElectricField) {
        tRootElectricField->Deactivate();
        tRootElectricField->Deinitialize();
    }


    return 0;
}
