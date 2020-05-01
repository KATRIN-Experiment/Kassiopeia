// Adapted from TestField.cxx

#include "KCommandLineTokenizer.hh"
#include "KConditionProcessor.hh"
#include "KConst.h"
#include "KEMVTKFieldCanvas.hh"
#include "KElementProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KPrintProcessor.hh"
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

using std::string;
using std::stringstream;

using namespace Kassiopeia;
using namespace katrin;

int main(int argc, char** argv)
{
    // read in xml file
    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine(argc, argv);

    KXMLTokenizer tXMLTokenizer;
    KVariableProcessor tVariableProcessor(tCommandLine.GetVariables());
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter(&tXMLTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tIncludeProcessor.InsertAfter(&tFormulaProcessor);
    tLoopProcessor.InsertAfter(&tIncludeProcessor);
    tConditionProcessor.InsertAfter(&tLoopProcessor);
    tPrintProcessor.InsertAfter(&tConditionProcessor);
    tTagProcessor.InsertAfter(&tPrintProcessor);
    tElementProcessor.InsertAfter(&tTagProcessor);

    // read in command line parameters
    if (argc <= 2) {
        mainmsg(eWarning) << ret;
        mainmsg << "usage:" << ret;
        mainmsg << "TestPotentialmap <direct_field_component> <potentialmap_field_component> [config_file] " << ret;
        mainmsg << ret;
        mainmsg << "default field components found in " << CONFIG_DEFAULT_DIR << "/Validation/TestPotentialmap.xml"
                << ret;
        mainmsg << eom;
        return -1;
    }

    string tPathDirect(argv[1]);
    string tPathPotentialmap(argv[2]);

    KTextFile tInputFile;
    if (argc >= 4) {
        tInputFile.AddToBases(argv[3]);
        tInputFile.AddToPaths(string("."));
    }
    else {
        tInputFile.AddToBases("TestPotentialmap.xml");
        tInputFile.AddToPaths(string(CONFIG_DEFAULT_DIR) + string("/Validation"));
    }
    tXMLTokenizer.ProcessFile(&tInputFile);

    unsigned int tCount = 100;
    bool tPlotPotential = true;
    bool tPlotField = true;

    // initialize electric field
    KSElectricField* tRootElectricFieldDirect = getElectricField(tPathDirect);

    if (tRootElectricFieldDirect) {
        tRootElectricFieldDirect->Initialize();
        tRootElectricFieldDirect->Activate();
    }

    KSElectricField* tRootElectricFieldFromMap = getElectricField(tPathPotentialmap);

    if (tRootElectricFieldFromMap) {
        tRootElectricFieldFromMap->Initialize();
        tRootElectricFieldFromMap->Activate();
    }

    KThreeVector tPoint, tElectricFieldDirect, tElectricFieldFromMap;
    double tElectricPotentialDirect, tElectricPotentialFromMap;

    for (unsigned int i = 0; i < tCount; i++) {
        tPoint[0] = KRandom::GetInstance().Uniform(-0.045, 0.045);
        tPoint[1] = KRandom::GetInstance().Uniform(-0.045, 0.045);
        tPoint[2] = KRandom::GetInstance().Uniform(-0.045, 0.045);

        if (tRootElectricFieldDirect) {
            tRootElectricFieldDirect->CalculatePotential(tPoint, 0, tElectricPotentialDirect);
            tRootElectricFieldDirect->CalculateField(tPoint, 0, tElectricFieldDirect);
        }

        if (tRootElectricFieldFromMap) {
            tRootElectricFieldFromMap->CalculatePotential(tPoint, 0, tElectricPotentialFromMap);
            tRootElectricFieldFromMap->CalculateField(tPoint, 0, tElectricFieldFromMap);
        }

        std::cout << "Point: " << tPoint << std::endl;
        if (tRootElectricFieldDirect) {
            std::cout << "  Direct Electric Potential: " << tElectricPotentialDirect << std::endl;
            std::cout << "  Direct Electric Field:     " << tElectricFieldDirect << std::endl;
        }
        if (tRootElectricFieldFromMap) {
            std::cout << "  Mapped Electric Potential: " << tElectricPotentialFromMap << std::endl;
            std::cout << "  Mapped Electric Field:     " << tElectricFieldFromMap << std::endl;
        }
        if (tRootElectricFieldDirect && tRootElectricFieldFromMap) {
            std::cout << "  Electric Potential Diff:   " << (tElectricPotentialDirect - tElectricPotentialFromMap)
                      << std::endl;
            std::cout << "  Electric Field Diff:       " << (tElectricFieldDirect - tElectricFieldFromMap) << std::endl;
        }
    }

    if (tRootElectricFieldDirect && tRootElectricFieldFromMap) {
        double z1 = -0.045;
        double z2 = 0.045;
        double x1 = 0.;
        double x2 = 0.045;

        double dx = 1.e-4;
        double dz = 1.e-4;

        if (tPlotPotential) {
            KEMField::KEMFieldCanvas* fieldCanvas = NULL;

            fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);

            if (fieldCanvas) {
                int N_x = (int) ((x2 - x1) / dx);
                int N_z = (int) ((z2 - z1) / dz);

                int counter = 0;
                int countermax = N_z * N_x;

                std::vector<double> x_;
                std::vector<double> y_;
                std::vector<double> V1_, V2_, V_;

                double spacing[2] = {dz, dx};

                double phiDirect, phiFromMap, phiDelta;

                clock_t start, end;
                double time;

                std::cout << "Computing potential differences on a " << N_x << " by " << N_z << " grid (" << countermax
                          << " points)" << std::endl;

                start = clock();

                for (int g = 0; g < N_z; g++)
                    x_.push_back(z1 + g * spacing[0] + spacing[0] / 2);

                for (int h = 0; h < N_x; h++)
                    y_.push_back(x1 + h * spacing[1] + spacing[1] / 2);

                for (int g = 0; g < N_z; g++) {
                    for (int h = 0; h < N_x; h++) {
                        double P[3] = {y_[h], 0., x_[g]};

                        tRootElectricFieldDirect->CalculatePotential(P, 0, phiDirect);
                        tRootElectricFieldFromMap->CalculatePotential(P, 0, phiFromMap);
                        phiDelta = phiDirect - phiFromMap;

                        counter++;
                        if (counter % 10 == 0) {
                            std::cout << "\r";
                            std::cout << int((float) counter / countermax * 100) << " %";
                            std::cout.flush();
                        }

                        V1_.push_back(phiDirect);
                        V2_.push_back(phiFromMap);
                        V_.push_back(phiDelta);
                    }
                }
                std::cout << "\r";
                std::cout.flush();

                end = clock();
                std::cout << "Finished computing potential differences" << std::endl;
                time = ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
                std::cout << "   total time spent = " << time << std::endl;
                time /= (double) (countermax);
                std::cout << "   time per direct potential evaluation = " << time << std::endl;

                fieldCanvas->DrawFieldMap(x_, y_, V_, false, .5);
                fieldCanvas->LabelAxes("z (m)", "r (m)", "#Delta#Phi (V)");
                fieldCanvas->SaveAs("VPotentialmapPhiDifferenceMap_rz.png");

                fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);
                fieldCanvas->DrawFieldMap(x_, y_, V1_, false, .5);
                fieldCanvas->LabelAxes("z (m)", "r (m)", "#Phi_{direct} (V)");
                fieldCanvas->SaveAs("VPotentialmapPhiDirectMap_rz.png");

                fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);
                fieldCanvas->DrawFieldMap(x_, y_, V2_, false, .5);
                fieldCanvas->LabelAxes("z (m)", "r (m)", "#Phi_{fromMap} (V)");
                fieldCanvas->SaveAs("VPotentialmapPhiResultMap_rz.png");
            }
        }

        if (tPlotField) {
            KEMField::KEMFieldCanvas* fieldCanvas = NULL;

            fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);

            if (fieldCanvas) {
                int N_x = (int) ((x2 - x1) / dx);
                int N_z = (int) ((z2 - z1) / dz);

                int counter = 0;
                int countermax = N_z * N_x;

                std::vector<double> x_;
                std::vector<double> y_;
                std::vector<double> V1_, V2_, V_;

                double spacing[2] = {dz, dx};

                KThreeVector fieldDirect, fieldFromMap, fieldDelta;

                clock_t start, end;
                double time;

                std::cout << "Computing field differences on a " << N_x << " by " << N_z << " grid (" << countermax
                          << " points)" << std::endl;

                start = clock();

                for (int g = 0; g < N_z; g++)
                    x_.push_back(z1 + g * spacing[0] + spacing[0] / 2);

                for (int h = 0; h < N_x; h++)
                    y_.push_back(x1 + h * spacing[1] + spacing[1] / 2);

                for (int g = 0; g < N_z; g++) {
                    for (int h = 0; h < N_x; h++) {
                        double P[3] = {y_[h], 0., x_[g]};

                        tRootElectricFieldDirect->CalculateField(P, 0, fieldDirect);
                        tRootElectricFieldFromMap->CalculateField(P, 0, fieldFromMap);
                        fieldDelta = fieldDirect - fieldFromMap;

                        counter++;
                        if (counter % 10 == 0) {
                            std::cout << "\r";
                            std::cout << int((float) counter / countermax * 100) << " %";
                            std::cout.flush();
                        }

                        V1_.push_back(fieldDirect.Magnitude());
                        V2_.push_back(fieldFromMap.Magnitude());
                        V_.push_back(fieldDelta.Magnitude());
                    }
                }
                std::cout << "\r";
                std::cout.flush();

                end = clock();
                std::cout << "Finished computing field differences" << std::endl;
                time = ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
                std::cout << "   total time spent = " << time << std::endl;
                time /= (double) (countermax);
                std::cout << "   time per direct field evaluation = " << time << std::endl;

                fieldCanvas->DrawFieldMap(x_, y_, V_, false, .5);
                fieldCanvas->LabelAxes("z (m)", "r (m)", "#DeltaE (V)");
                fieldCanvas->SaveAs("VPotentialmapFieldDifferenceMap_rz.png");

                fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);
                fieldCanvas->DrawFieldMap(x_, y_, V1_, false, .5);
                fieldCanvas->LabelAxes("z (m)", "r (m)", "E_{direct} (V)");
                fieldCanvas->SaveAs("VPotentialmapFieldDirectMap_rz.png");

                fieldCanvas = new KEMField::KEMVTKFieldCanvas(z1, z2, x1, x2, 1.e30, true);
                fieldCanvas->DrawFieldMap(x_, y_, V2_, false, .5);
                fieldCanvas->LabelAxes("z (m)", "r (m)", "E_{fromMap} (V)");
                fieldCanvas->SaveAs("VPotentialmapFieldResultMap_rz.png");
            }
        }
    }

    // deinitialize electric field
    if (tRootElectricFieldDirect) {
        tRootElectricFieldDirect->Deactivate();
        tRootElectricFieldDirect->Deinitialize();
    }

    if (tRootElectricFieldFromMap) {
        tRootElectricFieldFromMap->Deactivate();
        tRootElectricFieldFromMap->Deinitialize();
    }


    return 0;
}