// Compare fields and potentials of two input Kbd files with integrating field solver
// Author: Daniel Hilk
// Date: 07.06.2016

#include "KBinaryDataStreamer.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KDataDisplay.hh"
#include "KEMConstants.hh"
#include "KEMFieldCanvas.hh"
#include "KEMFileInterface.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KSADataStreamer.hh"
#include "KSerializer.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#include "KOpenCLSurfaceContainer.hh"
#endif

#include "TApplication.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1D.h"

double IJKLRANDOM;
void subrn(double* u, int len);
double randomnumber();

#define SEPCOMP 0

using namespace KEMField;

int main(int argc, char* argv[])
{

    std::string usage =
        "\n"
        "Usage: CompareFieldsAndPotentialsROOT <options>\n"
        "\n"
        "This program takes two KEMField files and compares the charge density values. These files must contain the same geometry.\n"
        "\n"
        "\tAvailable options:\n"
        "\t -h, --help               (shows this message and exits)\n"
        "\t -a, --fileA              (specify the first file)\n"
        "\t -b, --fileB              (specify the second file)\n"
        "\t -n, --nameA              (specify the surface container name in file A)\n"
        "\t -m, --nameB              (specify the surface container name in file B)\n"
        "\t -s, --size               (size of box for potential comparison)\n"
        "\n";

    static struct option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                          {"fileA", required_argument, nullptr, 'a'},
                                          {"fileB", required_argument, nullptr, 'b'},
                                          {"nameA", required_argument, nullptr, 'n'},
                                          {"nameB", required_argument, nullptr, 'm'},
                                          {"size", required_argument, nullptr, 's'}};

    static const char* optString = "ha:b:n:m:s:";

    std::string inFile1 = "";
    std::string inFile2 = "";
    std::string containerName1 = "surfaceContainer";
    std::string containerName2 = "surfaceContainer";
    double len = 1.0;
    (void) len;

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                break;
            case ('a'):
                inFile1 = std::string(optarg);
                break;
            case ('b'):
                inFile2 = std::string(optarg);
                break;
            case ('n'):
                containerName1 = std::string(optarg);
                break;
            case ('m'):
                containerName2 = std::string(optarg);
                break;
            case ('s'):
                len = atof(optarg);
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    std::string suffix1 = inFile1.substr(inFile1.find_last_of("."), std::string::npos);
    std::string suffix2 = inFile2.substr(inFile2.find_last_of("."), std::string::npos);

    struct stat fileInfo1;
    bool exists1;
    int fileStat1;

    // Attempt to get the file attributes
    fileStat1 = stat(inFile1.c_str(), &fileInfo1);
    if (fileStat1 == 0)
        exists1 = true;
    else
        exists1 = false;

    if (!exists1) {
        std::cout << "Error: file \"" << inFile1 << "\" cannot be read." << std::endl;
        return 1;
    }

    struct stat fileInfo2;
    bool exists2;
    int fileStat2;

    // Attempt to get the file attributes
    fileStat2 = stat(inFile2.c_str(), &fileInfo2);
    if (fileStat2 == 0)
        exists2 = true;
    else
        exists2 = false;

    if (!exists2) {
        std::cout << "Error: file \"" << inFile2 << "\" cannot be read." << std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix1.compare(binaryDataStreamer.GetFileSuffix()) != 0) {
        std::cout << "Error: unkown file extension \"" << suffix1 << "\"" << std::endl;
        return 1;
    }

    if (suffix2.compare(binaryDataStreamer.GetFileSuffix()) != 0) {
        std::cout << "Error: unkown file extension \"" << suffix2 << "\"" << std::endl;
        return 1;
    }

    //inspect the files
    KEMFileInterface::GetInstance()->Inspect(inFile1);
    KEMFileInterface::GetInstance()->Inspect(inFile2);

    //now read in the surface containers
    KSurfaceContainer surfaceContainer1;
    KSurfaceContainer surfaceContainer2;
    KEMFileInterface::GetInstance()->Read(inFile1, surfaceContainer1, containerName1);
    KEMFileInterface::GetInstance()->Read(inFile2, surfaceContainer2, containerName2);

    std::cout << "Surface container with name " << containerName1 << " in file 1 has size: " << surfaceContainer1.size()
              << std::endl;
    std::cout << "Surface container with name " << containerName2 << " in file 2 has size: " << surfaceContainer2.size()
              << std::endl;

    //now create the direct solver

#ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer* oclContainer1;
    oclContainer1 = new KOpenCLSurfaceContainer(surfaceContainer1);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer1);
    KOpenCLElectrostaticBoundaryIntegrator integrator1{KoclEBIFactory::MakeNumeric(*oclContainer1)};
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver1 =
        new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer1, integrator1);
    direct_solver1->Initialize();
#else
    KElectrostaticBoundaryIntegrator integrator1{KEBIFactory::MakeDefault()};
    auto* direct_solver1 =
        new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer1, integrator1);
#endif

    //now create the direct solver

#ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer* oclContainer2;
    oclContainer2 = new KOpenCLSurfaceContainer(surfaceContainer2);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer2);
    KOpenCLElectrostaticBoundaryIntegrator integrator2{KoclEBIFactory::MakeNumeric(*oclContainer2)};
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver2 =
        new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer2, integrator2);
    direct_solver2->Initialize();
#else
    KElectrostaticBoundaryIntegrator integrator2{KEBIFactory::MakeDefault()};
    auto* direct_solver2 =
        new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer2, integrator2);
#endif

    // Dice N points per evaluation

    const unsigned int noPoints(100);

    // Computation of difference between potentials and fields

    // Write out point if error is above user-defined threshold (valid for all steps)
    bool printDiff = false;
    const double diffPotThreshold = 3.5;

    ///////////////////////////////////////////////////
    KEMField::cout << "(1) Random points in center volume (cylinder)" << KEMField::endl;
    ///////////////////////////////////////////////////

    double cylZmin(-4.43);
    double cylZmax(4.43);
    double cylR(2.5);
    std::vector<KThreeVector> tPositions;

    for (unsigned int i = 0; i < noPoints; i++) {
        IJKLRANDOM = i + 1;

        double zRnd = randomnumber();
        double phi = 2. * M_PI * randomnumber();
        double r = cylR * sqrt(randomnumber());

        double x = cos(phi) * r;                          // x
        double y = sin(phi) * r;                          // y
        double z = cylZmin + zRnd * (cylZmax - cylZmin);  // z

        tPositions.push_back(KThreeVector(x, y, z));
    }

    double pot1, pot2, potDiff, fieldDiff;
    KThreeVector field1, field2;
    std::pair<KThreeVector, double> set1;
    std::pair<KThreeVector, double> set2;

    std::vector<double> potDiffContainer;
    std::vector<double> fieldDiffContainer;

    for (unsigned int i = 0; i < tPositions.size(); i++) {
#if SEPCOMP == 1
        pot1 = direct_solver1->Potential(tPositions[i]);
        pot2 = direct_solver2->Potential(tPositions[i]);
        field1 = direct_solver1->ElectricField(tPositions[i]);
        field2 = direct_solver2->ElectricField(tPositions[i]);
#else
        set1 = direct_solver1->ElectricFieldAndPotential(tPositions[i]);
        set2 = direct_solver2->ElectricFieldAndPotential(tPositions[i]);
        pot1 = set1.second;
        field1 = set1.first;
        pot2 = set2.second;
        field2 = set2.first;
#endif

        KEMField::cout << "Current field point: " << i << "\t\r";
        KEMField::cout.flush();

        potDiff = fabs(pot2 - pot1);
        fieldDiff = fabs(field2[0] - field1[0]) + fabs(field2[1] - field1[1]) + fabs(field2[2] - field1[2]);

        potDiffContainer.push_back(1e3 * potDiff);
        fieldDiffContainer.push_back(1e3 * fieldDiff);
    }

    // get min and max values stored in container

    double minPot = potDiffContainer[0];
    double maxPot = minPot;
    double minField = fieldDiffContainer[0];
    double maxField = minField;

    for (unsigned int i = 0; i < potDiffContainer.size(); i++) {
        if (potDiffContainer[i] < minPot)
            minPot = potDiffContainer[i];
        if (potDiffContainer[i] > maxPot)
            maxPot = potDiffContainer[i];
        if (fieldDiffContainer[i] < minField)
            minField = fieldDiffContainer[i];
        if (fieldDiffContainer[i] > maxField)
            maxField = fieldDiffContainer[i];
    }

    // create two TH1D in one window

    auto* fAppWindow = new TApplication("fAppWindow", nullptr, nullptr);

    std::stringstream outStream;
    outStream << "Absolute potential error in inner volume;Absolute pot. error (mV);N_{field points}";
    unsigned int tmpValue = static_cast<unsigned int>(maxPot) - static_cast<unsigned int>(minPot);
    unsigned int potBins = 25;  //(tmpValue - (tmpValue%100))*1.0; /* bin size */
    TH1D* thInnerPot = new TH1D("thInnerPot", outStream.str().c_str(), potBins, minPot, maxPot);
    outStream.str("");

    outStream << "Absolute field error in inner volume;Absolute field error (mV/m);N_{field points}";
    tmpValue = static_cast<unsigned int>(maxField) - static_cast<unsigned int>(minField);
    unsigned int fieldBins = (tmpValue - (tmpValue % 20)) * 0.2; /* bin size */
    TH1D* thInnerField = new TH1D("thInnerField", outStream.str().c_str(), fieldBins, minField, maxField);
    outStream.str("");

    for (unsigned int i = 0; i < potDiffContainer.size(); i++) {
        thInnerPot->Fill(potDiffContainer[i]);
        thInnerField->Fill(fieldDiffContainer[i]);
    }

    TCanvas c1("cIn", "Absolute field/potential difference in inner volume", 0, 0, 960, 760);
    c1.Divide(2, 1);
    c1.cd(1);
    thInnerPot->Draw();
    c1.cd(2);
    thInnerField->Draw();
    c1.Update();

    tPositions.clear();
    potDiffContainer.clear();
    fieldDiffContainer.clear();

    /////////////////////////////////////////////////
    KEMField::cout << "(2) Random points on outer cylinder surface" << KEMField::endl;
    /////////////////////////////////////////////////

    const double cylOutR = 4.67;

    for (unsigned int i = 0; i < noPoints; i++) {
        IJKLRANDOM = i + 1;

        double zRnd = randomnumber();
        double phi = 2. * M_PI * randomnumber();
        double r = cylOutR;

        double x = cos(phi) * r;                          // x
        double y = sin(phi) * r;                          // y
        double z = cylZmin + zRnd * (cylZmax - cylZmin);  // z

        tPositions.push_back(KThreeVector(x, y, z));
    }

    for (unsigned int i = 0; i < tPositions.size(); i++) {
#if SEPCOMP == 1
        pot1 = direct_solver1->Potential(tPositions[i]);
        pot2 = direct_solver2->Potential(tPositions[i]);
        field1 = direct_solver1->ElectricField(tPositions[i]);
        field2 = direct_solver2->ElectricField(tPositions[i]);
#else
        set1 = direct_solver1->ElectricFieldAndPotential(tPositions[i]);
        set2 = direct_solver2->ElectricFieldAndPotential(tPositions[i]);
        pot1 = set1.second;
        field1 = set1.first;
        pot2 = set2.second;
        field2 = set2.first;
#endif
        KEMField::cout << "Current field point: " << i << "\t\r";
        KEMField::cout.flush();

        potDiff = fabs(pot2 - pot1);
        fieldDiff = fabs(field2[0] - field1[0]) + fabs(field2[1] - field1[1]) + fabs(field2[2] - field1[2]);

        if (printDiff && (potDiff > diffPotThreshold))
            KEMField::cout << "Potential difference at " << tPositions[i] << " : " << potDiff << KEMField::endl;

        potDiffContainer.push_back(1e3 * potDiff);
        fieldDiffContainer.push_back(1e3 * fieldDiff);
    }

    // get min and max values stored in container

    minPot = potDiffContainer[0];
    maxPot = minPot;
    minField = fieldDiffContainer[0];
    maxField = minField;

    for (unsigned int i = 0; i < potDiffContainer.size(); i++) {
        if (potDiffContainer[i] < minPot)
            minPot = potDiffContainer[i];
        if (potDiffContainer[i] > maxPot)
            maxPot = potDiffContainer[i];
        if (fieldDiffContainer[i] < minField)
            minField = fieldDiffContainer[i];
        if (fieldDiffContainer[i] > maxField)
            maxField = fieldDiffContainer[i];
    }

    // create two TH1D in one window

    outStream << "Absolute potential error on outer surface;Absolute pot. error (mV);N_{field points}";
    tmpValue = static_cast<unsigned int>(maxPot) - static_cast<unsigned int>(minPot);
    potBins = (tmpValue - (tmpValue % 10)) * 0.1; /* bin size */
    TH1D* thOuterPot = new TH1D("thOuterPot", outStream.str().c_str(), potBins, minPot, maxPot);
    outStream.str("");

    outStream << "Absolute field error on outer surface;Absolute field error (mV/m);N_{field points}";
    tmpValue = static_cast<unsigned int>(maxField) - static_cast<unsigned int>(minField);
    fieldBins = (tmpValue - (tmpValue % 10)) * 0.1; /* bin size */
    TH1D* thOuterField = new TH1D("thOuterField", outStream.str().c_str(), fieldBins, minField, maxField);
    outStream.str("");

    for (unsigned int i = 0; i < potDiffContainer.size(); i++) {
        thOuterPot->Fill(potDiffContainer[i]);
        thOuterField->Fill(fieldDiffContainer[i]);
    }

    TCanvas c2("cOut", "Absolute field/potential difference on outer surface", 0, 0, 960, 760);
    c2.Divide(2, 1);
    c2.cd(1);
    thOuterPot->Draw();
    c2.cd(2);
    thOuterField->Draw();

    tPositions.clear();
    potDiffContainer.clear();
    fieldDiffContainer.clear();

    ////////////////////////////////////////////////////////////////////////////////////
    KEMField::cout << "(3) Random points on circle surface (xy) at z=0 with r=4.67 (r=4.672m -> caps)"
                   << KEMField::endl;
    ////////////////////////////////////////////////////////////////////////////////////

    double cylZ(0.);
    cylR = 4.67;

    for (unsigned int i = 0; i < noPoints; i++) {
        IJKLRANDOM = i + 1;

        double phi = 2. * M_PI * randomnumber();
        double r = cylR * sqrt(randomnumber());

        double x = cos(phi) * r;  // x
        double y = sin(phi) * r;  // y
        double z = cylZ;          // z

        tPositions.push_back(KThreeVector(x, y, z));
    }


    for (unsigned int i = 0; i < tPositions.size(); i++) {
#if SEPCOMP == 1
        pot1 = direct_solver1->Potential(tPositions[i]);
        pot2 = direct_solver2->Potential(tPositions[i]);
        field1 = direct_solver1->ElectricField(tPositions[i]);
        field2 = direct_solver2->ElectricField(tPositions[i]);
#else
        set1 = direct_solver1->ElectricFieldAndPotential(tPositions[i]);
        set2 = direct_solver2->ElectricFieldAndPotential(tPositions[i]);
        pot1 = set1.second;
        field1 = set1.first;
        pot2 = set2.second;
        field2 = set2.first;
#endif

        KEMField::cout << "Current field point: " << i << "\t\r";
        KEMField::cout.flush();

        potDiff = fabs(pot2 - pot1);
        fieldDiff = fabs(field2[0] - field1[0]) + fabs(field2[1] - field1[1]) + fabs(field2[2] - field1[2]);

        potDiffContainer.push_back(1e3 * potDiff);
        fieldDiffContainer.push_back(1e3 * fieldDiff);
    }

    // get min and max values stored in container

    minPot = potDiffContainer[0];
    maxPot = minPot;
    minField = fieldDiffContainer[0];
    maxField = minField;

    for (unsigned int i = 0; i < potDiffContainer.size(); i++) {
        if (potDiffContainer[i] < minPot)
            minPot = potDiffContainer[i];
        if (potDiffContainer[i] > maxPot)
            maxPot = potDiffContainer[i];
        if (fieldDiffContainer[i] < minField)
            minField = fieldDiffContainer[i];
        if (fieldDiffContainer[i] > maxField)
            maxField = fieldDiffContainer[i];
    }

    // create two TH1D in one window

    outStream << "Absolute potential error on analyzing plane surface;Absolute pot. error (mV);N_{field points}";
    tmpValue = static_cast<unsigned int>(maxPot) - static_cast<unsigned int>(minPot);
    potBins = (tmpValue - (tmpValue % 100)) * 1.0; /* bin size */
    TH1D* thSurfPot = new TH1D("thSurfPot", outStream.str().c_str(), potBins, minPot, maxPot);
    outStream.str("");

    outStream << "Absolute field error on analyzing plane surface;Absolute field error (mV/m);N_{field points}";
    tmpValue = static_cast<unsigned int>(maxField) - static_cast<unsigned int>(minField);
    fieldBins = (tmpValue - (tmpValue % 20)) * 0.2; /* bin size */
    TH1D* thSurfField = new TH1D("thSurfField", outStream.str().c_str(), fieldBins, minField, maxField);
    outStream.str("");

    for (unsigned int i = 0; i < potDiffContainer.size(); i++) {
        thSurfPot->Fill(potDiffContainer[i]);
        thSurfField->Fill(fieldDiffContainer[i]);
    }

    TCanvas c3("cSurf", "Absolute field/potential difference on analyzing plane surface", 0, 0, 960, 760);
    c3.Divide(2, 1);
    c3.cd(1);
    thSurfPot->Draw();
    c3.cd(2);
    thSurfField->Draw();
    c3.Update();

    tPositions.clear();
    potDiffContainer.clear();
    fieldDiffContainer.clear();

    ///////////////////////////////////////
    KEMField::cout << "(4) Points on line at z=0 from center to outer radii" << KEMField::endl;
    ///////////////////////////////////////

    double maxR = 4.67;
    double stepSize = maxR / noPoints;

    auto* plotDiffPot = new TGraph(noPoints);
    plotDiffPot->SetTitle("Error of radial potential");
    plotDiffPot->SetDrawOption("AC");
    plotDiffPot->SetMarkerColor(kRed);
    plotDiffPot->SetLineWidth(1);
    plotDiffPot->SetLineColor(kRed);
    plotDiffPot->SetMarkerSize(0.2);
    plotDiffPot->SetMarkerStyle(8);

    auto* plotDiffField = new TGraph(noPoints);
    plotDiffField->SetTitle("Error of radial field");
    plotDiffField->SetDrawOption("AC");
    plotDiffField->SetMarkerColor(kRed);
    plotDiffField->SetLineWidth(1);
    plotDiffField->SetLineColor(kRed);
    plotDiffField->SetMarkerSize(0.2);
    plotDiffField->SetMarkerStyle(8);

    KThreeVector tPos;

    for (unsigned int i = 0; i < noPoints; i++) {

        tPos.SetComponents(0., (i * stepSize), 0.);

        KEMField::cout << "Current field point: " << i << "\t\r";
        KEMField::cout.flush();

#if SEPCOMP == 1
        pot1 = direct_solver1->Potential(tPos);
        pot2 = direct_solver2->Potential(tPos);
        field1 = direct_solver1->ElectricField(tPos);
        field2 = direct_solver2->ElectricField(tPos);
#else
        set1 = direct_solver1->ElectricFieldAndPotential(tPos);
        set2 = direct_solver2->ElectricFieldAndPotential(tPos);
        pot1 = set1.second;
        field1 = set1.first;
        pot2 = set2.second;
        field2 = set2.first;
#endif

        potDiff = fabs(pot2 - pot1);
        fieldDiff = fabs(field2[0] - field1[0]) + fabs(field2[1] - field1[1]) + fabs(field2[2] - field1[2]);

        plotDiffPot->SetPoint(i, i * stepSize, 1.e3 * potDiff);
        plotDiffField->SetPoint(i, i * stepSize, 1.e3 * fieldDiff);
    }

    TCanvas c4("cRadial", "Absolute field/potential difference radial in analyzing plane surface", 0, 0, 960, 760);
    c4.Divide(2, 1);
    c4.cd(1);
    plotDiffPot->Draw();
    c4.cd(2);
    plotDiffField->Draw();
    c4.Update();

    fAppWindow->Run();

    return 0;
}

void subrn(double* u, int len)
{
    // This subroutine computes random numbers u[1],...,u[len]
    // in the (0,1) interval. It uses the 0<IJKLRANDOM<900000000
    // integer as initialization seed.
    //  In the calling program the dimension
    // of the u[] vector should be larger than len (the u[0] value is
    // not used).
    // For each IJKLRANDOM
    // numbers the program computes completely independent random number
    // sequences (see: F. James, Comp. Phys. Comm. 60 (1990) 329, sec. 3.3).

    static int iff = 0;
    static long ijkl, ij, kl, i, j, k, l, ii, jj, m, i97, j97, ivec;
    static float s, t, uu[98], c, cd, cm, uni;
    if (iff == 0) {
        if (IJKLRANDOM == 0) {
            std::cout << "Message from subroutine subrn:\n";
            std::cout << "the global integer IJKLRANDOM should be larger than 0 !!!\n";
            std::cout << "Computation is  stopped !!! \n";
            exit(0);
        }
        ijkl = IJKLRANDOM;
        if (ijkl < 1 || ijkl >= 900000000)
            ijkl = 1;
        ij = ijkl / 30082;
        kl = ijkl - 30082 * ij;
        i = ((ij / 177) % 177) + 2;
        j = (ij % 177) + 2;
        k = ((kl / 169) % 178) + 1;
        l = kl % 169;
        for (ii = 1; ii <= 97; ii++) {
            s = 0;
            t = 0.5;
            for (jj = 1; jj <= 24; jj++) {
                m = (((i * j) % 179) * k) % 179;
                i = j;
                j = k;
                k = m;
                l = (53 * l + 1) % 169;
                if ((l * m) % 64 >= 32)
                    s = s + t;
                t = 0.5 * t;
            }
            uu[ii] = s;
        }
        c = 362436. / 16777216.;
        cd = 7654321. / 16777216.;
        cm = 16777213. / 16777216.;
        i97 = 97;
        j97 = 33;
        iff = 1;
    }
    for (ivec = 1; ivec <= len; ivec++) {
        uni = uu[i97] - uu[j97];
        if (uni < 0.)
            uni = uni + 1.;
        uu[i97] = uni;
        i97 = i97 - 1;
        if (i97 == 0)
            i97 = 97;
        j97 = j97 - 1;
        if (j97 == 0)
            j97 = 97;
        c = c - cd;
        if (c < 0.)
            c = c + cm;
        uni = uni - c;
        if (uni < 0.)
            uni = uni + 1.;
        if (uni == 0.) {
            uni = uu[j97] * 0.59604644775391e-07;
            if (uni == 0.)
                uni = 0.35527136788005e-14;
        }
        u[ivec] = uni;
    }
    return;
}

////////////////////////////////////////////////////////////////

double randomnumber()
{
    // This function computes 1 random number in the (0,1) interval,
    // using the subrn subroutine.

    double u[2];
    subrn(u, 1);
    return u[1];
}
