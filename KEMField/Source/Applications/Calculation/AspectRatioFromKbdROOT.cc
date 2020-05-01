// AspectRatioFromKbdROOT
// This program computes the aspect ratio distribution from a given Kbd file.
// Author: Daniel Hilk
// Date: 19.04.2016

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"
#include "KSADataStreamer.hh"
#include "KSerializer.hh"
#include "KSurfaceContainer.hh"
#include "KTypelist.hh"
#include "TApplication.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TStyle.h"

#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace KEMField;

namespace KEMField
{
class AspectRatioVisitor : public KSelectiveVisitor<KShapeVisitor, KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>
{
  public:
    using KSelectiveVisitor<KShapeVisitor, KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>::Visit;

    AspectRatioVisitor() {}

    void Visit(KTriangle& t) override
    {
        ProcessTriangle(t);
    }
    void Visit(KRectangle& r) override
    {
        ProcessRectangle(r);
    }
    void Visit(KLineSegment& l) override
    {
        ProcessLineSegment(l);
    }

    void ProcessTriangle(KTriangle& tri)
    {
        fShapeType = KThreeVector(1, 0, 0);

        const double data[11] = {tri.GetA(),
                                 tri.GetB(),
                                 tri.GetP0().X(),
                                 tri.GetP0().Y(),
                                 tri.GetP0().Z(),
                                 tri.GetN1().X(),
                                 tri.GetN1().Y(),
                                 tri.GetN1().Z(),
                                 tri.GetN2().X(),
                                 tri.GetN2().Y(),
                                 tri.GetN2().Z()};

        const double P0[3] = {data[2], data[3], data[4]};
        const double P1[3] = {data[2] + (data[0] * data[5]),
                              data[3] + (data[0] * data[6]),
                              data[4] + (data[0] * data[7])};
        const double P2[3] = {data[2] + (data[1] * data[8]),
                              data[3] + (data[1] * data[9]),
                              data[4] + (data[1] * data[10])};

        double a, b, c, max;
        double delx, dely, delz;

        delx = P1[0] - P0[0];
        dely = P1[1] - P0[1];
        delz = P1[2] - P0[2];

        a = std::sqrt(delx * delx + dely * dely + delz * delz);

        delx = P2[0] - P0[0];
        dely = P2[1] - P0[1];
        delz = P2[2] - P0[2];

        b = std::sqrt(delx * delx + dely * dely + delz * delz);

        delx = P1[0] - P2[0];
        dely = P1[1] - P2[1];
        delz = P1[2] - P2[2];

        c = std::sqrt(delx * delx + dely * dely + delz * delz);

        KThreeVector PA;
        KThreeVector PB;
        KThreeVector PC;
        KThreeVector V;
        KThreeVector X;
        KThreeVector Y;
        KThreeVector Q;
        KThreeVector SUB;

        //find the longest side:
        if (a > b) {
            max = a;
            PA = KThreeVector(P2[0], P2[1], P2[2]);
            PB = KThreeVector(P0[0], P0[1], P0[2]);
            PC = KThreeVector(P1[0], P1[1], P1[2]);
        }
        else {
            max = b;
            PA = KThreeVector(P1[0], P1[1], P1[2]);
            PB = KThreeVector(P2[0], P2[1], P2[2]);
            PC = KThreeVector(P0[0], P0[1], P0[2]);
        }

        if (c > max) {
            max = c;
            PA = KThreeVector(P0[0], P0[1], P0[2]);
            PB = KThreeVector(P1[0], P1[1], P1[2]);
            PC = KThreeVector(P2[0], P2[1], P2[2]);
        }

        //the line pointing along v is the y-axis
        V = PC - PB;
        Y = V.Unit();

        //q is closest point to fP[0] on line connecting fP[1] to fP[2]
        double t = (PA.Dot(V) - PB.Dot(V)) / (V.Dot(V));
        Q = PB + t * V;

        //the line going from fP[0] to fQ is the x-axis
        X = Q - PA;
        //gram-schmidt out any y-axis component in the x-axis
        double proj = X.Dot(Y);
        SUB = proj * Y;
        X = X - SUB;
        double H = X.Magnitude();  //compute triangle height along x

        //compute the triangles aspect ratio
        fAspectRatio = max / H;
    }

    void ProcessRectangle(KRectangle& r)
    {
        fShapeType = KThreeVector(0, 1, 0);

        //figure out which vertices make the sides
        KThreeVector p[4];
        p[0] = r.GetP0();
        p[1] = r.GetP1();
        p[2] = r.GetP2();
        p[3] = r.GetP3();

        double d01 = (p[0] - p[1]).Magnitude();
        double d02 = (p[0] - p[2]).Magnitude();
        double d03 = (p[0] - p[3]).Magnitude();

        int a_mid, b_mid;
        double max_dist = d01;
        a_mid = 2;
        b_mid = 3;
        if (d02 > max_dist) {
            max_dist = d02;
            a_mid = 3;
            b_mid = 1;
        }
        if (d03 > max_dist) {
            max_dist = d03;
            a_mid = 1;
            b_mid = 2;
        }

        double a = (p[a_mid] - p[0]).Magnitude();
        double b = (p[b_mid] - p[0]).Magnitude();

        double val = a / b;
        if (val < 1.0) {
            fAspectRatio = 1.0 / val;
        }
        else {
            fAspectRatio = val;
        }
    }

    void ProcessLineSegment(KLineSegment& l)
    {
        fShapeType = KThreeVector(0, 0, 1);

        fAspectRatio = (l.GetP1() - l.GetP0()).Magnitude() / l.GetDiameter();
    }

    double GetAspectRatio()
    {
        return fAspectRatio;
    }
    KThreeVector GetShapeType()
    {
        return fShapeType;
    }

  private:
    double fAspectRatio;
    KThreeVector fShapeType; /* ( vector component 0=triangle, 1=rectangle, 2=line segment ) */
};
}  // namespace KEMField

void GetMinMaxAspectRatio(double* retValues, unsigned int containerSize, std::vector<KThreeVector>& shapeTypes,
                          std::vector<double>& arValues)
{
    double aspectRatioTri = 0.;
    double aspectRatioTriMin = 0.;
    double aspectRatioTriMax = 0.;

    double aspectRatioRect = 0.;
    double aspectRatioRectMin = 0.;
    double aspectRatioRectMax = 0.;

    double aspectRatioLine = 0.;
    double aspectRatioLineMin = 0.;
    double aspectRatioLineMax = 0.;

    // getting minimal and maximum aspect ratio

    for (unsigned int i = 0; i < containerSize; i++) {
        if (shapeTypes[i].X() == 1.) {  // triangle
            aspectRatioTri = arValues[i];

            if (aspectRatioTri < aspectRatioTriMin)
                aspectRatioTriMin = aspectRatioTri;
            if (aspectRatioTri > aspectRatioTriMax)
                aspectRatioTriMax = aspectRatioTri;

            if (i == 0) {
                aspectRatioTriMin = aspectRatioTri;
                aspectRatioTriMax = aspectRatioTri;
            }
        }
        if (shapeTypes[i].Y() == 1.) {  // rectangle
            aspectRatioRect = arValues[i];
            if (aspectRatioRect < aspectRatioRectMin)
                aspectRatioRectMin = aspectRatioRect;
            if (aspectRatioRect > aspectRatioRectMax)
                aspectRatioRectMax = aspectRatioRect;

            if (i == 0) {
                aspectRatioRectMin = aspectRatioRect;
                aspectRatioRectMax = aspectRatioRect;
            }
        }
        if (shapeTypes[i].Z() == 1.) {  // line segment
            aspectRatioLine = arValues[i];
            if (aspectRatioLine < aspectRatioLineMin)
                aspectRatioLineMin = aspectRatioLine;

            if (aspectRatioLine < aspectRatioLineMin)
                aspectRatioLineMin = aspectRatioLine;
            if (aspectRatioLine > aspectRatioLineMax)
                aspectRatioLineMax = aspectRatioLine;

            if (i == 0) {
                aspectRatioLineMin = aspectRatioLine;
                aspectRatioLineMax = aspectRatioLine;
            }
        }
    }

    /////////////////

    retValues[0] = aspectRatioTriMin;
    retValues[1] = aspectRatioTriMax;
    retValues[2] = aspectRatioRectMin;
    retValues[3] = aspectRatioRectMax;
    retValues[4] = aspectRatioLineMin;
    retValues[5] = aspectRatioLineMax;

    KEMField::cout << "Min. aspect ratio for triangles: " << aspectRatioTriMin << ", max: " << aspectRatioTriMax
                   << KEMField::endl;
    KEMField::cout << "Min. aspect ratio for rectangles: " << aspectRatioRectMin << ", max: " << aspectRatioRectMax
                   << KEMField::endl;
    KEMField::cout << "Min. aspect ratio for line segments: " << aspectRatioLineMin << ", max: " << aspectRatioLineMax
                   << KEMField::endl;

    return;
}

std::vector<TH1D*> ComputeAspectRatios(unsigned int contSize, std::vector<KThreeVector>& types,
                                       std::vector<double>& arvalues)
{
    // get min and max aspect ratio values
    double minmax[6];
    GetMinMaxAspectRatio(minmax, contSize, types, arvalues);

    // histograms
    unsigned int tmpValue;

    std::stringstream outStream;

    outStream << "Distribution for triangles;aspect ratio;no. of triangles";
    tmpValue = static_cast<unsigned int>(minmax[1]) - static_cast<unsigned int>(minmax[0]);
    //unsigned int triBins = (tmpValue - (tmpValue%10))*0.1;
    unsigned int triBins = tmpValue;
    TH1D* thTriP0 = new TH1D("arTri", outStream.str().c_str(), triBins, minmax[0], minmax[1]);
    outStream.str("");

    outStream << "Distribution for rectangles;aspect ratio;no. of rectangles";
    tmpValue = static_cast<unsigned int>(minmax[3]) - static_cast<unsigned int>(minmax[2]);
    //unsigned int rectBins = (tmpValue - (tmpValue%10))*0.1;
    unsigned int rectBins = tmpValue;
    TH1D* thRectP0 = new TH1D("arRect", outStream.str().c_str(), rectBins, minmax[2], minmax[3]);
    outStream.str("");

    outStream << "Distribution for line segments;aspect ratio;no. of line segments";
    tmpValue = static_cast<unsigned int>(minmax[5]) - static_cast<unsigned int>(minmax[4]);
    //unsigned int lineBins = (tmpValue - (tmpValue%10))*0.1;
    unsigned int lineBins = tmpValue;
    TH1D* thLineP0 = new TH1D("arLine", outStream.str().c_str(), lineBins, minmax[4], minmax[5]);
    outStream.str("");

    std::cout << std::endl << "Bin sizes: \n";
    std::cout << "* Tri: " << triBins << std::endl;
    std::cout << "* Rect: " << rectBins << std::endl;
    std::cout << "* Line: " << lineBins << std::endl << std::endl;

    // fill histogram

    for (unsigned int i = 0; i < contSize; i++) {
        if (types[i].X() != 0) {  // triangle
            thTriP0->Fill(arvalues[i]);
        }
        if (types[i].Y() != 0) {  // rectangle
            thRectP0->Fill(arvalues[i]);
        }
        if (types[i].Z() != 0) {  // line segment
            thLineP0->Fill(arvalues[i]);
        }
    }

    std::vector<TH1D*> retData;
    retData.push_back(thTriP0);
    retData.push_back(thRectP0);
    retData.push_back(thLineP0);

    return retData;
}

int main(int argc, char* argv[])
{

    std::string usage =
        "\n"
        "Usage: AspectRatioFromKbdROOT <options>\n"
        "\n"
        "This program takes two KEMField files and compares the charge density values. These files must contain the same geometry.\n"
        "\n"
        "\tAvailable options:\n"
        "\t -h, --help               (shows this message and exits)\n"
        "\t -f, --file               (specify the input kbd file)\n"
        "\t -n, --name               (specify the name of the surface container)\n"
        "\n";

    static struct option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                          {"file", required_argument, nullptr, 'f'},
                                          {"name", required_argument, nullptr, 'n'}};

    static const char* optString = "h:f:n:";

    std::string inFile = "";
    std::string containerName = "surfaceContainer";

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                break;
            case ('f'):
                inFile = std::string(optarg);
                break;
            case ('n'):
                containerName = std::string(optarg);
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    std::string suffix = inFile.substr(inFile.find_last_of("."), std::string::npos);

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFile.c_str(), &fileInfo);
    if (fileStat == 0)
        exists = true;
    else
        exists = false;

    if (!exists) {
        std::cout << "Error: file \"" << inFile << "\" cannot be read." << std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix.compare(binaryDataStreamer.GetFileSuffix()) != 0) {
        std::cout << "Error: unkown file extension \"" << suffix << "\"" << std::endl;
        return 1;
    }

    //inspect the files
    KEMFileInterface::GetInstance()->Inspect(inFile);

    //now read in the surface containers
    KSurfaceContainer surfaceContainer;
    KEMFileInterface::GetInstance()->Read(inFile, surfaceContainer, containerName);

    std::cout << "Surface container with name " << containerName << " in file has size: " << surfaceContainer.size()
              << std::endl;

    //loop over every element in the container and retrieve shape data and the charge density

    AspectRatioVisitor fShapeVisitor;
    std::vector<double> values;
    std::vector<KThreeVector> types;
    KThreeVector countShapeTypes(0, 0, 0);

    KSurfaceContainer::iterator it;

    for (it = surfaceContainer.begin<KElectrostaticBasis>(); it != surfaceContainer.end<KElectrostaticBasis>(); ++it) {
        (*it)->Accept(fShapeVisitor);

        // count different shape types and save
        types.push_back(fShapeVisitor.GetShapeType());
        countShapeTypes += fShapeVisitor.GetShapeType();

        // save aspect ratios
        values.push_back(fShapeVisitor.GetAspectRatio());
    }

    // turn off statistics box
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // compute aspect ratios separately for different shape types
    std::vector<TH1D*> data;

    auto* fAppWindow = new TApplication("fAppWindow", nullptr, nullptr);

    std::string outputPath("./AspectRatioFromKdb.root");

    auto* gROOTFile = new TFile(outputPath.c_str(), "RECREATE");

    data = ComputeAspectRatios(surfaceContainer.size(), types, values);
    TCanvas c0("cAR", "Aspect ratio distribution", 0, 0, 960, 760);


    //c0.Divide( 3, 1 );
    //c0.cd(1);
    if (countShapeTypes[0] > 0)
        data[0]->Draw();
    //c0.cd(2);
    if (countShapeTypes[1] > 0)
        data[1]->Draw();
    //c0.cd(3);
    if (countShapeTypes[2] > 0)
        data[2]->Draw();
    //c0.Update();

    if (countShapeTypes[0] > 0)
        data[0]->Write("arTri");
    if (countShapeTypes[1] > 0)
        data[1]->Write("arRect");
    if (countShapeTypes[2] > 0)
        data[2]->Write("arLine");

    fAppWindow->Run();

    gROOTFile->Close();

    return 0;
}
