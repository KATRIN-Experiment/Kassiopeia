// DistanceRatioFromKbdROOT
// This program computes the distance ratio distribution from a given Kbd file.
// Author: Daniel Hilk
// Date: 08.05.2016

#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>

#include "KTypelist.hh"
#include "KSurfaceContainer.hh"
#include "KEMFileInterface.hh"
#include "KBinaryDataStreamer.hh"

#include "KSADataStreamer.hh"
#include "KSerializer.hh"
#include "KSADataStreamer.hh"

#include "TStyle.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TFile.h"

using namespace KEMField;


namespace KEMField{
class DistRatioVisitor :
        public KSelectiveVisitor<KShapeVisitor,
        KTYPELIST_3(KTriangle,KRectangle,KLineSegment)>
{
public:
    using KSelectiveVisitor<KShapeVisitor,KTYPELIST_3(KTriangle,KRectangle,KLineSegment)>::Visit;

    DistRatioVisitor(){}

    void Visit(KTriangle& t) { ProcessTriangle(t); }
    void Visit(KRectangle& r) { ProcessRectangle(r); }
    void Visit(KLineSegment& l) { ProcessLineSegment(l); }

    void ProcessTriangle(KTriangle& t)
    {
        // get missing side length
        const double lengthP1P2 = (t.GetP2() - t.GetP1()).Magnitude();

        fAverageSideLength = (t.GetA() + t.GetB() + lengthP1P2)/3.;

        // centroid
        fShapeCentroid = t.Centroid();

        // shape type
        fShapeType.SetComponents( 1, 0, 0 );
    }

    void ProcessRectangle(KRectangle& r)
    {
        fAverageSideLength = 0.5*(r.GetA() + r.GetB());

        // centroid
        fShapeCentroid = r.Centroid();

        // shape type
        fShapeType.SetComponents( 0, 1, 0 );
    }

    void ProcessLineSegment(KLineSegment& l)
    {
        // length of line segment
        fAverageSideLength = (l.GetP1()-l.GetP0()).Magnitude();
        // centroid
        fShapeCentroid = l.Centroid();

        // shape type
        fShapeType.SetComponents( 0, 0, 1 );
    }

    double GetAverageSideLength() { return fAverageSideLength; }
    KThreeVector GetCentroid(){ return fShapeCentroid; }
    KThreeVector GetShapeType(){ return fShapeType; }

private:
    double fAverageSideLength;
    KThreeVector fShapeCentroid;
    KThreeVector fShapeType; /* ( triangle, rectangle, line segment ) */
};
} /* KEMField namespace*/

void GetMinMaxDistanceRatio( double* retValues, KThreeVector fP, unsigned int containerSize, std::vector<KThreeVector> &shapeTypes,
        std::vector<double> &avgLengths, std::vector<KThreeVector> &centerPoints )
{
    double distanceRatioTri = 0.;
    double distanceRatioTriMin = 0.;
    double distanceRatioTriMax = 0.;

    double distanceRatioRect = 0.;
    double distanceRatioRectMin = 0.;
    double distanceRatioRectMax = 0.;

    double distanceRatioLine = 0.;
    double distanceRatioLineMin = 0.;
    double distanceRatioLineMax = 0.;

    // getting minimal and maximum distance ratio

    for( unsigned int i=0; i<containerSize; i++ ) {
        if( shapeTypes[i].X() == 1. ) { // triangle
            distanceRatioTri = ( (fP - centerPoints[i]).Magnitude() / avgLengths[i] );

            if(distanceRatioTri < distanceRatioTriMin) distanceRatioTriMin=distanceRatioTri;
            if(distanceRatioTri > distanceRatioTriMax) distanceRatioTriMax=distanceRatioTri;

            if( i==0 ) {
                distanceRatioTriMin=distanceRatioTri;
                distanceRatioTriMax=distanceRatioTri;
            }
        }
        if( shapeTypes[i].Y() == 1. ) { // rectangle
            distanceRatioRect = ( (fP - centerPoints[i]).Magnitude() / avgLengths[i] );
            if(distanceRatioRect < distanceRatioRectMin) distanceRatioRectMin=distanceRatioRect;
            if(distanceRatioRect > distanceRatioRectMax) distanceRatioRectMax=distanceRatioRect;

            if( i==0 ) {
                distanceRatioRectMin=distanceRatioRect;
                distanceRatioRectMax=distanceRatioRect;
            }

        }
        if( shapeTypes[i].Z() == 1. ) { // line segment
            distanceRatioLine = ( (fP - centerPoints[i]).Magnitude() / avgLengths[i] );
            if(distanceRatioLine < distanceRatioLineMin) distanceRatioLineMin=distanceRatioLine;

            if(distanceRatioLine < distanceRatioLineMin) distanceRatioLineMin=distanceRatioLine;
            if(distanceRatioLine > distanceRatioLineMax) distanceRatioLineMax=distanceRatioLine;

            if( i==0 ) {
                distanceRatioLineMin=distanceRatioLine;
                distanceRatioLineMax=distanceRatioLine;
            }
        }
    }

    /////////////////

    retValues[0] = distanceRatioTriMin;
    retValues[1] = distanceRatioTriMax;
    retValues[2] = distanceRatioRectMin;
    retValues[3] = distanceRatioRectMax;
    retValues[4] = distanceRatioLineMin;
    retValues[5] = distanceRatioLineMax;

    KEMField::cout << "For field point " << fP << KEMField::endl;
    KEMField::cout << "Min. distance ratio for triangles: " << distanceRatioTriMin << ", max: " << distanceRatioTriMax << KEMField::endl;
    KEMField::cout << "Min. distance ratio for rectangles: " << distanceRatioRectMin << ", max: " << distanceRatioRectMax << KEMField::endl;
    KEMField::cout << "Min. distance ratio for line segments: " << distanceRatioLineMin << ", max: " << distanceRatioLineMax << KEMField::endl;

    return;
}

std::vector<TH1D*> ComputeDistanceRatios( KThreeVector selectedPoint,
        unsigned int contSize, std::vector<KThreeVector> &types, std::vector<double> &lengths, std::vector<KThreeVector> &centroids )
{
    // get min and max dr values for chosen field point
    double minmax[6];
    GetMinMaxDistanceRatio( minmax, selectedPoint, contSize, types, lengths, centroids );

    // histograms
    unsigned int tmpValue;

    std::stringstream outStream;

    outStream << "Distribution for triangles, P=(" << selectedPoint.X() << " | " << selectedPoint.Y() << " | " << selectedPoint.Z() << ") ;distance ratio;no. of triangles";
    tmpValue = static_cast<unsigned int>(minmax[1]) - static_cast<unsigned int>(minmax[0]);
    unsigned int triBins = (tmpValue - (tmpValue%10))*0.1;
    TH1D* thTriP0 = new TH1D("drTri", outStream.str().c_str(), triBins, minmax[0], minmax[1]);
    outStream.str("");

    outStream << "Distribution for rectangles, P=(" << selectedPoint.X() << " | " << selectedPoint.Y() << " | " << selectedPoint.Z() << ") ;distance ratio;no. of rectangles";
    tmpValue = static_cast<unsigned int>(minmax[3]) - static_cast<unsigned int>(minmax[2]);
    unsigned int rectBins = (tmpValue - (tmpValue%10))*0.1;
    TH1D* thRectP0 = new TH1D("drRect", outStream.str().c_str(), rectBins, minmax[2], minmax[3]);
    outStream.str("");

    outStream << "Distribution for line segments, P=(" << selectedPoint.X() << " | " << selectedPoint.Y() << " | " << selectedPoint.Z() << ") ;distance ratio;no. of line segments";
    tmpValue = static_cast<unsigned int>(minmax[5]) - static_cast<unsigned int>(minmax[4]);
    unsigned int lineBins = (tmpValue - (tmpValue%10))*0.1;
    TH1D* thLineP0 = new TH1D("drLine", outStream.str().c_str(), lineBins, minmax[4], minmax[5]);
    outStream.str("");

    std::cout << std::endl << "Bin sizes: \n";
    std::cout << "* Tri: " << triBins << std::endl;
    std::cout << "* Rect: " << rectBins << std::endl;
    std::cout << "* Line: " << lineBins << std::endl << std::endl;

    // fill histogram

    double drTri = 0.;
    double drRect = 0.;
    double drLine = 0.;

    for( unsigned int i=0; i<contSize; i++ ) {
        if( types[i].X() == 1. ) { // triangle
            drTri = ( (selectedPoint - centroids[i]).Magnitude() / lengths[i] );
            thTriP0->Fill(drTri);
        }
        if( types[i].Y() == 1. ) { // rectangle
            drRect = ( (selectedPoint - centroids[i]).Magnitude() / lengths[i] );
            thRectP0->Fill(drRect);
        }
        if( types[i].Z() == 1. ) { // line segment
            drLine = ( (selectedPoint - centroids[i]).Magnitude() / lengths[i] );
            thLineP0->Fill(drLine);
        }
    }

    std::vector<TH1D*> retData;
    retData.push_back( thTriP0 );
    retData.push_back( thRectP0 );
    retData.push_back( thLineP0 );


    return retData;
}

int main(int argc, char* argv[])
{

    std::string usage =
    "\n"
    "Usage: DistanceRatioFromKbdROOT <options>\n"
    "\n"
    "This program computes the distance ratio distribution from a given Kbd file.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -f, --file               (specify the input kbd file)\n"
    "\t -n, --name               (specify the name of the surface container)\n"
    "\n";

    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
        {"file", required_argument, 0, 'f'},
        {"name", required_argument, 0, 'n'}
    };

    static const char *optString = "h:f:n:";

    std::string inFile = "";
    std::string containerName = "surfaceContainer";

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId) {
        case('h'): // help
            std::cout<<usage<<std::endl;
        break;
        case('f'):
            inFile = std::string(optarg);
        break;
        case ('n'):
            containerName = std::string(optarg);
        break;
        default: // unrecognized option
            std::cout<<usage<<std::endl;
        return 1;
        }
    }

    std::string suffix = inFile.substr(inFile.find_last_of("."),std::string::npos);

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFile.c_str(),&fileInfo);
    if(fileStat == 0)
    exists = true;
    else
    exists = false;

    if (!exists) {
		std::cout << "Error: file \"" << inFile <<"\" cannot be read." << std::endl;
		return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix.compare(binaryDataStreamer.GetFileSuffix()) != 0) {
        std::cout<<"Error: unkown file extension \""<<suffix<<"\"" << std::endl;
        return 1;
    }

    //inspect the files
    KEMFileInterface::GetInstance()->Inspect(inFile);

    //now read in the surface containers
    KSurfaceContainer surfaceContainer;
    KEMFileInterface::GetInstance()->Read(inFile,surfaceContainer,containerName);

    std::cout << "Surface container with name " << containerName << " in file has size: " << surfaceContainer.size() << std::endl;

    //loop over every element in the container and retrieve shape data and the charge density

    KThreeVector fieldPoint( 0., 0., 0. );

    DistRatioVisitor fShapeVisitor;
    std::vector<double> lengths;
    std::vector<KThreeVector> centroids;
    std::vector<KThreeVector> types;
    KThreeVector countShapeTypes( 0, 0, 0 );

    KSurfaceContainer::iterator it;

    for( it=surfaceContainer.begin<KElectrostaticBasis>(); it!=surfaceContainer.end<KElectrostaticBasis>(); ++it ) {
        (*it)->Accept(fShapeVisitor);

        // count different shape types and save
        types.push_back(fShapeVisitor.GetShapeType());
        countShapeTypes += fShapeVisitor.GetShapeType();

        // save center points
        centroids.push_back(fShapeVisitor.GetCentroid());

        // save average lengths
        lengths.push_back(fShapeVisitor.GetAverageSideLength());
    }

    // turn off statistics box
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // compute distance ratios separately for different shape types
    std::vector<TH1D*> data;

    TApplication* fAppWindow = new TApplication("fAppWindow", 0, NULL);

    std::string outputPath("./DistRatioFromKdb.root");

    TFile* gROOTFile = new TFile(outputPath.c_str(), "RECREATE");

    data = ComputeDistanceRatios( fieldPoint, surfaceContainer.size(), types, lengths, centroids );
    TCanvas c0("cDR", "Distance ratio distributions", 0, 0, 960, 760);
    //c0.Divide( 3, 1 );
    //c0.cd(1);
    if( countShapeTypes[0]>0 ) data[0]->Draw();
    //c0.cd(2);
    if( countShapeTypes[1]>0 ) data[1]->Draw();
    //c0.cd(3);
    if( countShapeTypes[2]>0 ) data[2]->Draw();
    //c0.Update();

    if( countShapeTypes[0]>0 ) data[0]->Write("drTri");
    if( countShapeTypes[1]>0 ) data[1]->Write("drRect");
    if( countShapeTypes[2]>0 ) data[2]->Write("drLine");

    fAppWindow->Run();

    gROOTFile->Close();

    return 0;
}
