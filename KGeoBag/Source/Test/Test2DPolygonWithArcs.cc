#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DPolygonWithArcs.hh"
#include "KThreeVector.h"
#include "KTwoVector.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TH2D.h"
#include "TLine.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TStyle.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{

    //the vertices
    const Int_t N = 7;
    std::vector<katrin::KTwoVector> vertices;
    vertices.resize(N);

    vertices[0] = katrin::KTwoVector(2.5, 0);
    vertices[1] = katrin::KTwoVector(0.5, 5.0);
    vertices[2] = katrin::KTwoVector(5.5, 9.0);
    vertices[3] = katrin::KTwoVector(0, 17.);
    vertices[4] = katrin::KTwoVector(-5.5, 9.0);
    vertices[5] = katrin::KTwoVector(-0.5, 5.0);
    vertices[6] = katrin::KTwoVector(-2.5, 0);

    Double_t r1 = 4.5;
    Double_t r2 = 3.21;
    Double_t r3 = 19.5;

    std::vector<KGVertexSideDescriptor> vx_desc;
    vx_desc.resize(N);


    vx_desc[0].Vertex = vertices[0];
    vx_desc[0].IsArc = true;
    vx_desc[0].Radius = r1;
    vx_desc[0].IsRight = true;
    vx_desc[0].IsCCW = false;

    vx_desc[1].Vertex = vertices[1];
    vx_desc[1].IsArc = true;
    vx_desc[1].Radius = r2;
    vx_desc[1].IsRight = true;
    vx_desc[1].IsCCW = true;

    vx_desc[2].Vertex = vertices[2];
    vx_desc[2].IsArc = true;
    vx_desc[2].Radius = r3;
    vx_desc[2].IsRight = true;
    vx_desc[2].IsCCW = false;

    vx_desc[3].Vertex = vertices[3];
    vx_desc[3].IsArc = true;
    vx_desc[3].Radius = r3;
    vx_desc[3].IsRight = true;
    vx_desc[3].IsCCW = false;

    vx_desc[4].Vertex = vertices[4];
    vx_desc[4].IsArc = true;
    vx_desc[4].Radius = r2;
    vx_desc[4].IsRight = true;
    vx_desc[4].IsCCW = true;

    vx_desc[5].Vertex = vertices[5];
    vx_desc[5].IsArc = true;
    vx_desc[5].Radius = r1;
    vx_desc[5].IsRight = true;
    vx_desc[5].IsCCW = false;

    vx_desc[6].Vertex = vertices[6];
    vx_desc[6].IsArc = false;
    vx_desc[6].Radius = r1;
    vx_desc[6].IsRight = true;
    vx_desc[6].IsCCW = false;


    KG2DPolygonWithArcs* polygon;
    polygon = new KG2DPolygonWithArcs();

    polygon->SetDescriptors(&vx_desc);

    std::cout << "Polygon has " << vertices.size() << " vertices." << std::endl;

    Bool_t isSimple = polygon->IsSimple();

    if (isSimple) {
        std::cout << "Polygon is simple." << std::endl;
    }
    else {
        std::cout << "Polygon is not simple." << std::endl;
    }


    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR", &argc, argv);

    Double_t limx = 10;
    Double_t limy = 20;


    TH2D* back = new TH2D("test", "test", 800, -1 * limx, limx, 800, -1, limy);
    TH2D* inside = new TH2D("inside", "inside", 800, -1 * limx, limx, 800, -1, limy);
    TH2D* intersect = new TH2D("inter", "inter", 800, -1 * limx, limx, 800, -1, limy);

    TCanvas* canvas = new TCanvas("Polygon", "Polygon Test", 50, 50, 600, 600);
    canvas->SetFillColor(0);
    canvas->SetBorderSize(0);
    canvas->SetRightMargin(0.2);
    //draw the polygon by filling in the points inside it
    Double_t x;
    Double_t y;
    katrin::KTwoVector temp;
    for (Int_t xbin = 0; xbin < 800; xbin++) {
        for (Int_t ybin = 0; ybin < 800; ybin++) {
            x = back->GetXaxis()->GetBinCenter(xbin);
            y = back->GetYaxis()->GetBinCenter(ybin);
            temp = katrin::KTwoVector(x, y);
            if (polygon->IsInside(temp)) {
                back->Fill(x, y, 0.1);
            }
            else {
                back->Fill(x, y, 10);
            }
        }
    }
    back->Draw("ACOL");


    //now lets test some points to see if the in/out routine works
    UInt_t NPoints = 100000;
    TRandom3* rand = new TRandom3(0);
    for (UInt_t i = 0; i < NPoints; i++) {
        x = rand->Uniform(-1.0 * limx, limx);
        y = rand->Uniform(-1.0, limy);
        temp = katrin::KTwoVector(x, y);

        if (polygon->IsInside(temp)) {
            inside->Fill(x, y);
        }
    }
    inside->SetMarkerColor(kBlack);
    inside->Draw("SAME SCAT");

    //now lets test some points to see if the nearestpoint routine works
    UInt_t NNearestPoints = 1000;
    katrin::KTwoVector temp1;
    katrin::KTwoVector temp2;
    TLine** l = new TLine*[NNearestPoints];
    for (UInt_t i = 0; i < NNearestPoints; i++) {
        x = rand->Uniform(-1.0 * limx, limx);
        y = rand->Uniform(-1.0, limy);
        temp1 = katrin::KTwoVector(x, y);
        polygon->NearestPoint(temp1, temp2);
        l[i] = new TLine(x, y, temp2.X(), temp2.Y());
        l[i]->SetLineColor(kYellow);
        l[i]->Draw("SAME");
    }


    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 100000;
    Double_t x1, y1, x2, y2;
    //    katrin::KTwoVector temp1;
    //    katrin::KTwoVector temp2;
    TLine** moreLines = new TLine*[NLines];
    for (UInt_t i = 0; i < NLines; i++) {
        x1 = rand->Uniform(-1.0 * limx, limx);
        y1 = rand->Uniform(-1.0, limy);
        x2 = rand->Uniform(-1.0 * limx, limx);
        y2 = rand->Uniform(-1.0, limy);

        temp1 = katrin::KTwoVector(x1, y1);
        temp2 = katrin::KTwoVector(x2, y2);
        katrin::KTwoVector inter;
        katrin::KTwoVector nearest;

        //std::cout<<"flag0"<<std::endl;
        Bool_t result = false;
        polygon->NearestIntersection(temp1, temp2, result, inter);

        if (result == true) {
            moreLines[i] = new TLine(temp1.X(), temp1.Y(), temp2.X(), temp2.Y());
            moreLines[i]->SetLineColor(2);
            intersect->Fill(inter.X(), inter.Y());
            polygon->NearestPoint(inter, nearest);
            //moreLines[i]->Draw("SAME");
        }
        else {
            moreLines[i] = new TLine(temp1.X(), temp1.Y(), temp2.X(), temp2.Y());
            moreLines[i]->SetLineColor(4);
            //moreLines[i]->Draw("SAME");
        }
    }

    intersect->SetMarkerColor(kGreen);
    intersect->SetMarkerStyle(kCircle);
    intersect->Draw("SAME");

    App->Run();
}
