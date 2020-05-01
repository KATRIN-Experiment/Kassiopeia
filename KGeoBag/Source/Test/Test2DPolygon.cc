#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
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

    if (argc < 2) {
        std::cout << "" << std::endl;
        std::cout << "Usage: Test2DPolygon < integer >= 3 >" << std::endl;
        std::cout << "" << std::endl;
        return 1;
    }

    std::string num_sides(argv[1]);
    std::stringstream ss;
    ss.clear();
    ss << num_sides;
    Int_t N;
    ss >> N;

    //compute the vertices for an N sided polygon
    katrin::KTwoVector* vert = new katrin::KTwoVector[N];
    Double_t two_pi = 2.0 * TMath::Pi();
    Double_t deg = two_pi / ((Double_t) N);
    std::vector<katrin::KTwoVector> vertices;

    for (Int_t i = 0; i < N; i++) {
        vert[i] = katrin::KTwoVector(TMath::Cos(deg * i), TMath::Sin(deg * i));
        vertices.push_back(vert[i]);
    };

    for (Int_t i = 0; i < N; i++) {
        vertices.push_back(0.75 * vert[i]);
    };

    for (Int_t i = 0; i < N; i++) {
        vertices.push_back(0.5 * vert[i]);
    };

    for (Int_t i = 0; i < N; i++) {
        vertices.push_back(0.25 * vert[i]);
    };


    KG2DPolygon* polygon;
    polygon = new KG2DPolygon();

    polygon->SetVertices(&vertices);

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

    Double_t lim = 1.1;


    TH2D* back = new TH2D("test", "test", 800, -1 * lim, lim, 800, -1.0 * lim, lim);
    TH2D* intersect = new TH2D("inter", "inter", 800, -1 * lim, lim, 800, -1.0 * lim, lim);


    TCanvas* canvas = new TCanvas("Polygon", "Polygon Test", 50, 50, 600, 600);
    canvas->SetFillColor(0);
    canvas->SetBorderSize(0);
    canvas->SetRightMargin(0.2);
    back->Draw("A");

    TLine** lines = new TLine*[vertices.size()];
    std::vector<KG2DLineSegment> sides;

    polygon->GetSides(&sides);
    katrin::KTwoVector begin;
    katrin::KTwoVector end;

    //draw the polygon
    for (UInt_t s = 0; s < sides.size(); s++) {
        begin = sides[s].GetFirstPoint();
        end = sides[s].GetSecondPoint();
        lines[s] = new TLine(begin.X(), begin.Y(), end.X(), end.Y());
        lines[s]->Draw("SAME");
    }

    //now lets test some points to see if the in/out routine works
    UInt_t NPoints = 10000;
    Double_t x, y;
    katrin::KTwoVector temp;
    TRandom3* rand = new TRandom3();
    for (UInt_t i = 0; i < NPoints; i++) {
        x = rand->Uniform(-1.0 * lim, lim);
        y = rand->Uniform(-1.0 * lim, lim);
        temp = katrin::KTwoVector(x, y);

        //std::cout<<"flag0"<<std::endl;

        if (polygon->IsInside(temp)) {
            back->Fill(x, y);
        }
    }
    back->Draw("SAME SCAT");

    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 5000;
    Double_t x1, y1, x2, y2;
    katrin::KTwoVector temp1;
    katrin::KTwoVector temp2;
    TLine** moreLines = new TLine*[NLines];
    for (UInt_t i = 0; i < NLines; i++) {
        x1 = rand->Uniform(-1.0 * lim, lim);
        y1 = rand->Uniform(-1.0 * lim, lim);

        x2 = rand->Uniform(-1.0 * lim, lim);
        y2 = rand->Uniform(-1.0 * lim, lim);
        temp1 = katrin::KTwoVector(x1, y1);
        temp2 = katrin::KTwoVector(x2, y2);
        katrin::KTwoVector inter;
        Double_t dist;

        //std::cout<<"flag0"<<std::endl;
        Bool_t result = false;
        polygon->NearestIntersection(temp1, temp2, result, inter);

        if (result == true) {
            moreLines[i] = new TLine(temp1.X(), temp1.Y(), temp2.X(), temp2.Y());
            moreLines[i]->SetLineColor(2);
            intersect->Fill(inter.X(), inter.Y());
            polygon->NearestDistance(inter, dist);
            //std::cout<<"intersection is "<<dist<<" away."<<std::endl;
            //moreLines[i]->Draw("SAME");
        }
        else {
            moreLines[i] = new TLine(temp1.X(), temp1.Y(), temp2.X(), temp2.Y());
            moreLines[i]->SetLineColor(4);
            //moreLines[i]->Draw("SAME");
        }
    }

    intersect->SetMarkerColor(kRed);
    intersect->SetMarkerStyle(kCircle);
    intersect->Draw("SAME");

    App->Run();

    delete[] vert;
}
