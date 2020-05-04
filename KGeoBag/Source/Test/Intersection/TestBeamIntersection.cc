#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DPolygonWithArcs.hh"
#include "KGBeam.hh"
#include "KGExtrudedSurface.hh"
#include "KGPlanarArea.hh"
#include "KThreeVector.h"
#include "KTwoVector.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TH2D.h"
#include "TLine.h"
#include "TMath.h"
#include "TPolyLine3D.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TStyle.h"
#include "TView3D.h"

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
        std::cout << "Usage: TestBeamIntersection < integer >= 3 > <scale_factor> <closed=1, open=0>" << std::endl;
        std::cout << "scale_factor = 1 is the defaults if no parameter is given" << std::endl;
        std::cout << "" << std::endl;
        return 1;
    }

    std::string num_sides(argv[1]);
    std::stringstream ss;
    ss.clear();
    ss << num_sides;
    Int_t N;
    ss >> N;

    Double_t scale = 1;
    Int_t closed_open = 0;

    if (argc >= 3) {
        std::string sc(argv[2]);
        ss.clear();
        ss << sc;
        ss >> scale;
    }


    if (argc >= 4) {
        std::string co(argv[3]);
        ss.clear();
        ss << co;
        ss >> closed_open;
    }


    //compute the vertices for an N sided polygon
    katrin::KTwoVector* vert = new katrin::KTwoVector[N];
    Double_t two_pi = 2.0 * TMath::Pi();
    Double_t deg = two_pi / ((Double_t) N);
    std::vector<katrin::KTwoVector> vertices;


    for (Int_t i = 0; i < N; i++) {
        vert[i] = katrin::KTwoVector(TMath::Cos(deg * i), TMath::Sin(deg * i));
        vertices.push_back(vert[i]);
    };

    Double_t** v = new Double_t*[N];
    Double_t** sv = new Double_t*[N];

    for (UInt_t i = 0; i < N; i++) {
        v[i] = new Double_t[3];
        sv[i] = new Double_t[3];
    }

    for (UInt_t i = 0; i < N; i++) {
        v[i][0] = (vertices[i]).X();
        v[i][1] = (vertices[i]).Y();
        sv[i][0] = scale * (vertices[i]).X();
        sv[i][1] = scale * (vertices[i]).Y();
    }

    for (Int_t i = 0; i < N; i++) {
        v[i][2] = 0.;
        sv[i][0] += 3.;
        sv[i][1] += 6.0;
        sv[i][2] = 5.0;
    }

    Double_t point[3];
    Double_t normal[3];
    Double_t tempp[3];
    normal[0] = TMath::Sqrt(3);
    normal[1] = TMath::Sqrt(3);
    normal[2] = TMath::Sqrt(3);

    point[0] = 3;
    point[1] = 6;
    point[2] = 5;

    for (Int_t i = 0; i < N; i++) {
        KGBeam::LinePlaneIntersection(v[i], sv[i], point, normal, tempp);
        sv[i][0] = tempp[0];
        sv[i][1] = tempp[1];
        sv[i][2] = tempp[2];
    }


    KGBeam* beam = new KGBeam();
    if (closed_open) {
        beam->SetClosed();
    }
    else {
        beam->SetOpen();
    }

    for (UInt_t i = 0; i < N - 1; i++) {
        beam->AddStartLine(v[i], v[i + 1]);
        beam->AddEndLine(sv[i], sv[i + 1]);
    }
    //    beam->AddStartLine(v[N-1], v[0]);
    //    beam->AddEndLine(sv[N-1], sv[0]);

    beam->Initialize();

    Double_t limx = 20;
    Double_t limy = 20;
    Double_t limz = 20;

    //root app
    TApplication* App = new TApplication("test beam in 3d", &argc, argv);

    //make a canvas
    TCanvas* Canvas = new TCanvas("test beam in 3d", "test beam in 3d", 100, 100, 800, 800);

    //make a view
    Double_t MinRange[3] = {-1 * limx, -1 * limy, -1 * limz};
    Double_t MaxRange[3] = {limx, limy, limz};
    TView3D* View = new TView3D(1, MinRange, MaxRange);

    //the intersection points
    TPolyMarker3D* InterPoints = new TPolyMarker3D();  //panel 1
    UInt_t InterCount = 0;

    //   // lets transform the beam so it no longer lies perp to xy-plane
    //    KGTransformation* trans = new KGTransformation();
    //    trans->SetRotationConstructive(35., 21., 5.);
    //    trans->SetDisplacement(katrin::KThreeVector(4,5,0));
    //    beam->Transform(trans);

    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 1000000;
    Double_t x1, y1, z1, x2, y2, z2;
    katrin::KThreeVector temp1;
    katrin::KThreeVector temp2;
    TLine** moreLines = new TLine*[NLines];
    TRandom3* rand = new TRandom3(0);
    for (UInt_t i = 0; i < NLines; i++) {
        x1 = rand->Uniform(-1.0 * limx, limx);
        y1 = rand->Uniform(-1.0 * limy, limy);
        z1 = rand->Uniform(-1.0 * limz, limz);
        x2 = rand->Uniform(-1.0 * limx, limx);
        y2 = rand->Uniform(-1.0 * limy, limy);
        z2 = rand->Uniform(-1.0 * limz, limz);

        temp1 = katrin::KThreeVector(x1, y1, z1);
        temp2 = katrin::KThreeVector(x2, y2, z2);
        katrin::KThreeVector inter;

        Bool_t result = false;
        beam->NearestIntersection(temp1, temp2, result, inter);

        if (result == true) {
            InterPoints->SetPoint(InterCount, inter.X(), inter.Y(), inter.Z());
            InterCount++;
        }
    }

    std::cout << "number of intersections = " << InterCount << std::endl;


    Double_t xarrx[2] = {0, 10};
    Double_t xarry[2] = {0, 0};
    Double_t xarrz[2] = {0, 0};
    Double_t yarrx[2] = {0, 0};
    Double_t yarry[2] = {0, 10};
    Double_t yarrz[2] = {0, 0};
    Double_t zarrx[2] = {0, 0};
    Double_t zarry[2] = {0, 0};
    Double_t zarrz[2] = {0, 10};

    TPolyLine3D x_axis(2, xarrx, xarry, xarrz);
    TPolyLine3D y_axis(2, yarrx, yarry, yarrz);
    TPolyLine3D z_axis(2, zarrx, zarry, zarrz);

    x_axis.SetLineColor(kBlue);
    x_axis.Draw();
    y_axis.SetLineColor(kGreen);
    y_axis.Draw();
    z_axis.SetLineColor(kRed);
    z_axis.Draw();

    InterPoints->SetMarkerColor(kBlack);
    //InterPoints->SetMarkerStyle(kCircle);
    InterPoints->Draw();

    App->Run();
}
