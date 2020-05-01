#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DPolygonWithArcs.hh"
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
    vx_desc[0].IsRight = false;
    vx_desc[0].IsCCW = false;

    vx_desc[1].Vertex = vertices[1];
    vx_desc[1].IsArc = true;
    vx_desc[1].Radius = r2;
    vx_desc[1].IsRight = true;
    vx_desc[1].IsCCW = true;

    vx_desc[2].Vertex = vertices[2];
    vx_desc[2].IsArc = true;
    vx_desc[2].Radius = r3;
    vx_desc[2].IsRight = false;
    vx_desc[2].IsCCW = false;

    vx_desc[3].Vertex = vertices[3];
    vx_desc[3].IsArc = true;
    vx_desc[3].Radius = r3;
    vx_desc[3].IsRight = false;
    vx_desc[3].IsCCW = false;

    vx_desc[4].Vertex = vertices[4];
    vx_desc[4].IsArc = true;
    vx_desc[4].Radius = r2;
    vx_desc[4].IsRight = true;
    vx_desc[4].IsCCW = true;

    vx_desc[5].Vertex = vertices[5];
    vx_desc[5].IsArc = true;
    vx_desc[5].Radius = r1;
    vx_desc[5].IsRight = false;
    vx_desc[5].IsCCW = false;

    vx_desc[6].Vertex = vertices[6];
    vx_desc[6].IsArc = false;
    vx_desc[6].Radius = r1;
    vx_desc[6].IsRight = false;
    vx_desc[6].IsCCW = false;


    KG2DPolygonWithArcs* bare_polygon;
    bare_polygon = new KG2DPolygonWithArcs();

    bare_polygon->SetDescriptors(&vx_desc);

    //construct polygon embedded in 3d;
    KGPlanarArea<KG2DPolygonWithArcs>* polygon = new KGPlanarArea<KG2DPolygonWithArcs>(*bare_polygon);

    const KG2DPolygonWithArcs* ptr = dynamic_cast<const KG2DPolygonWithArcs*>(polygon->GetPlanarObject());

    if (ptr->IsSimple()) {
        std::cout << "Polygon is simple." << std::endl;
    }
    else {
        std::cout << "Polygon is not simple." << std::endl;
    }


    Double_t limx = 20;
    Double_t limy = 20;
    Double_t limz = 20;

    //root app
    TApplication* App = new TApplication("test polygon in 3d", &argc, argv);

    //make a canvas
    TCanvas* Canvas = new TCanvas("test polygon in 3d", "test polygon in 3d", 100, 100, 800, 800);

    //make a view
    Double_t MinRange[3] = {-1 * limx, -1 * limy, -1 * limz};
    Double_t MaxRange[3] = {limx, limy, limz};
    TView3D* View = new TView3D(1, MinRange, MaxRange);

    //the intersection points
    TPolyMarker3D* InterPoints = new TPolyMarker3D();  //panel 1
    UInt_t InterCount = 0;

    //lets transform the polygon so it no longer lies in the xy-plane
    //    KGTransformation* trans = new KGTransformation();
    //    trans->SetRotationConstructive(35., 21., 5.);
    //    polygon->Transform(trans);

    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 100000;
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

        //std::cout<<"flag0"<<std::endl;
        Bool_t result = false;
        polygon->NearestIntersection(temp1, temp2, result, inter);

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
    InterPoints->SetMarkerStyle(kCircle);
    InterPoints->Draw();

    App->Run();
}
