#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "TRandom3.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "TMath.h"
#include "KTwoVector.h"
#include "KThreeVector.h"
#include "TView3D.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"


#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DPolygonWithArcs.hh"
#include "KGPlanarArea.hh"
#include "KGExtrudedSurface.hh"


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{

    //the vertices
    const Int_t N = 7;
    Double_t v[7][2];
    Double_t sv[7][2];

    v[0][0] = 2.5; v[0][1] = 0.;
    v[1][0] = 0.5; v[1][1] = 5.0;
    v[2][0] = 5.5; v[2][1] = 9.0;
    v[3][0] = 0.0; v[3][1] = 17.;
    v[4][0] = -5.5; v[4][1] = 9.0;
    v[5][0] = -0.5; v[5][1] = 5.0;
    v[6][0] = -2.5; v[6][1] = 0.;

    Double_t r1 = 4.5;
    Double_t r2 = 3.21;
    Double_t r3 = 19.5;

    Double_t scale_factor = 0.4;
    for(Int_t i=0; i<N; i++)
    {
        sv[i][0] = scale_factor*v[i][0];
        sv[i][1] = scale_factor*v[i][1] +6.0;
    }

    KGExtrudedSurface* exSurf = new KGExtrudedSurface(0.0, -10.0, 5.0, 5, true, true);

    exSurf->AddOuterArc(v[0], v[1], r1, false);
    exSurf->AddOuterArc(v[1], v[2], r2, true);
    exSurf->AddOuterArc(v[2], v[3], r3, false);
    exSurf->AddOuterArc(v[3], v[4], r3, false);
    exSurf->AddOuterArc(v[4], v[5], r2, true);
    exSurf->AddOuterArc(v[5], v[6], r1, false);
    exSurf->AddOuterLine(v[6], v[0]);

    exSurf->AddInnerArc(sv[0], sv[1], scale_factor*r1, false);
    exSurf->AddInnerArc(sv[1], sv[2], scale_factor*r2, true);
    exSurf->AddInnerArc(sv[2], sv[3], scale_factor*r3, false);
    exSurf->AddInnerArc(sv[3], sv[4], scale_factor*r3, false);
    exSurf->AddInnerArc(sv[4], sv[5], scale_factor*r2, true);
    exSurf->AddInnerArc(sv[5], sv[6], scale_factor*r1, false);
    exSurf->AddInnerLine(sv[6], sv[0]);

//    exSurf->AddOuterLine(v[0], v[1]);
//    exSurf->AddOuterLine(v[1], v[2]);
//    exSurf->AddOuterLine(v[2], v[3]);
//    exSurf->AddOuterLine(v[3], v[4]);
//    exSurf->AddOuterLine(v[4], v[5]);
//    exSurf->AddOuterLine(v[5], v[6]);
//    exSurf->AddOuterLine(v[6], v[0]);

//    exSurf->AddInnerLine(sv[0], sv[1]);
//    exSurf->AddInnerLine(sv[1], sv[2]);
//    exSurf->AddInnerLine(sv[2], sv[3]);
//    exSurf->AddInnerLine(sv[3], sv[4]);
//    exSurf->AddInnerLine(sv[4], sv[5]);
//    exSurf->AddInnerLine(sv[5], sv[6]);
//    exSurf->AddInnerLine(sv[6], sv[0]);

    exSurf->Initialize();

    Double_t limx = 20;
    Double_t limy = 20;
    Double_t limz = 20;

    //root app
    TApplication* App = new TApplication("test polygon in 3d",&argc,argv);

    //make a canvas
    TCanvas* Canvas = new TCanvas("test polygon in 3d","test polygon in 3d",100,100,800,800);

    //make a view
    Double_t MinRange[3] = {-1*limx, -1*limy, -1*limz};
    Double_t MaxRange[3] = {limx, limy, limz};
    TView3D* View = new TView3D(1, MinRange, MaxRange);

    //the intersection points
    TPolyMarker3D* InterPoints = new TPolyMarker3D(); //panel 1
    UInt_t InterCount = 0;

//    //lets transform the extruded surf so it no longer lies perp to xy-plane
//    KGTransformation* trans = new KGTransformation();
//    trans->SetRotationConstructive(35., 21., 5.);
//    exSurf->Transform(trans);

    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 1000000;
    Double_t x1,y1,z1,x2,y2,z2;
    katrin::KThreeVector temp1;
    katrin::KThreeVector temp2;
    TLine** moreLines = new TLine*[NLines];
    TRandom3* rand = new TRandom3(0);
    for(UInt_t i=0; i<NLines; i++)
    {
        x1 = rand->Uniform(-1.0*limx, limx);
        y1 = rand->Uniform(-1.0*limy, limy);
        z1 = rand->Uniform(-1.0*limz, limz);
        x2 = rand->Uniform(-1.0*limx, limx);
        y2 = rand->Uniform(-1.0*limy, limy);
        z2 = rand->Uniform(-1.0*limz, limz);

        temp1 = katrin::KThreeVector(x1,y1,z1);
        temp2 = katrin::KThreeVector(x2,y2,z2);
        katrin::KThreeVector inter;

        Bool_t result = false;
        exSurf->NearestIntersection(temp1, temp2, result, inter);

        if(result == true)
        {
            InterPoints->SetPoint(InterCount, inter.X(), inter.Y(), inter.Z());
            InterCount++;
        }

      }

    std::cout<<"number of intersections = "<<InterCount<<std::endl;


    Double_t xarrx[2] = {0,10};
    Double_t xarry[2] = {0,0};
    Double_t xarrz[2] = {0,0};
    Double_t yarrx[2] = {0,0};
    Double_t yarry[2] = {0,10};
    Double_t yarrz[2] = {0,0};
    Double_t zarrx[2] = {0,0};
    Double_t zarry[2] = {0,0};
    Double_t zarrz[2] = {0,10};
    
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
