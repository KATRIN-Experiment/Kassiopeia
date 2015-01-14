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
#include "KGRotatedSurface.hh"


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{

    //the vertices
    const Int_t N = 7;
    Double_t v[7][2];
    Double_t ev[4][2];

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

    //shift all points away from the axis by 10;
    for(Int_t i=0; i<N; i++)
    {
        v[i][0] += 10.;
    }

    KGRotatedSurface* rotSurf = new KGRotatedSurface(10,10);
    rotSurf->SetClosed();


    rotSurf->AddArc(v[0], v[1], r1, false);
    rotSurf->AddArc(v[1], v[2], r2, true);

    //changed these sides into cone sections to test feature
    rotSurf->AddArc(v[2], v[3], r3, false);
    rotSurf->AddArc(v[3], v[4], r3, false);
//    rotSurf->AddLine(v[2], v[3]);
//    rotSurf->AddLine(v[3], v[4]);

    rotSurf->AddArc(v[4], v[5], r2, true);
    rotSurf->AddArc(v[5], v[6], r1, false);
    rotSurf->AddLine(v[6], v[0]);
    rotSurf->Initialize();


    ev[0][0] = 0.0; ev[0][1] = -17;
    ev[1][0] = 17; ev[1][1] = 0.0;
    ev[2][0] = 17; ev[2][1] = 17;
    ev[3][0] = 0.0; ev[3][1] = 34;

    Double_t r4 = 17;
    KGRotatedSurface* rotSurf2 = new KGRotatedSurface(10,10);
    rotSurf2->SetOpen();

    rotSurf2->AddArc(ev[0], ev[1], r4, true);
    rotSurf2->AddLine(ev[1], ev[2]);
    rotSurf2->AddArc(ev[2], ev[3], r4, true);

    //need this extra 'line' to end the polyline...hmmm, might want to re-think this
    rotSurf2->AddLine(ev[3], ev[3]); 


    rotSurf2->Initialize();
    
    Double_t limx = 40;
    Double_t limy = 40;
    Double_t limz = 40;

    //root app
    TApplication* App = new TApplication("test rotated surf",&argc,argv);

    //make a canvas
    TCanvas* Canvas = new TCanvas("test rotated surf","test rotated surf",100,100,800,800);

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
//    trans->SetDisplacement(katrin::KThreeVector(5.2,0.,1.1));
//    rotSurf->Transform(trans);
//    rotSurf2->Transform(trans);

    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 100000;
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
        rotSurf->NearestIntersection(temp1, temp2, result, inter);

        if(result == true)
        {
            InterPoints->SetPoint(InterCount, inter.X(), inter.Y(), inter.Z());
            InterCount++;
        }

//        result = false;
//        rotSurf2->NearestIntersection(temp1, temp2, result, inter);

//        if(result == true)
//        {
//            InterPoints->SetPoint(InterCount, inter.X(), inter.Y(), inter.Z());
//            InterCount++;
//        }

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
