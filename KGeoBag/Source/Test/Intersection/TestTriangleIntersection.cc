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

#include "KGTriangle.hh"

using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{

    KGTriangle* tri1 = new KGTriangle(8,4,4);

    katrin::KThreeVector v1(5,0,0);
    katrin::KThreeVector v2(0,5,0);
    katrin::KThreeVector v3(0,0,5);

    KGTriangle* tri2 = new KGTriangle(v1,v2,v3);

    Double_t limx = 20;
    Double_t limy = 20;
    Double_t limz = 20;

    //root app
    TApplication* App = new TApplication("test triangle in 3d",&argc,argv);

    //make a canvas
    TCanvas* Canvas = new TCanvas("test triangle in 3d","test triangle in 3d",100,100,800,800);

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
//    tri1->Transform(trans);

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
        tri1->NearestIntersection(temp1, temp2, result, inter);

        if(result == true)
        {
            InterPoints->SetPoint(InterCount, inter.X(), inter.Y(), inter.Z());
            InterCount++;
        }


//        tri2->NearestIntersection(temp1, temp2, result, inter);

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
