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
#include "KGTransformation.hh"
#include "KGExtrudedSurface.hh"
#include "KGBeam.hh"
#include "KGWireArray.hh"


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{

    if (argc < 2)
    {
        std::cout<<""<<std::endl;
        std::cout<<"Usage: TestWireArrayIntersection <n wires>"<<std::endl;
        std::cout<<""<<std::endl;
        return 1;
    }

    std::string num_sides(argv[1]);
    std::stringstream ss;
    ss.clear();
    ss << num_sides;
    Int_t N;
    ss >> N;


    //diameter
    Double_t dia = 0.05;

    //centers
    Double_t cen_a[3];
    cen_a[0] = 0.; cen_a[1] = 0.; cen_a[0] = 0;
    Double_t cen_b[3];
    cen_b[0] = 0.; cen_b[1] = 0.; cen_b[0] = 1;

    //start wire
    Double_t fwa[3];
    Double_t fwb[3];
    fwa[0] = 0; fwa[1] = 0; fwa[2] = 0;
    fwb[0] = 0; fwb[1] = 0; fwb[2] = 1;

    //end wire
    Double_t ewa[3];
    Double_t ewb[3];
    ewa[0] = 0; ewa[1] = 1; ewa[2] = 0;
    ewb[0] = 0; ewb[1] = 1; ewb[2] = 1;

    KGWireArray* Warr = new KGWireArray(0, dia, 5, 2, N, cen_a, cen_b, fwa, fwb, ewa, ewb);

    std::cout<<"flag0"<<std::endl;

    Double_t limx = 10;
    Double_t limy = 10;
    Double_t limz = 10;



    //root app
    TApplication* App = new TApplication("test beam in 3d",&argc,argv);

    //make a canvas
    TCanvas* Canvas = new TCanvas("test beam in 3d","test beam in 3d",100,100,800,800);

    //make a view
    Double_t MinRange[3] = {-1*limx, -1*limy, -1*limz};
    Double_t MaxRange[3] = {limx, limy, limz};
    TView3D* View = new TView3D(1, MinRange, MaxRange);

    //the intersection points
    TPolyMarker3D* InterPoints = new TPolyMarker3D(); //panel 1
    UInt_t InterCount = 0;

   // lets transform the beam so it no longer lies perp to xy-plane
    KGTransformation* trans = new KGTransformation();
    trans->SetRotationConstructive(35., 21., 5.);
    trans->SetDisplacement(katrin::KThreeVector(4,5,0));
    Warr->Transform(trans);

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
        Warr->Intersection(temp1, temp2, result, inter);

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
