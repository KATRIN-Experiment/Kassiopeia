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

#include "KG2DLineSegment.hh"
#include "KG2DArc.hh"
#include "KG2DPolygon.hh"

using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{


    Double_t R = 3.21;;
    Double_t angle = 15.*(TMath::Pi()/180.);
//    katrin::KTwoVector p1(1.0, 0.0);
//    katrin::KTwoVector p2(-1., 0.);
      katrin::KTwoVector p1(5.0, 0.5);
      katrin::KTwoVector p2(9.0, 5.5);

    katrin::KTwoVector center(0.,0.);

    KG2DArc* arc = new KG2DArc();
    Bool_t isRight = true;
    Bool_t isCCW = true;

    //arc->SetCenterRadiusAngles(center, R, 0, (TMath::Pi() )/2.0 );
    //arc->SetStartPointCenterAngle(center, p1, -5.0*(TMath::Pi()/4.0) );
    arc->SetPointsRadiusOrientation(p1, p2, R, isRight, isCCW);

    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR",&argc,argv);

    Double_t lim = 10.0;

    TH2D* inside = new TH2D("inside","inside", 400, -1*lim, lim, 400, -1.0*lim, lim);
    TH2D* intersect = new TH2D("inter","inter", 400, -1*lim, lim, 400, -1.0*lim, lim);

    TCanvas* canvas = new TCanvas("Arc","Arc Test", 50, 50, 600, 600);
    canvas->SetFillColor(0);
    canvas->SetBorderSize(0);
    canvas->SetRightMargin(0.2);
    inside->Draw("A");

    //now lets test some points to see if the in/out routine works
    UInt_t NPoints = 100000;
    Double_t x,y;
    katrin::KTwoVector temp;
    TRandom3* rand = new TRandom3(0);
    for(UInt_t i=0; i<NPoints; i++)
    {
        x = rand->Uniform(-1.0*lim, lim);
        y = rand->Uniform(-1.0*lim, lim);
        temp.SetComponents(x,y);

        //std::cout<<"flag0"<<std::endl;

        if(arc->IsInsideCircularSegment(temp))
        {
            inside->Fill(x,y);
        }
    }
    inside->Draw("SAME SCAT");

//    //now lets test some points to see if the nearestpoint routine works
//    UInt_t NPoints = 2000;
//    Double_t x,y;
//    katrin::KTwoVector temp, temp1;
//    katrin::KTwoVector temp2;
//    TLine** l = new TLine*[NPoints];
//    TRandom3* rand = new TRandom3(0);
//    for(UInt_t i=0; i<NPoints; i++)
//    {
//        x = rand->Uniform(-1.0*lim, lim);
//        y = rand->Uniform(-1.0*lim, lim);
//        temp1 = katrin::KTwoVector(x,y);

//        arc->NearestPoint(temp1, temp2);
//        inside->Fill(temp2.X() , temp2.Y() );
//        l[i] = new TLine(x,y, temp2.X(), temp2.Y());
//        l[i]->Draw("SAME");


//    }
//    inside->Draw("SAME SCAT");

//    //now lets test some lines to see if the intersection routine works
//    Int_t NLines = 300;
//    Double_t x1,y1,x2,y2;
//    katrin::KTwoVector temp1;
//    katrin::KTwoVector temp2;
//    TLine** moreLines = new TLine*[NLines];
//    for(UInt_t i=0; i<NLines; i++)
//    {
//        x1 = rand->Uniform(-1.0*lim, lim);
//        y1 = rand->Uniform(-1.0*lim, lim);

//        x2 = rand->Uniform(-1.0*lim, lim);
//        y2 = rand->Uniform(-1.0*lim, lim);
//        temp1.SetComponents(x1,y1);
//        temp2.SetComponents(x2,y2);
//        katrin::KTwoVector inter;
//        Double_t dist;

//        //std::cout<<"flag0"<<std::endl;
//        Bool_t result = false;
//        arc->NearestIntersection(temp1, temp2, result, inter);

//        if(result == true)
//        {
//            moreLines[i] = new TLine(temp1.X(),temp1.Y(),temp2.X(), temp2.Y());
//            moreLines[i]->SetLineColor(2);
//            intersect->Fill(inter.X(), inter.Y());
//            arc->NearestDistance(inter, dist);
//            std::cout<<"intersection is "<<dist<<" away."<<std::endl;
//            moreLines[i]->Draw("SAME");

//        }
//        else
//        {
//            moreLines[i] = new TLine(temp1.X(),temp1.Y(),temp2.X(), temp2.Y());
//            moreLines[i]->SetLineColor(4);
//            moreLines[i]->Draw("SAME");
//        }
//      }

//    intersect->SetMarkerColor(kRed);
//    intersect->SetMarkerStyle(kCircle);
//    intersect->Draw("SAME");

    App->Run();

}
