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


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{



    katrin::KTwoVector p1(-1.2, -0.98);
    katrin::KTwoVector p2(1.3, 0.756);

    KG2DLineSegment* line = new KG2DLineSegment();
    line->SetPoints(p1, p2);

    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR",&argc,argv);

    Double_t lim = 2.0;

    TH2D* inside = new TH2D("nearest","nearest", 100, -1*lim, lim, 100, -1.0*lim, lim);
    TH2D* intersect = new TH2D("inter","inter", 100, -1*lim, lim, 100, -1.0*lim, lim);

    TCanvas* canvas = new TCanvas("Line","Line Test", 50, 50, 600, 600);
    canvas->SetFillColor(0);
    canvas->SetBorderSize(0);
    canvas->SetRightMargin(0.2);
    inside->Draw("A");

    //now lets test some points to see if the nearestpoint routine works
    UInt_t NPoints = 500;
    Double_t x,y;
    katrin::KTwoVector temp1;
    katrin::KTwoVector temp2;
    TLine** l = new TLine*[NPoints];
    TRandom3* rand = new TRandom3();
    for(UInt_t i=0; i<NPoints; i++)
    {
        x = rand->Uniform(-1.0*lim, lim);
        y = rand->Uniform(-1.0*lim, lim);
        temp1 = katrin::KTwoVector(x,y);

        line->NearestPoint(temp1, temp2);
        inside->Fill(temp2.X() , temp2.Y() );
        l[i] = new TLine(x,y, temp2.X(), temp2.Y());
        l[i]->Draw("SAME");


    }
    inside->Draw("SAME SCAT");

    //now lets test some lines to see if the intersection routine works
    Int_t NLines = 100;
    Double_t x1,y1,x2,y2;
    TLine** moreLines = new TLine*[NLines];
    for(Int_t i=0; i<NLines; i++)
    {
        x1 = rand->Uniform(-1.0*lim, lim);
        y1 = rand->Uniform(-1.0*lim, lim);

        x2 = rand->Uniform(-1.0*lim, lim);
        y2 = rand->Uniform(-1.0*lim, lim);
        temp1 = katrin::KTwoVector(x1,y1);
        temp2 = katrin::KTwoVector(x2,y2);
        katrin::KTwoVector inter;
        Double_t dist;

        //std::cout<<"flag0"<<std::endl;
        Bool_t result = false;
        line->NearestIntersection(temp1, temp2, result, inter);

        if(result == true)
        {
            moreLines[i] = new TLine(temp1.X(),temp1.Y(),temp2.X(), temp2.Y());
            moreLines[i]->SetLineColor(2);
            intersect->Fill(inter.X(), inter.Y());
            //arc->NearestDistance(inter, dist);
            //std::cout<<"intersection is "<<dist<<" away."<<std::endl;
            moreLines[i]->Draw("SAME");

        }
        else
        {   
            moreLines[i] = new TLine(temp1.X(),temp1.Y(),temp2.X(), temp2.Y());
            moreLines[i]->SetLineColor(4);
            moreLines[i]->Draw("SAME");
        }
      }

    intersect->SetMarkerColor(kRed);
    intersect->SetMarkerStyle(kCircle);
    intersect->Draw("SAME");

    App->Run();

}
