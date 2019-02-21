#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#ifdef KEMFIELD_USE_GSL
#include <gsl/gsl_rng.h>
#endif

#include "KSurfaceTypes.hh"
#include "KSurface.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#include "KSurfaceContainer.hh"
#include "KBoundaryIntegralMatrix.hh"

#include "KEMConstants.hh"

#include "KFMPoint.hh"
#include "KFMPointCloud.hh"

#include "KThreeVector_KEMField.hh"

#include "KVMPathIntegral.hh"
#include "KVMLineIntegral.hh"
#include "KVMSurfaceIntegral.hh"
#include "KVMFluxIntegral.hh"

#include "KVMField.hh"
#include "KVMFieldWrapper.hh"

#include "KVMLineSegment.hh"
#include "KVMTriangularSurface.hh"
#include "KVMRectangularSurface.hh"

#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"
#include "KFMElectrostaticMultipoleCalculatorNumeric.hh"

#include "KSAStructuredASCIIHeaders.hh"



#ifdef KEMFIELD_USE_ROOT
#include "TCanvas.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "TMath.h"
#include "TVector2.h"
#include "TVector3.h"
#include "TView3D.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "TRandom3.h"
#include "TH2D.h"
#include "TApplication.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TVector3.h"
#endif

using namespace KEMField;


int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    #ifdef KEMFIELD_USE_GSL

    int NTriangles = 4000;

    //generate sets of three points in bounding sphere for triangles
    std::vector< std::vector< std::vector<double> > > TrianglePoints;
    std::vector< KFMPointCloud<3> > TrianglePointClouds;
    KFMPointCloud<3> TempCloud;
    KFMPoint<3> TempPoint;

    TrianglePoints.resize(NTriangles);

    for(int i=0; i<NTriangles; i++)
    {
        TrianglePoints[i].resize(3);
        for(int j = 0; j<3; j++)
        {
            TrianglePoints[i][j].resize(3);
        }
    }


	const gsl_rng_type* T;
	gsl_rng * r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);

    double RMin = 1.0;
    double x,y,z;
    double p0[3];
    double p1[3];
    double p2[3];
    // bool isAcute;
    double a,b,c;
    double delx, dely, delz;
    double min, max;
    // double mid;

    for(int i=0; i<NTriangles; i++)
    {
        // isAcute = false;
        do
        {

            for(int j = 0; j<3; j++)
            {
                do
                {
                    x = -1.0*RMin + gsl_rng_uniform(r)*(2.0*RMin);
                    y = -1.0*RMin + gsl_rng_uniform(r)*(2.0*RMin);
                    z = -1.0*RMin + gsl_rng_uniform(r)*(2.0*RMin);
                }
                while(x*x + y*y + z*z > RMin*RMin);

                //normalize so that the vertices lie on the units sphere's surface

                double norm = std::sqrt(x*x + y*y + z*z);

                TrianglePoints[i][j][0] = x/norm;
                TrianglePoints[i][j][1] = y/norm;
                TrianglePoints[i][j][2] = z/norm;

            }


            TempCloud.Clear();

            p0[0] = TrianglePoints[i][0][0];
            p0[1] = TrianglePoints[i][0][1];
            p0[2] = TrianglePoints[i][0][2];

            TempPoint[0] =  p0[0];
            TempPoint[1] =  p0[1];
            TempPoint[2] =  p0[2];
            TempCloud.AddPoint(TempPoint);

            p1[0] = TrianglePoints[i][1][0];
            p1[1] = TrianglePoints[i][1][1];
            p1[2] = TrianglePoints[i][1][2];

            TempPoint[0] =  p1[0];
            TempPoint[1] =  p1[1];
            TempPoint[2] =  p1[2];
            TempCloud.AddPoint(TempPoint);

            p2[0] = TrianglePoints[i][2][0];
            p2[1] = TrianglePoints[i][2][1];
            p2[2] = TrianglePoints[i][2][2];

            TempPoint[0] =  p2[0];
            TempPoint[1] =  p2[1];
            TempPoint[2] =  p2[2];
            TempCloud.AddPoint(TempPoint);

            delx = TrianglePoints[i][1][0] - TrianglePoints[i][0][0];
            dely = TrianglePoints[i][1][1] - TrianglePoints[i][0][1];
            delz = TrianglePoints[i][1][2] - TrianglePoints[i][0][2];
            a = std::sqrt(delx*delx + dely*dely + delz*delz);

            delx = TrianglePoints[i][2][0] - TrianglePoints[i][0][0];
            dely = TrianglePoints[i][2][1] - TrianglePoints[i][0][1];
            delz = TrianglePoints[i][2][2] - TrianglePoints[i][0][2];
            b = std::sqrt(delx*delx + dely*dely + delz*delz);

            delx = TrianglePoints[i][2][0] - TrianglePoints[i][1][0];
            dely = TrianglePoints[i][2][1] - TrianglePoints[i][1][1];
            delz = TrianglePoints[i][2][2] - TrianglePoints[i][1][2];
            c = std::sqrt(delx*delx + dely*dely + delz*delz);

            //sort the side lengths according to size
            if(a < b)
            {
                max = b;
                // mid = b;
                min = a;
            }
            else
            {
                max = a;
                // mid = a;
                min = b;
            }

            if(c < min)
            {
                // mid = min;
                min = c;
            }
            else
            {
                if(c > max)
                {
                    max = c;
                }
                else
                {
                    // mid = c;
                }
            }

//            double hypot = std::sqrt( min*min + mid*mid);
//            if( hypot <= max)
//            {
//                isAcute = true;
//            }

        }
        while(false);
//        while(!isAcute);

        TrianglePointClouds.push_back(TempCloud);

    }



    typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>
    KEMTriangle;

    KSurfaceContainer sC;

    //make an array of KTTriangleElectrode's
    std::vector< KEMTriangle* > Triangles;
    Triangles.resize(NTriangles);

    double n1[3];
    double n2[3];
    double U = 1.0;

    int count = 0;
    for(int i=0; i<NTriangles; i++)
    {

        p0[0] = TrianglePoints[i][0][0];
        p0[1] = TrianglePoints[i][0][1];
        p0[2] = TrianglePoints[i][0][2];

        n1[0] = TrianglePoints[i][1][0] - p0[0];
        n1[1] = TrianglePoints[i][1][1] - p0[1];
        n1[2] = TrianglePoints[i][1][2] - p0[2];

        n2[0] = TrianglePoints[i][2][0] - p0[0];
        n2[1] = TrianglePoints[i][2][1] - p0[1];
        n2[2] = TrianglePoints[i][2][2] - p0[2];

        a = std::sqrt( n1[0]*n1[0] +  n1[1]*n1[1] + n1[2]*n1[2] );
        b = std::sqrt( n2[0]*n2[0] +  n2[1]*n2[1] + n2[2]*n2[2] );

        for(int j = 0; j<3; j++)
        {
            n1[j] *= 1.0/a;
            n2[j] *= 1.0/b;
        }

        Triangles[count] = new KEMTriangle();
        Triangles[count]->SetA(a);
        Triangles[count]->SetB(b);
        Triangles[count]->SetP0(KPosition(p0[0],p0[1],p0[2]));
        Triangles[count]->SetN1(KDirection(n1[0],n1[1],n1[2]));
        Triangles[count]->SetN2(KDirection(n2[0],n2[1],n2[2]));
        Triangles[count]->SetBoundaryValue(U);

        sC.push_back(Triangles[count]);

        count++;

    }


  KElectrostaticBoundaryIntegrator integrator {KEBIFactory::MakeDefault()};
  KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(sC,integrator);

    //normalize charge density on all triangles so they have unit potential
    KPosition centroid;

    count = 0;
    for(int i=0; i<NTriangles; i++)
    {
        centroid = Triangles[count]->Centroid();
        double pot = (integrator.Potential(Triangles[count], centroid));
        Triangles[count]->SetSolution(1.0/pot);
        pot =  (integrator.Potential(Triangles[count], centroid))*(integrator.BasisValue(Triangles[count],0));
        count++;
    }


    //now lets make the multipole calculators
    KFMElectrostaticMultipoleCalculatorAnalytic* aCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();

    KFMElectrostaticMultipoleCalculatorNumeric* n2Calc = new KFMElectrostaticMultipoleCalculatorNumeric();
    KFMElectrostaticMultipoleCalculatorNumeric* n4Calc = new KFMElectrostaticMultipoleCalculatorNumeric();
    KFMElectrostaticMultipoleCalculatorNumeric* n6Calc = new KFMElectrostaticMultipoleCalculatorNumeric();
    KFMElectrostaticMultipoleCalculatorNumeric* n8Calc = new KFMElectrostaticMultipoleCalculatorNumeric();
    KFMElectrostaticMultipoleCalculatorNumeric* n10Calc = new KFMElectrostaticMultipoleCalculatorNumeric();
    KFMElectrostaticMultipoleCalculatorNumeric* n12Calc = new KFMElectrostaticMultipoleCalculatorNumeric();

    n2Calc->SetNumberOfQuadratureTerms(2);
    n4Calc->SetNumberOfQuadratureTerms(4);
    n6Calc->SetNumberOfQuadratureTerms(6);
    n8Calc->SetNumberOfQuadratureTerms(8);
    n10Calc->SetNumberOfQuadratureTerms(10);
    n12Calc->SetNumberOfQuadratureTerms(12);


    const unsigned int n_degree = 17;

    double degree[n_degree];
    degree[0] = 1;
    degree[1] = 2;
    degree[2] = 4;
    degree[3] = 6;
    degree[4] = 8;
    degree[5] = 10;
    degree[6] = 12;
    degree[7] = 14;
    degree[8] = 16;
    degree[9] = 18;
    degree[10] = 20;
    degree[11] = 22;
    degree[12] = 24;
    degree[13] = 26;
    degree[14] = 28;
    degree[15] = 30;
    degree[16] = 32;


    //storage for the times
    double aTime[n_degree];
    double n4Time[n_degree];
    double n6Time[n_degree];
    double n8Time[n_degree];
    double n10Time[n_degree];



    for(unsigned int i=0; i < n_degree; i++)
    {
        std::cout<<"working on degree = "<<degree[i]<<std::endl;
        aCalc->SetDegree(degree[i]);
        n4Calc->SetDegree(degree[i]);
        n6Calc->SetDegree(degree[i]);
        n6Calc->SetDegree(degree[i]);
        n8Calc->SetDegree(degree[i]);
        n10Calc->SetDegree(degree[i]);


        //the origin we want to expand about is (0,0,0)
        double origin[3];
        origin[0] = 0.;
        origin[1] = 0.;
        origin[2] = 0.;

        //compute multipole moments for each electrode
        KFMScalarMultipoleExpansion expan;
        expan.SetDegree(degree[i]);

        clock_t start, end;
        start = clock();

        int n_expan;
        if(degree[i] < 8)
        {
            n_expan = NTriangles;
        }
        else if(degree[i] < 16)
        {
            n_expan = 1000;
        }
        else if(degree[i] < 24)
        {
            n_expan = 400;
        }
        else
        {
            n_expan = 100;
        }

        for(int j=0; j<n_expan; j++)
        {
            aCalc->ConstructExpansion(origin, &(TrianglePointClouds[j]), &expan);
        }

        end = clock();
        aTime[i] = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
        aTime[i] /= (double)n_expan;

        start = clock();


//        for(int j=0; j<n_expan; j++)
//        {
//            n2Calc->ConstructExpansion(origin, &(TrianglePointClouds[j]), &expan);
//        }

//        end = clock();
//        n2Time[i] = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
//        n2Time[i] /= (double)n_expan;


        for(int j=0; j<n_expan; j++)
        {
            n4Calc->ConstructExpansion(origin, &(TrianglePointClouds[j]), &expan);
        }

        end = clock();
        n4Time[i] = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
        n4Time[i] /= (double)n_expan;

        for(int j=0; j<n_expan; j++)
        {
            n6Calc->ConstructExpansion(origin, &(TrianglePointClouds[j]), &expan);
        }

        end = clock();
        n6Time[i] = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
        n6Time[i] /= (double)n_expan;

        for(int j=0; j<n_expan; j++)
        {
            n8Calc->ConstructExpansion(origin, &(TrianglePointClouds[j]), &expan);
        }

        end = clock();
        n8Time[i] = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
        n8Time[i] /= (double)n_expan;


        for(int j=0; j<n_expan; j++)
        {
            n10Calc->ConstructExpansion(origin, &(TrianglePointClouds[j]), &expan);
        }

        end = clock();
        n10Time[i] = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
        n10Time[i] /= (double)n_expan;


        std::cout<<"Number of triangles = "<<n_expan<<std::endl;
        std::cout<<"Degree of expansion = "<<degree[i]<<std::endl;
    }


    #ifdef KEMFIELD_USE_ROOT





    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR",&argc,argv);

    TStyle* myStyle = new TStyle("Plain", "Plain");
    myStyle->SetCanvasBorderMode(0);
    myStyle->SetPadBorderMode(0);
    myStyle->SetPadColor(0);
    myStyle->SetCanvasColor(0);
    myStyle->SetTitleColor(1);
    myStyle->SetPalette(1,0);   // nice color scale for z-axis
    myStyle->SetCanvasBorderMode(0); // gets rid of the stupid raised edge around the canvas
    myStyle->SetTitleFillColor(0); //turns the default dove-grey background to white
    myStyle->SetCanvasColor(0);
    myStyle->SetPadColor(0);
    myStyle->SetTitleFillColor(0);
    myStyle->SetStatColor(0); //this one may not work
    const int NRGBs = 5;
    const int NCont = 48;
    double stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    double red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    double green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    double blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    myStyle->SetNumberContours(NCont);
    myStyle->cd();

    TCanvas *c1 = new TCanvas("c1","speed",200,10,700,500);

    std::vector<TGraph*> graphs;
    graphs.resize(5);

    graphs[0] = new TGraph();//"Analytic Multipole expansion" , "Analytic Multipole expansion");
    graphs[1] = new TGraph();//"Gaussian Quadrature n=4", "Gaussian Quadrature n=4");
    graphs[2] = new TGraph();//"Gaussian Quadrature n=6", "Gaussian Quadrature n=6");
    graphs[3] = new TGraph();//"Gaussian Quadrature n=8" , "Gaussian Quadrature n=8");
    graphs[4] = new TGraph();//"Gaussian Quadrature n=10" , "Gaussian Quadrature n=10");
//    graphs[5] = new TGraph();//"Gaussian Quadrature n=12" , "Gaussian Quadrature n=12");

    for(unsigned int k=0; k<n_degree; k++)
    {
        graphs[0]->SetPoint(k,degree[k], aTime[k]);
        graphs[1]->SetPoint(k,degree[k], n4Time[k]);
        graphs[2]->SetPoint(k,degree[k], n6Time[k]);
        graphs[3]->SetPoint(k,degree[k], n8Time[k]);
        graphs[4]->SetPoint(k,degree[k], n10Time[k]);
//        graphs[5]->SetPoint(k,degree[k], n12Time[k]);
    }

    int color = 0;
    int color_array[] = {1,2,3,4,95,6,7,8,9,13,46,51};

    for(unsigned int k=0; k<5; k++)
    {
        graphs[k]->SetMaximum(1.0);
        graphs[k]->SetMinimum(1e-5);
        //graphs[k]->SetStats(0);


        graphs[k]->SetMarkerColor(color_array[color]);
        graphs[k]->SetMarkerSize(1.3);
        graphs[k]->SetLineColor(color_array[color]);
        graphs[k]->SetMarkerStyle(20 + color);
        graphs[k]->SetLineStyle((color)%3 + 1);

        color++;
        if(k==0)
        {
            graphs[0]->Draw();
            std::cout<<"Drawing graph: "<<k<<std::endl;
            graphs[0]->GetXaxis()->SetTitle("Degree of Expansion");
            graphs[0]->GetXaxis()->CenterTitle();
            graphs[0]->GetXaxis()->SetTitleOffset(1.3);
            graphs[0]->GetYaxis()->SetTitle("#splitline{Time to calculate multipole coefficients}{             of a single triangle (s)}");
            graphs[0]->GetYaxis()->CenterTitle();
            graphs[0]->GetYaxis()->SetTitleOffset(1.3);
            graphs[0]->Draw("ALP");
        }
        else
        {
            graphs[k]->Draw("SAME LP");
            std::cout<<"Drawing graph: "<<k<<std::endl;
        }
    }

    c1->SetLogy();


    std::stringstream name;

	TLegend* key = new TLegend(0.65,0.2,0.85,0.4,"Method");
	key->SetTextSize(0.03);
	key->SetFillColor(0);

	key->AddEntry(graphs[0], "Analytic (this work)", "lp");
	//key->AddEntry(graphs[1], "Quadrature (n=2)", "lp");
	key->AddEntry(graphs[1], "Quadrature (4 #times 4)", "lp");
	key->AddEntry(graphs[2], "Quadrature (6 #times 6)", "lp");
	key->AddEntry(graphs[3], "Quadrature (8 #times 8)", "lp");
	key->AddEntry(graphs[4], "Quadrature (10 #times 10)", "lp");
//	key->AddEntry(graphs[5], "Quadrature (n=12)", "lp");


	key->Draw();
	c1->Update();

    App->Run();

    #endif

    #else
        std::cout<<"To use this test program, please recompile witht the flag KEMFIELD_USE_GSL enabled"<<std::endl;
    #endif











  return 0;
}
