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

#include "KEMThreeVector.hh"

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
    #include "TMath.h"
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




double PotentialMultipole(int degree, double* point, double* origin, std::vector< std::complex<double> >* moments)
{
    int si;
    double re, im, alp;
    std::complex<double> ylm, mom;
    std::complex<double> pMulti = std::complex<double>(0.0,0.0);
    double phi = KFMMath::Phi(origin, point);
    double costheta = KFMMath::CosTheta(origin, point);
    double radius = KFMMath::Radius(origin, point);

    for(int l = 0; l <= degree; l++)
    {
        for(int m = -1*l; m <= l; m++)
        {
            si = l*(l+1) + m;
            alp = KFMMath::ALP_nm(l,std::fabs(m),costheta);
            re = alp*std::cos(m*phi);
            im = alp*std::sin(m*phi);
            ylm = std::complex<double>(re,im);
            mom = moments->at(si);
            pMulti += (std::pow(radius, -1*l -1))*(ylm*mom);
        }
    }

    pMulti *= (1.0/(4.0*M_PI*KEMConstants::Eps0));
    return pMulti.real();
}

#ifdef KEMFIELD_USE_ROOT

void BinLogX(TH1* h)
{

   TAxis *axis = h->GetXaxis();
   int bins = axis->GetNbins();

   Axis_t from = axis->GetXmin();
   Axis_t to = axis->GetXmax();
   Axis_t width = (to - from) / bins;
   Axis_t* new_bins = new Axis_t[bins + 1];

   for (int i = 0; i <= bins; i++)
   {
     new_bins[i] = TMath::Power(10, from + i * width);
   }
   axis->Set(bins, new_bins);
   delete[] new_bins;
}

#endif


namespace KEMField{

class ConfigureTestSingleTriangleMultipole: public KSAInputOutputObject
{
    public:

        ConfigureTestSingleTriangleMultipole()
        {
            fUseAnalyticCalculator = 0;
            fNTriangles = 0;
            fNSamplePointsPerTriangle = 0;
            fTriangleBoundingRadius = 1;
            fMaxSampleRadius = 10;
            fNQuadratureTerms = 0;
            fDegree.clear();
        }

        virtual ~ConfigureTestSingleTriangleMultipole(){;};

        virtual const char* GetName() const {return "ConfigureTestSingleTriangleMultipole"; };

        int GetUseAnalyticCalculator() const {return fUseAnalyticCalculator;};
        void SetUseAnalyticCalculator(const int& s){fUseAnalyticCalculator = s;};

        const std::vector<int>* GetDegree() const
        {
            return &fDegree;
        }
        void SetDegree(const std::vector<int>* deg){fDegree = *deg;};

        int GetNTriangles() const {return fNTriangles;};
        void SetNTriangles(const int& s){fNTriangles = std::fabs(s);};

        int GetNSamplePointsPerTriangle() const {return fNSamplePointsPerTriangle;};
        void SetNSamplePointsPerTriangle(const int& n){fNSamplePointsPerTriangle = std::fabs(n);};

        double GetTriangleBoundingRadius() const {return fTriangleBoundingRadius;};
        void SetTriangleBoundingRadius(const double& d){fTriangleBoundingRadius = std::fabs(d);};

        double GetMaxSampleRadius() const {return fMaxSampleRadius;};
        void SetMaxSampleRadius(const double& r){fMaxSampleRadius = std::fabs(r);};

        int GetNQuadratureTerms() const {return fNQuadratureTerms;};
        void SetNQuadratureTerms(const int& r){fNQuadratureTerms = std::fabs(r);};

        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(ConfigureTestSingleTriangleMultipole,UseAnalyticCalculator, int);
            AddKSAOutputFor(ConfigureTestSingleTriangleMultipole,NTriangles, int);
            AddKSAOutputFor(ConfigureTestSingleTriangleMultipole,NSamplePointsPerTriangle,int);
            AddKSAOutputFor(ConfigureTestSingleTriangleMultipole,TriangleBoundingRadius,double);
            AddKSAOutputFor(ConfigureTestSingleTriangleMultipole,MaxSampleRadius,double);
            AddKSAOutputFor(ConfigureTestSingleTriangleMultipole,NQuadratureTerms,int);
            node->AddChild(new KSAAssociatedPointerPODOutputNode< ConfigureTestSingleTriangleMultipole,
                           std::vector< int >, &ConfigureTestSingleTriangleMultipole::GetDegree >( std::string("Degree"), this) );
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(ConfigureTestSingleTriangleMultipole,UseAnalyticCalculator, int);
            AddKSAInputFor(ConfigureTestSingleTriangleMultipole,NTriangles, int);
            AddKSAInputFor(ConfigureTestSingleTriangleMultipole,NSamplePointsPerTriangle,int);
            AddKSAInputFor(ConfigureTestSingleTriangleMultipole,TriangleBoundingRadius,double);
            AddKSAInputFor(ConfigureTestSingleTriangleMultipole,MaxSampleRadius,double);
            AddKSAInputFor(ConfigureTestSingleTriangleMultipole,NQuadratureTerms,int);
            node->AddChild(new KSAAssociatedPointerPODInputNode< ConfigureTestSingleTriangleMultipole,
                           std::vector< int >, &ConfigureTestSingleTriangleMultipole::SetDegree >( std::string("Degree"), this) );
        }

        virtual const char* ClassName() const { return "ConfigureTestSingleTriangleMultipole"; };

    protected:

        int fUseAnalyticCalculator;
        std::vector<int> fDegree;
        int fNSamplePointsPerTriangle;
        int fNTriangles;
        double fTriangleBoundingRadius;
        double fMaxSampleRadius;
        int fNQuadratureTerms;

};

DefineKSAClassName( ConfigureTestSingleTriangleMultipole );

}


int main(int argc, char* argv[])
{

    (void) argc;
    (void) argv;

    if(argc < 2)
    {
        std::cout<<"Please specify the full path to the configuration file."<<std::endl;
        return 1;
    }

    #ifdef KEMFIELD_USE_GSL

    #ifdef KEMFIELD_USE_ROOT

    std::string input_file(argv[1]);

    KSAFileReader reader;
    reader.SetFileName(input_file);

    KSAInputCollector* in_collector = new KSAInputCollector();
    in_collector->SetFileReader(&reader);

    KSAObjectInputNode< ConfigureTestSingleTriangleMultipole >* config_input = new KSAObjectInputNode< ConfigureTestSingleTriangleMultipole >(std::string("ConfigureTestSingleTriangleMultipole"));

    std::cout<<"Reading configuration file. "<<std::endl;

    if( reader.Open() )
    {
        in_collector->ForwardInput(config_input);
    }
    else
    {
        std::cout<<"Could not open configuration file."<<std::endl;
        return 1;
    }

    int UseAnalyticCalculator = config_input->GetObject()->GetUseAnalyticCalculator();
    std::vector<int> Degree = *(config_input->GetObject()->GetDegree());
    int NTriangles = config_input->GetObject()->GetNTriangles();
    int NSamplePointsPerTriangle = config_input->GetObject()->GetNSamplePointsPerTriangle();
    double RMin = config_input->GetObject()->GetTriangleBoundingRadius();
    double RMax = config_input->GetObject()->GetMaxSampleRadius();
    int NQuad = config_input->GetObject()->GetNQuadratureTerms();

    int MaxDegree = -1;
    std::cout<<"number of degree = "<<Degree.size()<<std::endl;

    for(unsigned int i=0;i<Degree.size();i++)
    {
        if(Degree[i] > MaxDegree){MaxDegree = Degree[i];};
    }

    std::cout<<"max degree = "<<MaxDegree<<std::endl;

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

    double x,y,z;
    double p0[3];
    double p1[3];
    double p2[3];
    bool isAcute;
    double a,b,c;
    double delx, dely, delz;
    double min, mid, max;

    for(int i=0; i<NTriangles; i++)
    {
        isAcute = false;
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
                mid = b;
                min = a;
            }
            else
            {
                max = a;
                mid = a;
                min = b;
            }

            if(c < min)
            {
                mid = min;
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
                    mid = c;
                }
            }

            double hypot = std::sqrt( min*min + mid*mid);

            if( hypot <= max)
            {
                isAcute = true;
            }

        }
        while(!isAcute);

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

        //std::cout<<"before potential on triangle = "<<pot<<std::endl;
        Triangles[count]->SetSolution(1.0/pot);


        pot =  (integrator.Potential(Triangles[count], centroid))*(integrator.BasisValue(Triangles[count],0));
       //std::cout<<"potential on triangle = "<<pot<<std::endl;

        count++;

    }



//  for (unsigned int i=0;i<A.Dimension();i++)
//    for (unsigned int j=0;j<A.Dimension();j++)
//      std::cout<<"A("<<i<<","<<j<<"): "<<A(i,j)<<std::endl;

    //now lets make the multipole calculators
    KFMElectrostaticMultipoleCalculatorAnalytic* aCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();
    aCalc->SetDegree(MaxDegree);

    KFMElectrostaticMultipoleCalculatorNumeric* nCalc = new KFMElectrostaticMultipoleCalculatorNumeric();
    nCalc->SetDegree(MaxDegree);
    nCalc->SetNumberOfQuadratureTerms(NQuad);

    //the origin we want to expand about is (0,0,0)
    double origin[3];
    origin[0] = 0.;
    origin[1] = 0.;
    origin[2] = 0.;



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

    TCanvas *c1 = new TCanvas("c1","error",200,10,1300,900);

    std::vector<TProfile*> profiles;
    profiles.resize(Degree.size());
    std::stringstream ss;
//    double step_lower_bound;

    for(unsigned int k=0; k<Degree.size(); k++)
    {
        ss.str("");
        ss.clear();
        ss << "Error on ";
        ss << Degree[k];
        ss <<"-degree expansion.";
        profiles[k] = new TProfile( (ss.str()).c_str() , "", 100, TMath::Log10(RMin), TMath::Log10(RMax) );
        BinLogX(profiles[k]);
    }



    //compute multipole moments for each electrode
    KFMScalarMultipoleExpansion expan;
    expan.SetDegree(MaxDegree);
    std::vector< std::complex<double> > Moments;
    double p_multipole;
    double p_direct;
    double error;
    double point[3];
    double radius;
    //double cd, area;

    count = 0;
    for(int i=0; i<NTriangles; i++)
    {
        std::cout<<"i = "<<i<<std::endl;

        if(UseAnalyticCalculator != 0)
        {
            aCalc->ConstructExpansion(origin, &(TrianglePointClouds[i]), &expan);
        }
        else
        {
            nCalc->ConstructExpansion(origin, &(TrianglePointClouds[i]), &expan);
        }

        expan.GetMoments(&Moments);

        for(int j=0; j<NSamplePointsPerTriangle; j++)
        {
            if(j < NSamplePointsPerTriangle/4.0 )
            {
                //generate random point
                do
                {
                    x = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    y = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    z = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    radius = std::sqrt(x*x + y*y + z*z);
                }
                while( (radius > RMax/50.) || (radius < RMin) );
                point[0] = x; point[1] = y; point[2] = z;
            }
            else if(j < NSamplePointsPerTriangle/2.0 )
            {
                //generate random point
                do
                {
                    x = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    y = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    z = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    radius = std::sqrt(x*x + y*y + z*z);
                }
                while( (radius > RMax/10.0) || (radius < RMax/50.) );
                point[0] = x; point[1] = y; point[2] = z;
            }
            else if(j < 3.0*(NSamplePointsPerTriangle/4.0) )
            {
                //generate random point
                do
                {
                    x = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    y = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    z = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    radius = std::sqrt(x*x + y*y + z*z);
                }
                while( (radius > RMax/2.0) || (radius < RMax/10.) );
                point[0] = x; point[1] = y; point[2] = z;
            }
            else
            {
                //generate random point
                do
                {
                    x = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    y = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    z = -1.0*RMax + gsl_rng_uniform(r)*(2.0*RMax);
                    radius = std::sqrt(x*x + y*y + z*z);
                }
                while( (radius > RMax) || (radius < RMax/2.0) );
                point[0] = x; point[1] = y; point[2] = z;
            }

            for(unsigned int k=0; k<Degree.size(); k++)
            {
                //compute the sample points for each electrodes and the difference w/ the true potential and push into TProfile

                p_direct = (integrator.Potential(Triangles[count], point))*(integrator.BasisValue(Triangles[count],0));
                p_multipole = (PotentialMultipole(Degree[k], point, origin, &Moments) )*(integrator.BasisValue(Triangles[count],0))*(Triangles[count]->Area());



                error = std::fabs(p_direct - p_multipole);

//                std::cout<<"degree = "<<Degree[k]<<std::endl;
//                std::cout<<"p_direct = "<<p_direct<<std::endl;
//                std::cout<<"p_multipole = "<<p_multipole<<std::endl;
//////                std::cout<<"radius = "<<radius<<std::endl;
//                std::cout<<"error = "<<error<<std::endl;
                profiles[k]->Fill(radius, error, 1);
            }
        }

        count++;
    }

    int color = 0;

    int color_array[] = {1,2,3,4,95,6,7,8,9,13,46,51};


    for(unsigned int k=0; k<Degree.size(); k++)
    {
        profiles[k]->SetMaximum(1e-1);
        profiles[k]->SetMinimum(1e-16);
        profiles[k]->SetStats(0);

        profiles[k]->SetMarkerColor(color_array[color]);
        profiles[k]->SetMarkerSize(1.3);
        profiles[k]->SetLineColor(color_array[color]);
        profiles[k]->SetMarkerStyle(20 + color );
        profiles[k]->SetLineStyle((color)%3 + 1);

        color++;

        if(k==0)
        {
            profiles[k]->Draw();
            std::cout<<"Drawing profile: "<<k<<std::endl;
            profiles[k]->GetXaxis()->SetTitle("|#bf{x} - #bf{x}_{0}|/R");
            profiles[k]->GetYaxis()->SetTitle("Absolute Error (V)");
            profiles[k]->GetXaxis()->CenterTitle();
            profiles[k]->GetXaxis()->SetTitleOffset(1.3);
            profiles[k]->GetYaxis()->CenterTitle();
            profiles[k]->GetYaxis()->SetTitleOffset(1.3);
            profiles[k]->Draw("P");
        }
        else
        {
            profiles[k]->Draw("SAME P");
            std::cout<<"Drawing profile: "<<k<<std::endl;
        }
    }

    c1->SetLogy();
    c1->SetLogx();


    std::stringstream name;

//    double step_upper_bound;
	TLegend* key = new TLegend(0.65,0.7,0.9,0.9,"Degree of expansion");
	key->SetTextSize(0.03);
	key->SetFillColor(0);
    key->SetNColumns(2);
	for(unsigned int n = 0; n < Degree.size(); n++ )
    {
        std::string title("n = ");
        name.clear();
        name.str("");
        name << Degree[n];
        title.append(name.str());
		key->AddEntry(profiles[n], title.c_str(), "p");
	}
	key->Draw();
	c1->Update();

    App->Run();

    #else
        std::cout<<"To use this test program, please recompile with the flag KEMFIELD_USE_ROOT enabled"<<std::endl;
    #endif

    #else
        std::cout<<"To use this test program, please recompile witht the flag KEMFIELD_USE_GSL enabled"<<std::endl;
    #endif



  return 0;
}
