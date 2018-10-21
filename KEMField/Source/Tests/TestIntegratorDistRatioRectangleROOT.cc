#include <iostream>
#include <cstdlib>

#include "KEMThreeVector.hh"
#include "KSurfaceContainer.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticCubatureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"

#include "TStyle.h"
#include "TApplication.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TLatex.h"

#define POW2(x) ((x)*(x))

// VALUES
#define NUMRECTANGLES 1000 // number of rectangles for each Dr step
#define MINDR 2            // minimal distance ratio to be investigated
#define MAXDR 10000        // maximal distance ratio to be investigated
#define STEPSDR 1000       // steps between given distance ratio range
#define ACCURACY 1.E-15    // targeted accuracy for both electric potential and field
#define SEPARATECOMP	   // if this variable has been defined potentials and fields will be computed separately,
						   // hence 'ElectricFieldAndPotential' function won't be used
						   // both options have to produce same values
#define DRADDPERC 15       // additional fraction of distance ratio value at given accuracy to be added

// ROOT PLOTS AND COLORS (all settings apply for both field and potential)
#define PLOTANA 1
#define PLOTRWG 1
#define PLOTCUB4 0
#define PLOTCUB7 1
#define PLOTCUB9 0
#define PLOTCUB12 0
#define PLOTCUB17 0
#define PLOTCUB20 0
#define PLOTCUB33 1
#define PLOTNUM 1

#define COLANA kSpring
#define COLRWG kAzure
#define COLCUB4 kBlack
#define COLCUB7 kRed
#define COLCUB9 kRed-3
#define COLCUB12 kBlue
#define COLCUB17 kPink
#define COLCUB20 kOrange+3
#define COLCUB33 kMagenta
#define COLNUM kCyan

#define LINEWIDTH 1.

using namespace KEMField;

double IJKLRANDOM;
typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle> KEMRectangle;
void subrn(double *u,int len);
double randomnumber();

void printVec( std::string add, KEMThreeVector input )
{
	std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

namespace KEMField{

// visitor for rectangle geometry

class RectangleVisitor :
		public KSelectiveVisitor<KShapeVisitor,
		KTYPELIST_1(KRectangle)>
{
public:
	using KSelectiveVisitor<KShapeVisitor,KTYPELIST_1(KRectangle)>::Visit;

	RectangleVisitor(){}

	void Visit(KRectangle& r) { ProcessRectangle(r); }

	void ProcessRectangle(KRectangle& r)
	{
		// get missing side length
		fAverageSideLength = 0.5*(r.GetA() + r.GetB());

		// centroid
		fShapeCentroid = r.Centroid();
	}

	double GetAverageSideLength() { return fAverageSideLength; }
	KEMThreeVector GetCentroid(){ return fShapeCentroid; }

private:
	double fAverageSideLength;
	KEMThreeVector fShapeCentroid;
};

// visitor for computing fields and potentials

class RectangleVisitorForElectricFieldAndPotential :
		public KSelectiveVisitor<KShapeVisitor,
		KTYPELIST_1(KRectangle)>
{
public:
	using KSelectiveVisitor<KShapeVisitor,KTYPELIST_1(KRectangle)>::Visit;

	RectangleVisitorForElectricFieldAndPotential() {}

	void Visit(KRectangle& r) { ComputeElectricFieldAndPotential(r); }

	void ComputeElectricFieldAndPotential(KRectangle& r)
	{
		// rectangle data in array form
		const double data[11] = {
				r.GetA(),
				r.GetB(),
				r.GetP0().X(),
				r.GetP0().Y(),
				r.GetP0().Z(),
				r.GetN1().X(),
				r.GetN1().Y(),
				r.GetN1().Z(),
				r.GetN2().X(),
				r.GetN2().Y(),
				r.GetN2().Z()
		};

		// compute Gaussian points
		double rectQ4[12];
		fCubIntegrator.GaussPoints_Rect4P(data,rectQ4);
		double rectQ7[21];
		fCubIntegrator.GaussPoints_Rect7P(data,rectQ7);
		double rectQ9[27];
		fCubIntegrator.GaussPoints_Rect9P(data,rectQ9);
		double rectQ12[36];
		fCubIntegrator.GaussPoints_Rect12P(data,rectQ12);
		double rectQ17[51];
		fCubIntegrator.GaussPoints_Rect17P(data,rectQ17);
		double rectQ20[60];
		fCubIntegrator.GaussPoints_Rect20P(data,rectQ20);
		double rectQ33[99];
		fCubIntegrator.GaussPoints_Rect33P(data,rectQ33);

#ifdef SEPARATECOMP
		// separate field and potential computation

		fQuadElectricFieldAndPotential = std::make_pair(fQuadIntegrator.ElectricField(&r,fP),fQuadIntegrator.Potential(&r,fP));

		fAnaElectricFieldAndPotential = std::make_pair(fAnaIntegrator.ElectricField(&r,fP),fAnaIntegrator.Potential(&r,fP));
		fRwgElectricFieldAndPotential = std::make_pair(fRwgIntegrator.ElectricField(&r,fP),fRwgIntegrator.Potential(&r,fP));

		fCub4ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,4,rectQ4,gRectCub4w ),fCubIntegrator.Potential_RectNP( data,fP,4,rectQ4,gRectCub4w ));
		fCub7ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,7,rectQ7,gRectCub7w ),fCubIntegrator.Potential_RectNP( data,fP,7,rectQ7,gRectCub7w ));
		fCub9ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,9,rectQ9,gRectCub9w ),fCubIntegrator.Potential_RectNP( data,fP,9,rectQ9,gRectCub9w ));
		fCub12ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,12,rectQ12,gRectCub12w ),fCubIntegrator.Potential_RectNP( data,fP,12,rectQ12,gRectCub12w ));
		fCub17ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,17,rectQ17,gRectCub17w ),fCubIntegrator.Potential_RectNP( data,fP,17,rectQ17,gRectCub17w ));
		fCub20ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,20,rectQ20,gRectCub20w ),fCubIntegrator.Potential_RectNP( data,fP,20,rectQ20,gRectCub20w ));
		fCub33ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_RectNP( data,fP,33,rectQ33,gRectCub33w ),fCubIntegrator.Potential_RectNP( data,fP,33,rectQ33,gRectCub33w ));

		fNumElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField(&r,fP), fCubIntegrator.Potential(&r,fP) );
#else
		// simultaneous field and potential computation

		fQuadElectricFieldAndPotential = fQuadIntegrator.ElectricFieldAndPotential(&r,fP);

		fAnaElectricFieldAndPotential = fAnaIntegrator.ElectricFieldAndPotential(&r,fP);
		fRwgElectricFieldAndPotential = fRwgIntegrator.ElectricFieldAndPotential(&r,fP);

		fCub4ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,4,rectQ4,gRectCub4w );
		fCub7ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,7,rectQ7,gRectCub7w );
		fCub9ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,9,rectQ9,gRectCub9w );
		fCub12ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,12,rectQ12,gRectCub12w );
		fCub17ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,17,rectQ17,gRectCub17w );
		fCub20ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,20,rectQ20,gRectCub20w );
		fCub33ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_RectNP( data,fP,33,rectQ33,gRectCub33w );

		fNumElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential(&r,fP);
#endif
	}

	void SetPosition(const KPosition& p) const { fP = p; }

	std::pair<KEMThreeVector,double>& GetQuadElectricFieldAndPotential() const { return fQuadElectricFieldAndPotential;}

	std::pair<KEMThreeVector,double>& GetAnaElectricFieldAndPotential() const { return fAnaElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetRwgElectricFieldAndPotential() const { return fRwgElectricFieldAndPotential;}

	std::pair<KEMThreeVector,double>& GetCub4ElectricFieldAndPotential() const { return fCub4ElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetCub7ElectricFieldAndPotential() const { return fCub7ElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetCub9ElectricFieldAndPotential() const { return fCub9ElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetCub12ElectricFieldAndPotential() const { return fCub12ElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetCub17ElectricFieldAndPotential() const { return fCub17ElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetCub20ElectricFieldAndPotential() const { return fCub20ElectricFieldAndPotential;}
	std::pair<KEMThreeVector,double>& GetCub33ElectricFieldAndPotential() const { return fCub33ElectricFieldAndPotential;}

	std::pair<KEMThreeVector,double>& GetNumElectricFieldAndPotential() const { return fNumElectricFieldAndPotential;}

private:
	mutable KPosition fP;

	// Bi-Quadrature integrator as reference
	mutable std::pair<KEMThreeVector,double> fQuadElectricFieldAndPotential;
	KElectrostaticBiQuadratureRectangleIntegrator fQuadIntegrator;

	// analytical integration
	mutable std::pair<KEMThreeVector,double> fAnaElectricFieldAndPotential;
	KElectrostaticAnalyticRectangleIntegrator fAnaIntegrator;

	// analytical integration with RWG
	mutable std::pair<KEMThreeVector,double> fRwgElectricFieldAndPotential;
	KElectrostaticRWGRectangleIntegrator fRwgIntegrator;

	// cubature n-point integration rules
	mutable std::pair<KEMThreeVector,double> fCub4ElectricFieldAndPotential;
	mutable std::pair<KEMThreeVector,double> fCub7ElectricFieldAndPotential;
	mutable std::pair<KEMThreeVector,double> fCub9ElectricFieldAndPotential;
	mutable std::pair<KEMThreeVector,double> fCub12ElectricFieldAndPotential;
	mutable std::pair<KEMThreeVector,double> fCub17ElectricFieldAndPotential;
	mutable std::pair<KEMThreeVector,double> fCub20ElectricFieldAndPotential;
	mutable std::pair<KEMThreeVector,double> fCub33ElectricFieldAndPotential;
	KElectrostaticCubatureRectangleIntegrator fCubIntegrator;

	// adjusted cubature integrator dependent from distance ratio
	mutable std::pair<KEMThreeVector,double> fNumElectricFieldAndPotential;
};

} /* KEMField namespace*/

int main()
{
	// This program determines the accuracy of the rectangle integrators for a given distance ratio range.
	// distance ratio = distance to centroid / average side length

	// rectangle data
	double A,B;
	double P0[3];
	double N1[3];
	double N2[3];

	// assign a unique direction vector for field point to each rectangle and save into std::vector
	std::vector<KEMThreeVector> fPointDirections;

	// 'Num' rectangles will be diced in the beginning and added to a surface container
	// This values decides how much rectangles=field points will be computed for each distance ratio value

	KSurfaceContainer* container = new KSurfaceContainer();
	const unsigned int Num( NUMRECTANGLES ); /* number of rectangles */

	for( unsigned int i=0; i<Num; i++ ) {
		IJKLRANDOM = i+1;
		KEMRectangle* rectangle = new KEMRectangle();

		// dice rectangle geometry

		const double costheta = -1. + 2.*randomnumber();
		const double phi1 = 2. * M_PI * randomnumber();
		const double sintheta = sqrt( 1. - POW2(costheta) );

		N1[0] = sintheta*cos(phi1);
		N1[1] = sintheta*sin(phi1);
		N1[2] = costheta;

		const double phi2 = 2. * M_PI * randomnumber();

		N2[0] = cos(phi2)*sin(phi1) - sin(phi2)*costheta*cos(phi1);
		N2[1] = -cos(phi2)*cos(phi1) - sin(phi2)*costheta*sin(phi1);
		N2[2] = sin(phi2)*sintheta;

		A = randomnumber();
		B = randomnumber();

		P0[0] = -0.5*(B*N2[0]+A*N1[0]);
		P0[1] = -0.5*(B*N2[1]+A*N1[1]);
		P0[2] = -0.5*(B*N2[2]+A*N1[2]);

		rectangle->SetA( A );
		rectangle->SetB( B );
		rectangle->SetP0( KEMThreeVector(P0[0],P0[1],P0[2]) );
		rectangle->SetN1( KEMThreeVector(N1[0],N1[1],N1[2]) );
		rectangle->SetN2( KEMThreeVector(N2[0],N2[1],N2[2]) );

		rectangle->SetBoundaryValue( 1. );
		rectangle->SetSolution( 1. );

		container->push_back( rectangle );

		// dice direction vector of field points to be computed

		const double costhetaFP = -1.+2.*randomnumber();
		const double sinthetaFP = sqrt( 1. - POW2(costhetaFP) );
		const double phiFP = 2.*M_PI*randomnumber();

		fPointDirections.push_back( KEMThreeVector(
				sinthetaFP*cos(phiFP),
				sinthetaFP*sin(phiFP),
				costhetaFP ) );
	}

	// visitor for elements
	RectangleVisitor fRectangleVisitor;
	RectangleVisitorForElectricFieldAndPotential fComputeVisitor;

	KSurfaceContainer::iterator it;

	// distance ratios
	const double minDr( MINDR );
	const double maxDr( MAXDR );
	double Dr( 0. );
	const unsigned int kmax( STEPSDR);
	const double C = log(maxDr/minDr)/kmax;

	KEMField::cout << "Iterate from dist. ratio " << minDr << " to " << maxDr << " in " << kmax << " steps." << KEMField::endl;
	KEMField::cout << "Taking averaged relative error for " << container->size() << " rectangles for each dist. ratio value." << KEMField::endl;

	// field point
	KEMThreeVector fP;
//	double u[3];
//	double costheta, sintheta, phi;

	std::pair<KEMThreeVector,double> valQuad;
	std::pair<KEMThreeVector,double> valAna;
	std::pair<KEMThreeVector,double> valRwg;
	std::pair<KEMThreeVector,double> valCub[7];
	std::pair<KEMThreeVector,double> valNum;

	// variables for accuracy check

	// potential
	bool accFlagPotCub4( false );
	bool accFlagPotCub7( false );
	bool accFlagPotCub9( false );
	bool accFlagPotCub12( false );
	bool accFlagPotCub17( false );
	bool accFlagPotCub20( false );
	bool accFlagPotCub33( false );
	double drOptPotCub4( 0. );
	double drOptPotCub7( 0. );
	double drOptPotCub9( 0. );
	double drOptPotCub12( 0. );
	double drOptPotCub17( 0. );
	double drOptPotCub20( 0. );
	double drOptPotCub33( 0. );

	// field
	bool accFlagFieldCub4( false );
	bool accFlagFieldCub7( false );
	bool accFlagFieldCub9( false );
	bool accFlagFieldCub12( false );
	bool accFlagFieldCub17( false );
	bool accFlagFieldCub20( false );
	bool accFlagFieldCub33( false );
	double drOptFieldCub4( 0. );
	double drOptFieldCub7( 0. );
	double drOptFieldCub9( 0. );
	double drOptFieldCub12( 0. );
	double drOptFieldCub17( 0. );
	double drOptFieldCub20( 0. );
	double drOptFieldCub33( 0. );

	// plot

	TApplication* fAppWindow = new TApplication("fAppWindow", 0, NULL);

	gStyle->SetCanvasColor( kWhite );
	gStyle->SetLabelOffset( 0.03, "xyz" ); // values
	gStyle->SetTitleOffset( 1.6, "xyz" ); // label

	TMultiGraph *mgPot = new TMultiGraph();

	TGraph* plotDrPotAna = new TGraph( kmax+1 );
	plotDrPotAna->SetTitle( "Relative error of analytical rectangle potential" );
	plotDrPotAna->SetDrawOption( "AC" );
	plotDrPotAna->SetMarkerColor( COLANA );
	plotDrPotAna->SetLineWidth( LINEWIDTH );
	plotDrPotAna->SetLineColor( COLANA );
	plotDrPotAna->SetMarkerSize( 0.2 );
	plotDrPotAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgPot->Add( plotDrPotAna );

	TGraph* plotDrPotRwg = new TGraph( kmax+1 );
	plotDrPotRwg->SetTitle( "Relative error of rectangle RWG potential" );
	plotDrPotRwg->SetDrawOption( "same" );
	plotDrPotRwg->SetMarkerColor( COLRWG );
	plotDrPotRwg->SetLineWidth( LINEWIDTH );
	plotDrPotRwg->SetLineColor( COLRWG );
	plotDrPotRwg->SetMarkerSize( 0.2 );
	plotDrPotRwg->SetMarkerStyle( 8 );
	if( PLOTRWG ) mgPot->Add( plotDrPotRwg );

	TGraph* plotDrPotCub4 = new TGraph( kmax+1 );
	plotDrPotCub4->SetTitle( "Relative error of rectangle 4-point cubature potential" );
	plotDrPotCub4->SetDrawOption( "same" );
	plotDrPotCub4->SetMarkerColor( COLCUB4 );
	plotDrPotCub4->SetLineWidth( LINEWIDTH );
	plotDrPotCub4->SetLineColor( COLCUB4 );
	plotDrPotCub4->SetMarkerSize( 0.2 );
	plotDrPotCub4->SetMarkerStyle( 8 );
	if( PLOTCUB4 ) mgPot->Add( plotDrPotCub4 );

	TGraph* plotDrPotCub7 = new TGraph( kmax+1 );
	plotDrPotCub7->SetTitle( "Relative error of rectangle 7-point cubature potential" );
	plotDrPotCub7->SetDrawOption( "same" );
	plotDrPotCub7->SetMarkerColor( kRed );
	plotDrPotCub7->SetLineColor( kRed );
	plotDrPotCub7->SetMarkerSize( 0.2 );
	plotDrPotCub7->SetMarkerStyle( 8 );
	if( PLOTCUB7 ) mgPot->Add( plotDrPotCub7 );

	TGraph* plotDrPotCub9 = new TGraph( kmax+1 );
	plotDrPotCub9->SetTitle( "Relative error of rectangle 9-point cubature potential" );
	plotDrPotCub9->SetDrawOption( "same" );
	plotDrPotCub9->SetMarkerColor( COLCUB9 );
	plotDrPotCub9->SetLineWidth( LINEWIDTH );
	plotDrPotCub9->SetLineColor( COLCUB9 );
	plotDrPotCub9->SetMarkerSize( 0.2 );
	plotDrPotCub9->SetMarkerStyle( 8 );
	if( PLOTCUB9 ) mgPot->Add( plotDrPotCub9 );

	TGraph* plotDrPotCub12 = new TGraph( kmax+1 );
	plotDrPotCub12->SetTitle( "Relative error of rectangle 12-point cubature potential" );
	plotDrPotCub12->SetDrawOption( "same" );
	plotDrPotCub12->SetMarkerColor( COLCUB12 );
	plotDrPotCub12->SetLineWidth( LINEWIDTH );
	plotDrPotCub12->SetLineColor( COLCUB12 );
	plotDrPotCub12->SetMarkerSize( 0.2 );
	plotDrPotCub12->SetMarkerStyle( 8 );
	if( PLOTCUB12 ) mgPot->Add( plotDrPotCub12 );

	TGraph* plotDrPotCub17 = new TGraph( kmax+1 );
	plotDrPotCub17->SetTitle( "Relative error of rectangle 17-point cubature potential" );
	plotDrPotCub17->SetDrawOption( "same" );
	plotDrPotCub17->SetMarkerColor( COLCUB17 );
	plotDrPotCub17->SetLineWidth( LINEWIDTH );
	plotDrPotCub17->SetLineColor( COLCUB17 );
	plotDrPotCub17->SetMarkerSize( 0.2 );
	plotDrPotCub17->SetMarkerStyle( 8 );
	if( PLOTCUB17 ) mgPot->Add( plotDrPotCub17 );

	TGraph* plotDrPotCub20 = new TGraph( kmax+1 );
	plotDrPotCub20->SetTitle( "Relative error of rectangle 20-point cubature potential" );
	plotDrPotCub20->SetDrawOption( "same" );
	plotDrPotCub20->SetMarkerColor( COLCUB20 );
	plotDrPotCub20->SetLineWidth( LINEWIDTH );
	plotDrPotCub20->SetLineColor( COLCUB20 );
	plotDrPotCub20->SetMarkerSize( 0.2 );
	plotDrPotCub20->SetMarkerStyle( 8 );
	if( PLOTCUB20 ) mgPot->Add( plotDrPotCub20 );

	TGraph* plotDrPotCub33 = new TGraph( kmax+1 );
	plotDrPotCub33->SetTitle( "Relative error of rectangle 33-point cubature potential" );
	plotDrPotCub33->SetDrawOption( "same" );
	plotDrPotCub33->SetMarkerColor( COLCUB33 );
	plotDrPotCub33->SetLineWidth( LINEWIDTH );
	plotDrPotCub33->SetLineColor( COLCUB33 );
	plotDrPotCub33->SetMarkerSize( 0.2 );
	plotDrPotCub33->SetMarkerStyle( 8 );
	if( PLOTCUB33 ) mgPot->Add( plotDrPotCub33 );

	TGraph* plotDrPotNum = new TGraph( kmax+1 );
	plotDrPotNum->SetTitle( "Relative error of rectangle potential with adjusted numerical integrator" );
	plotDrPotNum->SetDrawOption( "same" );
	plotDrPotNum->SetMarkerColor( COLNUM );
	plotDrPotNum->SetLineWidth( LINEWIDTH );
	plotDrPotNum->SetLineColor( COLNUM );
	plotDrPotNum->SetMarkerSize( 0.2 );
	plotDrPotNum->SetMarkerStyle( 8 );
	if( PLOTNUM ) mgPot->Add( plotDrPotNum );

	TMultiGraph *mgField = new TMultiGraph();

	TGraph* plotDrFieldAna = new TGraph( kmax+1 );
	plotDrFieldAna->SetTitle( "Relative error of analytical rectangle field" );
	plotDrFieldAna->SetDrawOption( "AC" );
	plotDrFieldAna->SetMarkerColor( COLANA );
	plotDrFieldAna->SetLineWidth( LINEWIDTH );
	plotDrFieldAna->SetLineColor( COLANA );
	plotDrFieldAna->SetMarkerSize( 0.2 );
	plotDrFieldAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgField->Add( plotDrFieldAna );

	TGraph* plotDrFieldRwg = new TGraph( kmax+1 );
	plotDrFieldRwg->SetTitle( "Relative error of rectangle RWG field" );
	plotDrFieldRwg->SetDrawOption( "same" );
	plotDrFieldRwg->SetMarkerColor( COLRWG );
	plotDrFieldRwg->SetLineWidth( LINEWIDTH );
	plotDrFieldRwg->SetLineColor( COLRWG );
	plotDrFieldRwg->SetMarkerSize( 0.2 );
	plotDrFieldRwg->SetMarkerStyle( 8 );
	if( PLOTRWG ) mgField->Add( plotDrFieldRwg );

	TGraph* plotDrFieldCub4 = new TGraph( kmax+1 );
	plotDrFieldCub4->SetTitle( "Relative error of rectangle 4-point cubature field" );
	plotDrFieldCub4->SetDrawOption( "same" );
	plotDrFieldCub4->SetMarkerColor( COLCUB4 );
	plotDrFieldCub4->SetLineWidth( LINEWIDTH );
	plotDrFieldCub4->SetLineColor( COLCUB4 );
	plotDrFieldCub4->SetMarkerSize( 0.2 );
	plotDrFieldCub4->SetMarkerStyle( 8 );
	if( PLOTCUB4 ) mgField->Add( plotDrFieldCub4 );

	TGraph* plotDrFieldCub7 = new TGraph( kmax+1 );
	plotDrFieldCub7->SetTitle( "Relative error of rectangle 7-point cubature field" );
	plotDrFieldCub7->SetDrawOption( "same" );
	plotDrFieldCub7->SetMarkerColor( COLCUB7 );
	plotDrFieldCub7->SetLineWidth( LINEWIDTH );
	plotDrFieldCub7->SetLineColor( COLCUB7 );
	plotDrFieldCub7->SetMarkerSize( 0.2 );
	plotDrFieldCub7->SetMarkerStyle( 8 );
	if( PLOTCUB7 ) mgField->Add( plotDrFieldCub7 );

	TGraph* plotDrFieldCub9 = new TGraph( kmax+1 );
	plotDrFieldCub9->SetTitle( "Relative error of rectangle 9-point cubature field" );
	plotDrFieldCub9->SetDrawOption( "same" );
	plotDrFieldCub9->SetMarkerColor( COLCUB9 );
	plotDrFieldCub9->SetLineWidth( LINEWIDTH );
	plotDrFieldCub9->SetLineColor( COLCUB9 );
	plotDrFieldCub9->SetMarkerSize( 0.2 );
	plotDrFieldCub9->SetMarkerStyle( 8 );
	if( PLOTCUB9 ) mgField->Add( plotDrFieldCub9 );

	TGraph* plotDrFieldCub12 = new TGraph( kmax+1 );
	plotDrFieldCub12->SetTitle( "Relative error of rectangle 12-point cubature potential" );
	plotDrFieldCub12->SetDrawOption( "same" );
	plotDrFieldCub12->SetMarkerColor( COLCUB12 );
	plotDrFieldCub12->SetLineWidth( LINEWIDTH );
	plotDrFieldCub12->SetLineColor( COLCUB12 );
	plotDrFieldCub12->SetMarkerSize( 0.2 );
	plotDrFieldCub12->SetMarkerStyle( 8 );
	if( PLOTCUB12 ) mgField->Add( plotDrFieldCub12 );

	TGraph* plotDrFieldCub17 = new TGraph( kmax+1 );
	plotDrFieldCub17->SetTitle( "Relative error of rectangle 17-point cubature field" );
	plotDrFieldCub17->SetDrawOption( "same" );
	plotDrFieldCub17->SetMarkerColor( COLCUB17 );
	plotDrFieldCub17->SetLineWidth( LINEWIDTH );
	plotDrFieldCub17->SetLineColor( COLCUB17 );
	plotDrFieldCub17->SetMarkerSize( 0.2 );
	plotDrFieldCub17->SetMarkerStyle( 8 );
	if( PLOTCUB17 ) mgField->Add( plotDrFieldCub17 );

	TGraph* plotDrFieldCub20 = new TGraph( kmax+1 );
	plotDrFieldCub20->SetTitle( "Relative error of rectangle 20-point cubature field" );
	plotDrFieldCub20->SetDrawOption( "same" );
	plotDrFieldCub20->SetMarkerColor( COLCUB20 );
	plotDrFieldCub20->SetLineWidth( LINEWIDTH );
	plotDrFieldCub20->SetLineColor( COLCUB20 );
	plotDrFieldCub20->SetMarkerSize( 0.2 );
	plotDrFieldCub20->SetMarkerStyle( 8 );
	if( PLOTCUB20 ) mgField->Add( plotDrFieldCub20 );

	TGraph* plotDrFieldCub33 = new TGraph( kmax+1 );
	plotDrFieldCub33->SetTitle( "Relative error of rectangle 33-point cubature field" );
	plotDrFieldCub33->SetDrawOption( "same" );
	plotDrFieldCub33->SetMarkerColor( COLCUB33 );
	plotDrFieldCub33->SetLineWidth( LINEWIDTH );
	plotDrFieldCub33->SetLineColor( COLCUB33 );
	plotDrFieldCub33->SetMarkerSize( 0.2 );
	plotDrFieldCub33->SetMarkerStyle( 8 );
	if( PLOTCUB33 ) mgField->Add( plotDrFieldCub33 );

	TGraph* plotDrFieldNum = new TGraph( kmax+1 );
	plotDrFieldNum->SetTitle( "Relative error of triangle field with adjusted numerical integrator" );
	plotDrFieldNum->SetDrawOption( "same" );
	plotDrFieldNum->SetMarkerColor( COLNUM );
	plotDrFieldNum->SetLineWidth( LINEWIDTH );
	plotDrFieldNum->SetLineColor( COLNUM );
	plotDrFieldNum->SetMarkerSize( 0.2 );
	plotDrFieldNum->SetMarkerStyle( 8 );
	if( PLOTNUM )mgField->Add( plotDrFieldNum );

	double relAnaPot( 0. );
	double relRwgPot( 0. );
	double relCub4Pot( 0. );
	double relCub7Pot( 0. );
	double relCub9Pot( 0. );
	double relCub12Pot( 0. );
	double relCub17Pot( 0. );
	double relCub20Pot( 0. );
	double relCub33Pot( 0. );
	double relNumPot( 0. );

	double relAnaField( 0. );
	double relRwgField( 0. );
	double relCub4Field( 0. );
	double relCub7Field( 0. );
	double relCub9Field( 0. );
	double relCub12Field( 0. );
	double relCub17Field( 0. );
	double relCub20Field( 0. );
	double relCub33Field( 0. );
	double relNumField( 0. );

	const double targetAccuracy( ACCURACY );

	// iterate over distance ratios in log steps
	for( unsigned int k=0; k<=kmax; k++ ) {

		Dr = minDr * exp(C*k);

		KEMField::cout << "Current distance ratio: " << Dr << "\t\r";
		KEMField::cout.flush();

		unsigned int directionIndex( 0);

		// iterate over container elements and dice field point distance (direction vector has already been defined)
		for( it=container->begin<KElectrostaticBasis>(); it!=container->end<KElectrostaticBasis>(); ++it ) {

			IJKLRANDOM++;

			(*it)->Accept(fRectangleVisitor);

			// assign field point value
			fP = fRectangleVisitor.GetAverageSideLength()*Dr*fPointDirections[directionIndex];

			directionIndex++;

			fComputeVisitor.SetPosition(fP);

			(*it)->Accept(fComputeVisitor);

			valQuad = fComputeVisitor.GetQuadElectricFieldAndPotential();
			valAna = fComputeVisitor.GetAnaElectricFieldAndPotential();
			valRwg = fComputeVisitor.GetRwgElectricFieldAndPotential();
			valCub[0] = fComputeVisitor.GetCub4ElectricFieldAndPotential();
			valCub[1] = fComputeVisitor.GetCub7ElectricFieldAndPotential();
			valCub[2] = fComputeVisitor.GetCub9ElectricFieldAndPotential();
			valCub[3] = fComputeVisitor.GetCub12ElectricFieldAndPotential();
			valCub[4] = fComputeVisitor.GetCub17ElectricFieldAndPotential();
			valCub[5] = fComputeVisitor.GetCub20ElectricFieldAndPotential();
			valCub[6] = fComputeVisitor.GetCub33ElectricFieldAndPotential();
			valNum = fComputeVisitor.GetNumElectricFieldAndPotential();

			// sum for relative error

			relAnaPot += fabs((valAna.second-valQuad.second)/valQuad.second);
			relRwgPot += fabs((valRwg.second-valQuad.second)/valQuad.second);
			relCub4Pot += fabs((valCub[0].second-valQuad.second)/valQuad.second);
			relCub7Pot += fabs((valCub[1].second-valQuad.second)/valQuad.second);
			relCub9Pot += fabs((valCub[2].second-valQuad.second)/valQuad.second);
			relCub12Pot += fabs((valCub[3].second-valQuad.second)/valQuad.second);
			relCub17Pot += fabs((valCub[4].second-valQuad.second)/valQuad.second);
			relCub20Pot += fabs((valCub[5].second-valQuad.second)/valQuad.second);
			relCub33Pot += fabs((valCub[6].second-valQuad.second)/valQuad.second);
			relNumPot += fabs((valNum.second-valQuad.second)/valQuad.second);

			const double mag = sqrt(POW2(valQuad.first[0]) + POW2(valQuad.first[1]) + POW2(valQuad.first[2]));

			for( unsigned short i=0; i<3; i++ ) {
				relAnaField += fabs(valAna.first[i]-valQuad.first[i])/(mag);
				relRwgField += fabs(valRwg.first[i]-valQuad.first[i])/(mag);
				relCub4Field += fabs(valCub[0].first[i]-valQuad.first[i])/(mag);
				relCub7Field += fabs(valCub[1].first[i]-valQuad.first[i])/(mag);
				relCub9Field += fabs(valCub[2].first[i]-valQuad.first[i])/(mag);
				relCub12Field += fabs(valCub[3].first[i]-valQuad.first[i])/(mag);
				relCub17Field += fabs(valCub[4].first[i]-valQuad.first[i])/(mag);
				relCub20Field += fabs(valCub[5].first[i]-valQuad.first[i])/(mag);
				relCub33Field += fabs(valCub[6].first[i]-valQuad.first[i])/(mag);
				relNumField += fabs(valNum.first[i]-valQuad.first[i])/(mag);
			}
		}

		relAnaPot /= Num;
		relRwgPot /= Num;
		relCub4Pot /= Num;
		relCub7Pot /= Num;
		relCub9Pot /= Num;
		relCub12Pot /= Num;
		relCub17Pot /= Num;
		relCub20Pot /= Num;
		relCub33Pot /= Num;
		relNumPot /= Num;

		relAnaField /= Num;
		relRwgField /= Num;
		relCub4Field /= Num;
		relCub7Field /= Num;
		relCub9Field /= Num;
		relCub12Field /= Num;
		relCub17Field /= Num;
		relCub20Field /= Num;
		relCub33Field /= Num;
		relNumField /= Num;

		// potential

		if( (!accFlagPotCub4) && (relCub4Pot<=targetAccuracy) ) {
			drOptPotCub4 = Dr;
			accFlagPotCub4 = true;
		}
		if( (!accFlagPotCub7) && (relCub7Pot<=targetAccuracy) ) {
			drOptPotCub7 = Dr;
			accFlagPotCub7 = true;
		}
		if( (!accFlagPotCub9) && (relCub9Pot<=targetAccuracy) ) {
			drOptPotCub9 = Dr;
			accFlagPotCub9 = true;
		}
		if( (!accFlagPotCub12) && (relCub12Pot<=targetAccuracy) ) {
			drOptPotCub12 = Dr;
			accFlagPotCub12 = true;
		}
		if( (!accFlagPotCub17) && (relCub17Pot<=targetAccuracy) ) {
			drOptPotCub17 = Dr;
			accFlagPotCub17 = true;
		}
		if( (!accFlagPotCub20) && (relCub20Pot<=targetAccuracy) ) {
			drOptPotCub20 = Dr;
			accFlagPotCub20 = true;
		}
		if( (!accFlagPotCub33) && (relCub33Pot<=relRwgPot) ) {
			drOptPotCub33 = Dr;
			accFlagPotCub33 = true;
		}

		// field

		if( (!accFlagFieldCub4) && (relCub4Field<=targetAccuracy) ) {
			drOptFieldCub4 = Dr;
			accFlagFieldCub4 = true;
		}
		if( (!accFlagFieldCub7) && (relCub7Field<=targetAccuracy) ) {
			drOptFieldCub7 = Dr;
			accFlagFieldCub7 = true;
		}
		if( (!accFlagFieldCub9) && (relCub9Field<=targetAccuracy) ) {
			drOptFieldCub9 = Dr;
			accFlagFieldCub9 = true;
		}
		if( (!accFlagFieldCub12) && (relCub12Field<=targetAccuracy) ) {
			drOptFieldCub12 = Dr;
			accFlagFieldCub12 = true;
		}
		if( (!accFlagFieldCub17) && (relCub17Field<=targetAccuracy) ) {
			drOptFieldCub17 = Dr;
			accFlagFieldCub17 = true;
		}
		if( (!accFlagFieldCub20) && (relCub20Field<=targetAccuracy) ) {
			drOptFieldCub20 = Dr;
			accFlagFieldCub20 = true;
		}
		if( (!accFlagFieldCub33) && (relCub33Field<=relRwgField) ) {
			drOptFieldCub33 = Dr;
			accFlagFieldCub33 = true;
		}

		// save relative error of each integrator
		if( PLOTANA ) plotDrPotAna->SetPoint( k, Dr, relAnaPot );
		if( PLOTRWG ) plotDrPotRwg->SetPoint( k, Dr, relRwgPot );
		if( PLOTCUB4 ) plotDrPotCub4->SetPoint( k, Dr, relCub4Pot );
		if( PLOTCUB7 ) plotDrPotCub7->SetPoint( k, Dr, relCub7Pot );
		if( PLOTCUB9 ) plotDrPotCub9->SetPoint( k, Dr, relCub9Pot );
		if( PLOTCUB12 ) plotDrPotCub12->SetPoint( k, Dr, relCub12Pot );
		if( PLOTCUB17 ) plotDrPotCub17->SetPoint( k, Dr, relCub17Pot );
		if( PLOTCUB20 ) plotDrPotCub20->SetPoint( k, Dr, relCub20Pot );
		if( PLOTCUB33 ) plotDrPotCub33->SetPoint( k, Dr, relCub33Pot );
		if( PLOTNUM ) plotDrPotNum->SetPoint( k, Dr, relNumPot );

		// reset relative error
		relAnaPot = 0.;
		relRwgPot = 0.;
		relCub4Pot = 0.;
		relCub7Pot = 0.;
		relCub9Pot = 0.;
		relCub12Pot = 0.;
		relCub17Pot = 0.;
		relCub20Pot = 0.;
		relCub33Pot = 0.;
		relNumPot = 0.;

		if( PLOTANA ) plotDrFieldAna->SetPoint( k, Dr, relAnaField );
		if( PLOTRWG ) plotDrFieldRwg->SetPoint( k, Dr, relRwgField );
		if( PLOTCUB4 ) plotDrFieldCub4->SetPoint( k, Dr, relCub4Field );
		if( PLOTCUB7 ) plotDrFieldCub7->SetPoint( k, Dr, relCub7Field );
		if( PLOTCUB9 ) plotDrFieldCub9->SetPoint( k, Dr, relCub9Field );
		if( PLOTCUB12 ) plotDrFieldCub12->SetPoint( k, Dr, relCub12Field );
		if( PLOTCUB17 ) plotDrFieldCub17->SetPoint( k, Dr, relCub17Field );
		if( PLOTCUB20 ) plotDrFieldCub20->SetPoint( k, Dr, relCub20Field );
		if( PLOTCUB33 ) plotDrFieldCub33->SetPoint( k, Dr, relCub33Field );
		if( PLOTNUM ) plotDrFieldNum->SetPoint( k, Dr, relNumField );

		relAnaField = 0.;
		relRwgField = 0.;
		relCub4Field = 0.;
		relCub7Field = 0.;
		relCub9Field = 0.;
		relCub12Field = 0.;
		relCub17Field = 0.;
		relCub20Field = 0.;
		relCub33Field = 0.;
		relNumField = 0.;
	} /* distance ratio */

	const double drAdd( DRADDPERC/100. );

	KEMField::cout << "Recommended distance ratio values for target accuracy " << targetAccuracy << " (+" << (100*drAdd) << "%):" << KEMField::endl;
	KEMField::cout << "Rectangle potentials:" << KEMField::endl;
	KEMField::cout << "*  4-point cubature: " << ((1.+drAdd)*drOptPotCub4) << KEMField::endl;
	KEMField::cout << "*  7-point cubature: " << ((1.+drAdd)*drOptPotCub7) << KEMField::endl;
	KEMField::cout << "*  9-point cubature: " << ((1.+drAdd)*drOptPotCub9) << KEMField::endl;
	KEMField::cout << "* 12-point cubature: " << ((1.+drAdd)*drOptPotCub12) << KEMField::endl;
	KEMField::cout << "* 17-point cubature: " << ((1.+drAdd)*drOptPotCub17) << KEMField::endl;
	KEMField::cout << "* 20-point cubature: " << ((1.+drAdd)*drOptPotCub20) << KEMField::endl;
	KEMField::cout << "* 33-point cubature: " << ((1.)*drOptPotCub33) << " (no tolerance set here)" << KEMField::endl;
	KEMField::cout << "Rectangle fields (valid for all functions, implemented in integrator classes):" << KEMField::endl;
	KEMField::cout << "*  4-point cubature: " << ((1.+drAdd)*drOptFieldCub4) << KEMField::endl;
	KEMField::cout << "*  7-point cubature: " << ((1.+drAdd)*drOptFieldCub7) << KEMField::endl;
	KEMField::cout << "*  9-point cubature: " << ((1.+drAdd)*drOptFieldCub9) << KEMField::endl;
	KEMField::cout << "* 12-point cubature: " << ((1.+drAdd)*drOptFieldCub12) << KEMField::endl;
	KEMField::cout << "* 17-point cubature: " << ((1.+drAdd)*drOptFieldCub17) << KEMField::endl;
	KEMField::cout << "* 20-point cubature: " << ((1.+drAdd)*drOptFieldCub20) << KEMField::endl;
	KEMField::cout << "* 33-point cubature: " << ((1.)*drOptFieldCub33) << " (no tolerance set here)" << KEMField::endl;

	KEMField::cout << "Distance ratio analysis for cubature integrators finished." << KEMField::endl;

	TCanvas cPot("cPot","Averaged relative error of rectangle potential", 0, 0, 960, 760);
	cPot.SetMargin(0.16,0.06,0.15,0.06);
	cPot.SetLogx();
	cPot.SetLogy();

	// multigraph, create plot
	cPot.cd();

	mgPot->Draw( "apl" );
	mgPot->SetTitle( "Averaged error of rectangle potential" );
	mgPot->GetXaxis()->SetTitle( "distance ratio" );
	mgPot->GetXaxis()->CenterTitle();
	mgPot->GetYaxis()->SetTitle( "relative error" );
	mgPot->GetYaxis()->CenterTitle();

	TLatex l;
	l.SetTextAlign(11);
	l.SetTextFont(62);
	l.SetTextSize(0.032);

	if( PLOTRWG ) {
		l.SetTextAngle(29);
		l.SetTextColor( COLRWG );
		l.DrawLatex(500,1.5e-9,"Analytical (RWG)");
	}

	if( PLOTCUB4 ) {
		l.SetTextAngle(-43);
		l.SetTextColor( COLCUB4 );
		l.DrawLatex(9,1.e-6,"4-point cubature");
	}

	if( PLOTCUB7 ) {
		l.SetTextAngle(-54);
		l.SetTextColor( COLCUB7 );
		l.DrawLatex(4.,1.e-7,"7-point cubature");
	}

	if( PLOTCUB9 ) {
		l.SetTextAngle(-54);
		l.SetTextColor( COLCUB9 );
		l.DrawLatex(4.,1.e-7,"9-point cubature");
	}

	if( PLOTCUB12 ) {
		l.SetTextAngle(-65);
		l.SetTextColor( COLCUB12 );
		l.DrawLatex(2.9,1.e-8,"12-point cubature");
	}

	if( PLOTCUB17 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB17 );
		l.DrawLatex(2.5,3e-10,"17-point cubature");
	}

	if( PLOTCUB20 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB20 );
		l.DrawLatex(2.5,3e-10,"20-point cubature");
	}

	if( PLOTCUB33 ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLCUB33 );
		l.DrawLatex(3.,2.e-16,"33-point cubature");
	}

	if( PLOTNUM ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.5,3e-10,"Numerical cubature + RWG");
	}

	cPot.Update();

	TCanvas cField("cField","Averaged relative error of rectangle field", 0, 0, 960, 760);
	cField.SetMargin(0.16,0.06,0.15,0.06);
	cField.SetLogx();
	cField.SetLogy();

	// multigraph, create plot
	cField.cd();
	mgField->Draw( "apl" );
	mgField->SetTitle( "Averaged error of rectangle field" );
	mgField->GetXaxis()->SetTitle( "distance ratio" );
	mgField->GetXaxis()->CenterTitle();
	mgField->GetYaxis()->SetTitle( "relative error" );
	mgField->GetYaxis()->CenterTitle();

	if( PLOTRWG ) {
		l.SetTextAngle(29);
		l.SetTextColor( COLRWG );
		l.DrawLatex(500,1.5e-9,"Analytical (RWG)");
	}

	if( PLOTCUB4 ) {
		l.SetTextAngle(-43);
		l.SetTextColor( COLCUB4 );
		l.DrawLatex(9,1.e-6,"4-point cubature");
	}

	if( PLOTCUB7 ) {
		l.SetTextAngle(-54);
		l.SetTextColor( COLCUB7 );
		l.DrawLatex(4.,1.e-7,"7-point cubature");
	}

	if( PLOTCUB9 ) {
		l.SetTextAngle(-54);
		l.SetTextColor( COLCUB9 );
		l.DrawLatex(4.,1.e-7,"9-point cubature");
	}

	if( PLOTCUB12 ) {
		l.SetTextAngle(-65);
		l.SetTextColor( COLCUB12 );
		l.DrawLatex(2.9,1.e-8,"12-point cubature");
	}

	if( PLOTCUB17 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB17 );
		l.DrawLatex(2.5,3e-10,"17-point cubature");
	}

	if( PLOTCUB20 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB20 );
		l.DrawLatex(2.5,3e-10,"20-point cubature");
	}

	if( PLOTCUB33 ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLCUB33 );
		l.DrawLatex(3.,2.e-16,"33-point cubature");
	}

	if( PLOTNUM ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.5,3e-10,"Numerical cubature + RWG");
	}

	cField.Update();

	fAppWindow->Run();

	return 0;
}

void subrn(double *u,int len)
{
	// This subroutine computes random numbers u[1],...,u[len]
	// in the (0,1) interval. It uses the 0<IJKLRANDOM<900000000
	// integer as initialization seed.
	//  In the calling program the dimension
	// of the u[] vector should be larger than len (the u[0] value is
	// not used).
	// For each IJKLRANDOM
	// numbers the program computes completely independent random number
	// sequences (see: F. James, Comp. Phys. Comm. 60 (1990) 329, sec. 3.3).

	static int iff=0;
	static long ijkl,ij,kl,i,j,k,l,ii,jj,m,i97,j97,ivec;
	static float s,t,uu[98],c,cd,cm,uni;
	if(iff==0)
	{
		if(IJKLRANDOM==0)
		{
			std::cout << "Message from subroutine subrn:\n";
			std::cout << "the global integer IJKLRANDOM should be larger than 0 !!!\n";
			std::cout << "Computation is  stopped !!! \n";
			exit(0);
		}
		ijkl=IJKLRANDOM;
		if(ijkl<1 || ijkl>=900000000) ijkl=1;
		ij=ijkl/30082;
		kl=ijkl-30082*ij;
		i=((ij/177)%177)+2;
		j=(ij%177)+2;
		k=((kl/169)%178)+1;
		l=kl%169;
		for(ii=1;ii<=97;ii++)
		{ s=0; t=0.5;
		for(jj=1;jj<=24;jj++)
		{ m=(((i*j)%179)*k)%179;
		i=j; j=k; k=m;
		l=(53*l+1)%169;
		if((l*m)%64 >= 32) s=s+t;
		t=0.5*t;
		}
		uu[ii]=s;
		}
		c=362436./16777216.;
		cd=7654321./16777216.;
		cm=16777213./16777216.;
		i97=97;
		j97=33;
		iff=1;
	}
	for(ivec=1;ivec<=len;ivec++)
	{ uni=uu[i97]-uu[j97];
	if(uni<0.) uni=uni+1.;
	uu[i97]=uni;
	i97=i97-1;
	if(i97==0) i97=97;
	j97=j97-1;
	if(j97==0) j97=97;
	c=c-cd;
	if(c<0.) c=c+cm;
	uni=uni-c;
	if(uni<0.) uni=uni+1.;
	if(uni==0.)
	{ uni=uu[j97]*0.59604644775391e-07;
	if(uni==0.) uni=0.35527136788005e-14;
	}
	u[ivec]=uni;
	}
	return;
}

////////////////////////////////////////////////////////////////

double randomnumber()
{
	// This function computes 1 random number in the (0,1) interval,
	// using the subrn subroutine.

	double u[2];
	subrn(u,1);
	return u[1];
}
