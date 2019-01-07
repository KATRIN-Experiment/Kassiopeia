#include <iostream>
#include <cstdlib>

#include "KThreeVector_KEMField.hh"
#include "KSurfaceContainer.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KElectrostaticCubatureTriangleIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"

#include "TStyle.h"
#include "TApplication.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TLatex.h"

#define POW2(x) ((x)*(x))

// VALUES
#define NUMTRIANGLES 1000  // number of triangles for each Dr step
#define MINDR 2            // minimal distance ratio to be investigated
#define MAXDR 10000        // maximal distance ratio to be investigated
#define STEPSDR 1000       // steps between given distance ratio range
#define ACCURACY 1.E-15    // targeted accuracy for both electric potential and field
#define SEPARATECOMP	   // if this variable has been defined potentials and fields will be computed separately,
						   // hence 'ElectricFieldAndPotential' function won't be used
						   // both options have to produce same values
#define DRADDPERC 15       // additional fraction of distance ratio value at given accuracy to be added

// ROOT PLOTS AND COLORS (all settings apply for both field and potential)
#define PLOTANA 0
#define PLOTRWG 1
#define PLOTCUB4 0
#define PLOTCUB7 1
#define PLOTCUB12 0
#define PLOTCUB16 0
#define PLOTCUB19 0
#define PLOTCUB33 1
#define PLOTNUM 1

#define COLANA kBlue
#define COLRWG kAzure
#define COLCUB4 kBlack
#define COLCUB7 kCyan+3
#define COLCUB12 kOrange+7
#define COLCUB16 kPink
#define COLCUB19 kSpring
#define COLCUB33 kRed//kOrange+3
#define COLNUM kRed

#define LINEWIDTH 1.

using namespace KEMField;

double IJKLRANDOM;
typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle> KEMTriangle;
void subrn(double *u,int len);
double randomnumber();

void printVec( std::string add, KThreeVector input )
{
	std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

namespace KEMField{

// visitor for triangle geometry

class TriangleVisitor :
		public KSelectiveVisitor<KShapeVisitor,
		KTYPELIST_1(KTriangle)>
{
public:
	using KSelectiveVisitor<KShapeVisitor,KTYPELIST_1(KTriangle)>::Visit;

	TriangleVisitor(){}

	void Visit(KTriangle& t) { ProcessTriangle(t); }

	void ProcessTriangle(KTriangle& t)
	{
		// get missing side length
		const double lengthP1P2 = (t.GetP2() - t.GetP1()).Magnitude();
		fAverageSideLength = (t.GetA() + t.GetB() + lengthP1P2)/3.;

		// centroid
		fShapeCentroid = t.Centroid();
	}

	double GetAverageSideLength() { return fAverageSideLength; }
	KThreeVector GetCentroid(){ return fShapeCentroid; }

private:
	double fAverageSideLength;
	KThreeVector fShapeCentroid;
};

// visitor for computing fields and potentials

class TriangleVisitorForElectricFieldAndPotential :
		public KSelectiveVisitor<KShapeVisitor,
		KTYPELIST_1(KTriangle)>
{
public:
	using KSelectiveVisitor<KShapeVisitor,KTYPELIST_1(KTriangle)>::Visit;

	TriangleVisitorForElectricFieldAndPotential() {}

	void Visit(KTriangle& t) { ComputeElectricFieldAndPotential(t); }

	void ComputeElectricFieldAndPotential(KTriangle& t)
	{
		// triangle data in array form

		const double data[11] = {
				t.GetA(),
				t.GetB(),
				t.GetP0().X(),
				t.GetP0().Y(),
				t.GetP0().Z(),
				t.GetN1().X(),
				t.GetN1().Y(),
				t.GetN1().Z(),
				t.GetN2().X(),
				t.GetN2().Y(),
				t.GetN2().Z()
		};

		// compute Gaussian points

		double triQ4[12];
		fCubIntegrator.GaussPoints_Tri4P(data,triQ4);
		double triQ7[21];
		fCubIntegrator.GaussPoints_Tri7P(data,triQ7);
		double triQ12[36];
		fCubIntegrator.GaussPoints_Tri12P(data,triQ12);
		double triQ16[48];
		fCubIntegrator.GaussPoints_Tri16P(data,triQ16);
		double triQ19[57];
		fCubIntegrator.GaussPoints_Tri19P(data,triQ19);
		double triQ33[99];
		fCubIntegrator.GaussPoints_Tri33P(data,triQ33);

#ifdef SEPARATECOMP
		// separate field and potential computation

		fQuadElectricFieldAndPotential = std::make_pair( fQuadIntegrator.ElectricField(&t,fP), fQuadIntegrator.Potential(&t,fP) );

		fAnaElectricFieldAndPotential = std::make_pair( fAnaIntegrator.ElectricField(&t,fP), fAnaIntegrator.Potential(&t,fP) );
		fRwgElectricFieldAndPotential = std::make_pair( fRwgIntegrator.ElectricField(&t,fP), fRwgIntegrator.Potential(&t,fP) );

		fCub4ElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField_TriNP( data,fP,4,triQ4,gTriCub4w ), fCubIntegrator.Potential_TriNP( data,fP,4,triQ4,gTriCub4w ) );
		fCub7ElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField_TriNP( data,fP,7,triQ7,gTriCub7w ),fCubIntegrator.Potential_TriNP( data,fP,7,triQ7,gTriCub7w ) );
		fCub12ElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField_TriNP( data,fP,12,triQ12,gTriCub12w ),fCubIntegrator.Potential_TriNP( data,fP,12,triQ12,gTriCub12w ) );
		fCub16ElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField_TriNP( data,fP,16,triQ16,gTriCub16w ),fCubIntegrator.Potential_TriNP( data,fP,16,triQ16,gTriCub16w ) );
		fCub19ElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField_TriNP( data,fP,19,triQ19,gTriCub19w ),fCubIntegrator.Potential_TriNP( data,fP,19,triQ19,gTriCub19w ) );
		fCub33ElectricFieldAndPotential = std::make_pair(fCubIntegrator.ElectricField_TriNP( data,fP,33,triQ33,gTriCub33w ),fCubIntegrator.Potential_TriNP( data,fP,33,triQ33,gTriCub33w ));

		fNumElectricFieldAndPotential = std::make_pair( fCubIntegrator.ElectricField(&t,fP),fCubIntegrator.Potential(&t,fP) );
#else
		// simultaneous field and potential computation

		fQuadElectricFieldAndPotential = fQuadIntegrator.ElectricFieldAndPotential(&t,fP);

		fAnaElectricFieldAndPotential = fAnaIntegrator.ElectricFieldAndPotential(&t,fP);
		fRwgElectricFieldAndPotential = fRwgIntegrator.ElectricFieldAndPotential(&t,fP);

		fCub4ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_TriNP( data,fP,4,triQ4,gTriCub4w );
		fCub7ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_TriNP( data,fP,7,triQ7,gTriCub7w );
		fCub12ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_TriNP( data,fP,12,triQ12,gTriCub12w );
		fCub16ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_TriNP( data,fP,16,triQ16,gTriCub16w );
		fCub19ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_TriNP( data,fP,19,triQ19,gTriCub19w );
		fCub33ElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential_TriNP( data,fP,33,triQ33,gTriCub33w );

		fNumElectricFieldAndPotential = fCubIntegrator.ElectricFieldAndPotential(&t,fP);
#endif
	}

	void SetPosition(const KPosition& p) const { fP = p; }

	std::pair<KThreeVector,double>& GetQuadElectricFieldAndPotential() const { return fQuadElectricFieldAndPotential;}

	std::pair<KThreeVector,double>& GetAnaElectricFieldAndPotential() const { return fAnaElectricFieldAndPotential;}
	std::pair<KThreeVector,double>& GetRwgElectricFieldAndPotential() const { return fRwgElectricFieldAndPotential;}

	std::pair<KThreeVector,double>& GetCub4ElectricFieldAndPotential() const { return fCub4ElectricFieldAndPotential;}
	std::pair<KThreeVector,double>& GetCub7ElectricFieldAndPotential() const { return fCub7ElectricFieldAndPotential;}
	std::pair<KThreeVector,double>& GetCub12ElectricFieldAndPotential() const { return fCub12ElectricFieldAndPotential;}
	std::pair<KThreeVector,double>& GetCub16ElectricFieldAndPotential() const { return fCub16ElectricFieldAndPotential;}
	std::pair<KThreeVector,double>& GetCub19ElectricFieldAndPotential() const { return fCub19ElectricFieldAndPotential;}
	std::pair<KThreeVector,double>& GetCub33ElectricFieldAndPotential() const { return fCub33ElectricFieldAndPotential;}

	std::pair<KThreeVector,double>& GetNumElectricFieldAndPotential() const { return fNumElectricFieldAndPotential;}

private:
	mutable KPosition fP;

	// Bi-Quadrature integrator as reference
	mutable std::pair<KThreeVector,double> fQuadElectricFieldAndPotential;
	KElectrostaticBiQuadratureTriangleIntegrator fQuadIntegrator;

	// analytical integration
	mutable std::pair<KThreeVector,double> fAnaElectricFieldAndPotential;
	KElectrostaticAnalyticTriangleIntegrator fAnaIntegrator;

	// analytical integration with RWG
	mutable std::pair<KThreeVector,double> fRwgElectricFieldAndPotential;
	KElectrostaticRWGTriangleIntegrator fRwgIntegrator;

	// cubature n-point integration rules
	mutable std::pair<KThreeVector,double> fCub4ElectricFieldAndPotential;
	mutable std::pair<KThreeVector,double> fCub7ElectricFieldAndPotential;
	mutable std::pair<KThreeVector,double> fCub12ElectricFieldAndPotential;
	mutable std::pair<KThreeVector,double> fCub16ElectricFieldAndPotential;
	mutable std::pair<KThreeVector,double> fCub19ElectricFieldAndPotential;
	mutable std::pair<KThreeVector,double> fCub33ElectricFieldAndPotential;
	KElectrostaticCubatureTriangleIntegrator fCubIntegrator;

	// adjusted cubature integrator dependent from distance ratio
	mutable std::pair<KThreeVector,double> fNumElectricFieldAndPotential;
};

} /* KEMField namespace*/

int main()
{
	// This program determines the accuracy of the triangle integrators for a given distance ratio range.
	// distance ratio = distance to centroid / average side length

	// triangle data
	double A,B;
	double P0[3];
	double P1[3];
	double P2[3];
	double N1[3];
	double N2[3];

	// assign a unique direction vector for field point to each rectangle and save into std::vector
	std::vector<KThreeVector> fPointDirections;

	// 'Num' triangles will be diced in the beginning and added to a surface container
	// This values decides how much triangles=field points will be computed for each distance ratio value

	KSurfaceContainer* container = new KSurfaceContainer();
	const unsigned int Num( NUMTRIANGLES ); /* number of triangles */

	for( unsigned int i=0; i<Num; i++ ) {
		IJKLRANDOM = i+1;
		KEMTriangle* triangle = new KEMTriangle();

		// dice triangle geometry
		for( unsigned short l=0; l<3; l++ ) P0[l]=-1.+2.*randomnumber();
		for( unsigned short j=0; j<3; j++ ) P1[j]=-1.+2.*randomnumber();
		for( unsigned short k=0; k<3; k++ ) P2[k]=-1.+2.*randomnumber();

		// compute further triangle data
		A = sqrt(POW2(P1[0]-P0[0]) + POW2(P1[1]-P0[1]) + POW2(P1[2]-P0[2]));
		B = sqrt(POW2(P2[0]-P0[0]) + POW2(P2[1]-P0[1]) + POW2(P2[2]-P0[2]));

		N1[0] = (P1[0]-P0[0]) / A;
		N1[1] = (P1[1]-P0[1]) / A;
		N1[2] = (P1[2]-P0[2]) / A;
		N2[0] = (P2[0]-P0[0]) / B;
		N2[1] = (P2[1]-P0[1]) / B;
		N2[2] = (P2[2]-P0[2]) / B;

		triangle->SetA( A );
		triangle->SetB( B );
		triangle->SetP0( KThreeVector(P0[0],P0[1],P0[2]) );
		triangle->SetN1( KThreeVector(N1[0],N1[1],N1[2]) );
		triangle->SetN2( KThreeVector(N2[0],N2[1],N2[2]) );

		triangle->SetBoundaryValue( 1. );
		triangle->SetSolution( 1. );

		container->push_back( triangle );

		const double costhetaFP = -1.+2.*randomnumber();
		const double sinthetaFP = sqrt( 1. - POW2(costhetaFP) );
		const double phiFP = 2.*M_PI*randomnumber();

		fPointDirections.push_back( KThreeVector(
				sinthetaFP*cos(phiFP),
				sinthetaFP*sin(phiFP),
				costhetaFP ) );
	}

	// visitor for elements
	TriangleVisitor fTriangleVisitor;
	TriangleVisitorForElectricFieldAndPotential fComputeVisitor;

	KSurfaceContainer::iterator it;

	// distance ratios
	const double minDr( MINDR );
	const double maxDr( MAXDR );
	double Dr( 0. );
	const unsigned int kmax( STEPSDR );
	const double C = log(maxDr/minDr)/kmax;

	KEMField::cout << "Iterate from dist. ratio " << minDr << " to " << maxDr << " in " << kmax << " steps." << KEMField::endl;
	KEMField::cout << "Taking averaged relative error for " << container->size() << " triangles for each dist. ratio value." << KEMField::endl;

	// field point
	KThreeVector fP;

	// field and potential values
	std::pair<KThreeVector,double> valQuad;
	std::pair<KThreeVector,double> valAna;
	std::pair<KThreeVector,double> valRwg;
	std::pair<KThreeVector,double> valCub[6];
	std::pair<KThreeVector,double> valNum;

	// variables for accuracy check of n-point cubature integration

	// potential
	bool accFlagPotCub4( false );
	bool accFlagPotCub7( false );
	bool accFlagPotCub12( false );
	bool accFlagPotCub16( false );
	bool accFlagPotCub19( false );
	bool accFlagPotCub33( false );
	double drOptPotCub4( 0. );
	double drOptPotCub7( 0. );
	double drOptPotCub12( 0. );
	double drOptPotCub16( 0. );
	double drOptPotCub19( 0. );
	double drOptPotCub33( 0. );

	// field
	bool accFlagFieldCub4( false );
	bool accFlagFieldCub7( false );
	bool accFlagFieldCub12( false );
	bool accFlagFieldCub16( false );
	bool accFlagFieldCub19( false );
	bool accFlagFieldCub33( false );
	double drOptFieldCub4( 0. );
	double drOptFieldCub7( 0. );
	double drOptFieldCub12( 0. );
	double drOptFieldCub16( 0. );
	double drOptFieldCub19( 0. );
	double drOptFieldCub33( 0. );

	// plot

	TApplication* fAppWindow = new TApplication("fAppWindow", 0, NULL);

	gStyle->SetCanvasColor( kWhite );
	gStyle->SetLabelOffset( 0.03, "xyz" ); // values
	gStyle->SetTitleOffset( 1.8, "xyz" ); // label

	TMultiGraph *mgPot = new TMultiGraph();

	TGraph* plotDrPotAna = new TGraph( kmax+1 );
	plotDrPotAna->SetTitle( "Relative error of analytical triangle potential" );
	plotDrPotAna->SetDrawOption( "AC" );
	plotDrPotAna->SetMarkerColor( COLANA );
	plotDrPotAna->SetLineWidth( LINEWIDTH );
	plotDrPotAna->SetLineColor( COLANA );
	plotDrPotAna->SetMarkerSize( 0.2 );
	plotDrPotAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgPot->Add( plotDrPotAna );

	TGraph* plotDrPotRwg = new TGraph( kmax+1 );
	plotDrPotRwg->SetTitle( "Relative error of triangle RWG potential" );
	plotDrPotRwg->SetDrawOption( "same" );
	plotDrPotRwg->SetMarkerColor( COLRWG );
	plotDrPotRwg->SetLineWidth( LINEWIDTH );
	plotDrPotRwg->SetLineColor( COLRWG );
	plotDrPotRwg->SetMarkerSize( 0.2 );
	plotDrPotRwg->SetMarkerStyle( 8 );
	if( PLOTRWG ) mgPot->Add( plotDrPotRwg );

	TGraph* plotDrPotCub4 = new TGraph( kmax+1 );
	plotDrPotCub4->SetTitle( "Relative error of triangle 4-point cubature potential" );
	plotDrPotCub4->SetDrawOption( "same" );
	plotDrPotCub4->SetMarkerColor( COLCUB4 );
	plotDrPotCub4->SetLineWidth( LINEWIDTH );
	plotDrPotCub4->SetLineColor( COLCUB4 );
	plotDrPotCub4->SetMarkerSize( 0.2 );
	plotDrPotCub4->SetMarkerStyle( 8 );
	if( PLOTCUB4 ) mgPot->Add( plotDrPotCub4 );

	TGraph* plotDrPotCub7 = new TGraph( kmax+1 );
	plotDrPotCub7->SetTitle( "Relative error of triangle 7-point cubature potential" );
	plotDrPotCub7->SetDrawOption( "same" );
	plotDrPotCub7->SetMarkerColor( COLCUB7 );
	plotDrPotCub7->SetLineWidth( LINEWIDTH );
	plotDrPotCub7->SetLineColor( COLCUB7 );
	plotDrPotCub7->SetMarkerSize( 0.2 );
	plotDrPotCub7->SetMarkerStyle( 8 );
	if( PLOTCUB7 ) mgPot->Add( plotDrPotCub7 );

	TGraph* plotDrPotCub12 = new TGraph( kmax+1 );
	plotDrPotCub12->SetTitle( "Relative error of triangle 12-point cubature potential" );
	plotDrPotCub12->SetDrawOption( "same" );
	plotDrPotCub12->SetMarkerColor( COLCUB12 );
	plotDrPotCub12->SetLineWidth( LINEWIDTH );
	plotDrPotCub12->SetLineColor( COLCUB12 );
	plotDrPotCub12->SetMarkerSize( 0.2 );
	plotDrPotCub12->SetMarkerStyle( 8 );
	if( PLOTCUB12 ) mgPot->Add( plotDrPotCub12 );

	TGraph* plotDrPotCub16 = new TGraph( kmax+1 );
	plotDrPotCub16->SetTitle( "Relative error of triangle 16-point cubature potential" );
	plotDrPotCub16->SetDrawOption( "same" );
	plotDrPotCub16->SetMarkerColor( COLCUB16 );
	plotDrPotCub16->SetLineWidth( LINEWIDTH );
	plotDrPotCub16->SetLineColor( COLCUB16 );
	plotDrPotCub16->SetMarkerSize( 0.2 );
	plotDrPotCub16->SetMarkerStyle( 8 );
	if( PLOTCUB16 ) mgPot->Add( plotDrPotCub16 );

	TGraph* plotDrPotCub19 = new TGraph( kmax+1 );
	plotDrPotCub19->SetTitle( "Relative error of triangle 19-point cubature potential" );
	plotDrPotCub19->SetDrawOption( "same" );
	plotDrPotCub19->SetMarkerColor( COLCUB19 );
	plotDrPotCub19->SetLineWidth( LINEWIDTH );
	plotDrPotCub19->SetLineColor( COLCUB19 );
	plotDrPotCub19->SetMarkerSize( 0.2 );
	plotDrPotCub19->SetMarkerStyle( 8 );
	if( PLOTCUB19 ) mgPot->Add( plotDrPotCub19 );

	TGraph* plotDrPotCub33 = new TGraph( kmax+1 );
	plotDrPotCub33->SetTitle( "Relative error of triangle 33-point cubature potential" );
	plotDrPotCub33->SetDrawOption( "same" );
	plotDrPotCub33->SetMarkerColor( COLCUB33 );
	plotDrPotCub33->SetLineWidth( LINEWIDTH );
	plotDrPotCub33->SetLineColor( COLCUB33 );
	plotDrPotCub33->SetMarkerSize( 0.2 );
	plotDrPotCub33->SetMarkerStyle( 8 );
	if( PLOTCUB33 ) mgPot->Add( plotDrPotCub33 );

	TGraph* plotDrPotNum = new TGraph( kmax+1 );
	plotDrPotNum->SetTitle( "Relative error of triangle potential with adjusted numerical integrator" );
	plotDrPotNum->SetDrawOption( "same" );
	plotDrPotNum->SetMarkerColor( COLNUM );
	plotDrPotNum->SetLineWidth( LINEWIDTH );
	plotDrPotNum->SetLineColor( COLNUM );
	plotDrPotNum->SetMarkerSize( 0.2 );
	plotDrPotNum->SetMarkerStyle( 8 );
	if( PLOTNUM ) mgPot->Add( plotDrPotNum );

	TMultiGraph *mgField = new TMultiGraph();

	TGraph* plotDrFieldAna = new TGraph( kmax+1 );
	plotDrFieldAna->SetTitle( "Relative error of analytical triangle potential" );
	plotDrFieldAna->SetDrawOption( "AC" );
	plotDrFieldAna->SetMarkerColor( COLANA );
	plotDrFieldAna->SetLineWidth( LINEWIDTH );
	plotDrFieldAna->SetLineColor( COLANA );
	plotDrFieldAna->SetMarkerSize( 0.2 );
	plotDrFieldAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgField->Add( plotDrFieldAna );

	TGraph* plotDrFieldRwg = new TGraph( kmax+1 );
	plotDrFieldRwg->SetTitle( "Relative error of triangle RWG potential" );
	plotDrFieldRwg->SetDrawOption( "same" );
	plotDrFieldRwg->SetMarkerColor( COLRWG );
	plotDrFieldRwg->SetLineWidth( LINEWIDTH );
	plotDrFieldRwg->SetLineColor( COLRWG );
	plotDrFieldRwg->SetMarkerSize( 0.2 );
	plotDrFieldRwg->SetMarkerStyle( 8 );
	if( PLOTRWG ) mgField->Add( plotDrFieldRwg );

	TGraph* plotDrFieldCub4 = new TGraph( kmax+1 );
	plotDrFieldCub4->SetTitle( "Relative error of triangle 4-point cubature potential" );
	plotDrFieldCub4->SetDrawOption( "same" );
	plotDrFieldCub4->SetMarkerColor( COLCUB4 );
	plotDrFieldCub4->SetLineWidth( LINEWIDTH );
	plotDrFieldCub4->SetLineColor( COLCUB4 );
	plotDrFieldCub4->SetMarkerSize( 0.2 );
	plotDrFieldCub4->SetMarkerStyle( 8 );
	if( PLOTCUB4 ) mgField->Add( plotDrFieldCub4 );

	TGraph* plotDrFieldCub7 = new TGraph( kmax+1 );
	plotDrFieldCub7->SetTitle( "Relative error of triangle 7-point cubature potential" );
	plotDrFieldCub7->SetDrawOption( "same" );
	plotDrFieldCub7->SetMarkerColor( COLCUB7 );
	plotDrFieldCub7->SetLineWidth( LINEWIDTH );
	plotDrFieldCub7->SetLineColor( COLCUB7 );
	plotDrFieldCub7->SetMarkerSize( 0.2 );
	plotDrFieldCub7->SetMarkerStyle( 8 );
	if( PLOTCUB7 ) mgField->Add( plotDrFieldCub7 );

	TGraph* plotDrFieldCub12 = new TGraph( kmax+1 );
	plotDrFieldCub12->SetTitle( "Relative error of triangle 12-point cubature potential" );
	plotDrFieldCub12->SetDrawOption( "same" );
	plotDrFieldCub12->SetMarkerColor( COLCUB12 );
	plotDrFieldCub12->SetLineWidth( LINEWIDTH );
	plotDrFieldCub12->SetLineColor( COLCUB12 );
	plotDrFieldCub12->SetMarkerSize( 0.2 );
	plotDrFieldCub12->SetMarkerStyle( 8 );
	if( PLOTCUB12 ) mgField->Add( plotDrFieldCub12 );

	TGraph* plotDrFieldCub16 = new TGraph( kmax+1 );
	plotDrFieldCub16->SetTitle( "Relative error of triangle 16-point cubature potential" );
	plotDrFieldCub16->SetDrawOption( "same" );
	plotDrFieldCub16->SetMarkerColor( COLCUB16 );
	plotDrFieldCub16->SetLineWidth( LINEWIDTH );
	plotDrFieldCub16->SetLineColor( COLCUB16 );
	plotDrFieldCub16->SetMarkerSize( 0.2 );
	plotDrFieldCub16->SetMarkerStyle( 8 );
	if( PLOTCUB16 ) mgField->Add( plotDrFieldCub16 );

	TGraph* plotDrFieldCub19 = new TGraph( kmax+1 );
	plotDrFieldCub19->SetTitle( "Relative error of triangle 19-point cubature potential" );
	plotDrFieldCub19->SetDrawOption( "same" );
	plotDrFieldCub19->SetMarkerColor( COLCUB19 );
	plotDrFieldCub19->SetLineWidth( LINEWIDTH );
	plotDrFieldCub19->SetLineColor( COLCUB19 );
	plotDrFieldCub19->SetMarkerSize( 0.2 );
	plotDrFieldCub19->SetMarkerStyle( 8 );
	if( PLOTCUB19 ) mgField->Add( plotDrFieldCub19 );

	TGraph* plotDrFieldCub33 = new TGraph( kmax+1 );
	plotDrFieldCub33->SetTitle( "Relative error of triangle 33-point cubature potential" );
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
	double relCub12Pot( 0. );
	double relCub16Pot( 0. );
	double relCub19Pot( 0. );
	double relCub33Pot( 0. );
	double relNumPot( 0. );

	double relAnaField( 0. );
	double relRwgField( 0. );
	double relCub4Field( 0. );
	double relCub7Field( 0. );
	double relCub12Field( 0. );
	double relCub16Field( 0. );
	double relCub19Field( 0. );
	double relCub33Field( 0. );
	double relNumField( 0. );

	const double targetAccuracy( ACCURACY );

	// iterate over distance ratios in log steps
	for( unsigned int k=0; k<=kmax; k++ ) {

		Dr = minDr * exp(C*k);

		KEMField::cout << "Current distance ratio: " << Dr << "\t\r";
		KEMField::cout.flush();

		unsigned int directionIndex( 0 );

		// iterate over container elements
		for( it=container->begin<KElectrostaticBasis>(); it!=container->end<KElectrostaticBasis>(); ++it ) {

			IJKLRANDOM++;

			(*it)->Accept(fTriangleVisitor);

			// assign field point value
			fP = fTriangleVisitor.GetCentroid() + fTriangleVisitor.GetAverageSideLength()*Dr*fPointDirections[directionIndex];

			directionIndex++;

			fComputeVisitor.SetPosition(fP);

			(*it)->Accept(fComputeVisitor);

			valQuad = fComputeVisitor.GetQuadElectricFieldAndPotential();
			valAna = fComputeVisitor.GetAnaElectricFieldAndPotential();
			valRwg = fComputeVisitor.GetRwgElectricFieldAndPotential();
			valCub[0] = fComputeVisitor.GetCub4ElectricFieldAndPotential();
			valCub[1] = fComputeVisitor.GetCub7ElectricFieldAndPotential();
			valCub[2] = fComputeVisitor.GetCub12ElectricFieldAndPotential();
			valCub[3] = fComputeVisitor.GetCub16ElectricFieldAndPotential();
			valCub[4] = fComputeVisitor.GetCub19ElectricFieldAndPotential();
			valCub[5] = fComputeVisitor.GetCub33ElectricFieldAndPotential();
			valNum = fComputeVisitor.GetNumElectricFieldAndPotential();

			// sum for relative error

			relAnaPot += fabs((valAna.second-valQuad.second)/valQuad.second);
			relRwgPot += fabs((valRwg.second-valQuad.second)/valQuad.second);
			relCub4Pot += fabs((valCub[0].second-valQuad.second)/valQuad.second);
			relCub7Pot += fabs((valCub[1].second-valQuad.second)/valQuad.second);
			relCub12Pot += fabs((valCub[2].second-valQuad.second)/valQuad.second);
			relCub16Pot += fabs((valCub[3].second-valQuad.second)/valQuad.second);
			relCub19Pot += fabs((valCub[4].second-valQuad.second)/valQuad.second);
			relCub33Pot += fabs((valCub[5].second-valQuad.second)/valQuad.second);
			relNumPot += fabs((valNum.second-valQuad.second)/valQuad.second);

			const double mag = sqrt(POW2(valQuad.first[0]) + POW2(valQuad.first[1]) + POW2(valQuad.first[2]));

			for( unsigned short i=0; i<3; i++ ) {
				relAnaField += fabs(valAna.first[i]-valQuad.first[i])/(mag);
				relRwgField += fabs(valRwg.first[i]-valQuad.first[i])/(mag);
				relCub4Field += fabs(valCub[0].first[i]-valQuad.first[i])/(mag);
				relCub7Field += fabs(valCub[1].first[i]-valQuad.first[i])/(mag);
				relCub12Field += fabs(valCub[2].first[i]-valQuad.first[i])/(mag);
				relCub16Field += fabs(valCub[3].first[i]-valQuad.first[i])/(mag);
				relCub19Field += fabs(valCub[4].first[i]-valQuad.first[i])/(mag);
				relCub33Field += fabs(valCub[5].first[i]-valQuad.first[i])/(mag);
				relNumField += fabs(valNum.first[i]-valQuad.first[i])/(mag);
			}
		}

		relAnaPot /= Num;
		relRwgPot /= Num;
		relCub4Pot /= Num;
		relCub7Pot /= Num;
		relCub12Pot /= Num;
		relCub16Pot /= Num;
		relCub19Pot /= Num;
		relCub33Pot /= Num;
		relNumPot /= Num;

		relAnaField /= Num;
		relRwgField /= Num;
		relCub4Field /= Num;
		relCub7Field /= Num;
		relCub12Field /= Num;
		relCub16Field /= Num;
		relCub19Field /= Num;
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
		if( (!accFlagPotCub12) && (relCub12Pot<=targetAccuracy) ) {
			drOptPotCub12 = Dr;
			accFlagPotCub12 = true;
		}
		if( (!accFlagPotCub16) && (relCub16Pot<=targetAccuracy) ) {
			drOptPotCub16 = Dr;
			accFlagPotCub16 = true;
		}
		if( (!accFlagPotCub19) && (relCub19Pot<=targetAccuracy) ) {
			drOptPotCub19 = Dr;
			accFlagPotCub19 = true;
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
		if( (!accFlagFieldCub12) && (relCub12Field<=targetAccuracy) ) {
			drOptFieldCub12 = Dr;
			accFlagFieldCub12 = true;
		}
		if( (!accFlagFieldCub16) && (relCub16Field<=targetAccuracy) ) {
			drOptFieldCub16 = Dr;
			accFlagFieldCub16 = true;
		}
		if( (!accFlagFieldCub19) && (relCub19Field<=targetAccuracy) ) {
			drOptFieldCub19 = Dr;
			accFlagFieldCub19 = true;
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
		if( PLOTCUB12 ) plotDrPotCub12->SetPoint( k, Dr, relCub12Pot );
		if( PLOTCUB16 ) plotDrPotCub16->SetPoint( k, Dr, relCub16Pot );
		if( PLOTCUB19 ) plotDrPotCub19->SetPoint( k, Dr, relCub19Pot );
		if( PLOTCUB33 ) plotDrPotCub33->SetPoint( k, Dr, relCub33Pot );
		if( PLOTNUM ) plotDrPotNum->SetPoint( k, Dr, relNumPot );

		// reset relative error
		relAnaPot = 0.;
		relRwgPot = 0.;
		relCub4Pot = 0.;
		relCub7Pot = 0.;
		relCub12Pot = 0.;
		relCub16Pot = 0.;
		relCub19Pot = 0.;
		relCub33Pot = 0.;
		relNumPot = 0.;

		if( PLOTANA ) plotDrFieldAna->SetPoint( k, Dr, relAnaField );
		if( PLOTRWG ) plotDrFieldRwg->SetPoint( k, Dr, relRwgField );
		if( PLOTCUB4 ) plotDrFieldCub4->SetPoint( k, Dr, relCub4Field );
		if( PLOTCUB7 ) plotDrFieldCub7->SetPoint( k, Dr, relCub7Field );
		if( PLOTCUB12 ) plotDrFieldCub12->SetPoint( k, Dr, relCub12Field );
		if( PLOTCUB16 ) plotDrFieldCub16->SetPoint( k, Dr, relCub16Field );
		if( PLOTCUB19 ) plotDrFieldCub19->SetPoint( k, Dr, relCub19Field );
		if( PLOTCUB33 ) plotDrFieldCub33->SetPoint( k, Dr, relCub33Field );
		if( PLOTNUM ) plotDrFieldNum->SetPoint( k, Dr, relNumField );

		relAnaField = 0.;
		relRwgField = 0.;
		relCub4Field = 0.;
		relCub7Field = 0.;
		relCub12Field = 0.;
		relCub16Field = 0.;
		relCub19Field = 0.;
		relCub33Field = 0.;
		relNumField = 0.;
	} /* distance ratio */

	const double drAdd( DRADDPERC/100. );

	KEMField::cout << "Recommended distance ratio values for target accuracy " << targetAccuracy << " (+" << 100*drAdd << "%):" << KEMField::endl;
	KEMField::cout << "Triangle potentials:" << KEMField::endl;
	KEMField::cout << "*  4-point cubature: " << ((1.+drAdd)*drOptPotCub4) << KEMField::endl;
	KEMField::cout << "*  7-point cubature: " << ((1.+drAdd)*drOptPotCub7) << KEMField::endl;
	KEMField::cout << "* 12-point cubature: " << ((1.+drAdd)*drOptPotCub12) << KEMField::endl;
	KEMField::cout << "* 16-point cubature: " << ((1.+drAdd)*drOptPotCub16) << KEMField::endl;
	KEMField::cout << "* 19-point cubature: " << ((1.+drAdd)*drOptPotCub19) << KEMField::endl;
	KEMField::cout << "* 33-point cubature: " << ((1.)*drOptPotCub33) << " (no tolerance set here)" << KEMField::endl;
	KEMField::cout << "Triangle fields (valid for all functions, implemented in integrator classes):" << KEMField::endl;
	KEMField::cout << "*  4-point cubature: " << ((1.+drAdd)*drOptFieldCub4) << KEMField::endl;
	KEMField::cout << "*  7-point cubature: " << ((1.+drAdd)*drOptFieldCub7) << KEMField::endl;
	KEMField::cout << "* 12-point cubature: " << ((1.+drAdd)*drOptFieldCub12) << KEMField::endl;
	KEMField::cout << "* 16-point cubature: " << ((1.+drAdd)*drOptFieldCub16) << KEMField::endl;
	KEMField::cout << "* 19-point cubature: " << ((1.+drAdd)*drOptFieldCub19) << KEMField::endl;
	KEMField::cout << "* 33-point cubature: " << ((1.)*drOptFieldCub33) << " (no tolerance set here)" << KEMField::endl;

	KEMField::cout << "Distance ratio analysis for cubature integrators finished." << KEMField::endl;

	TCanvas cPot("cPot","Averaged relative error of triangle potential", 0, 0, 960, 760);
	cPot.SetMargin(0.16,0.06,0.15,0.06);
	cPot.SetLogx();
	cPot.SetLogy();

	// multigraph, create plot
	cPot.cd();
	mgPot->Draw( "apl" );
	mgPot->SetTitle( "Averaged error of triangle potential" );
	mgPot->GetXaxis()->SetTitle( "distance ratio" );
	mgPot->GetXaxis()->CenterTitle();
	mgPot->GetYaxis()->SetTitle( "relative error" );
	mgPot->GetYaxis()->CenterTitle();

	TLatex l;
	l.SetTextAlign(11);
	l.SetTextFont(62);
	l.SetTextSize(0.032);

	if( (PLOTRWG)|(PLOTANA) ) {
		l.SetTextAngle(29);
		if( PLOTRWG ) l.SetTextColor( COLRWG );
		if( (PLOTANA)&&!(PLOTRWG) ) l.SetTextColor( COLANA );
		if( PLOTRWG ) l.DrawLatex(500,1.5e-9,"Analytical (RWG)");
		if( (PLOTANA)&&!(PLOTRWG) ) l.DrawLatex(500,1.5e-9,"Analytical");
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

	if( PLOTCUB12 ) {
		l.SetTextAngle(-65);
		l.SetTextColor( COLCUB12 );
		l.DrawLatex(2.9,1.e-8,"12-point cubature");
	}

	if( PLOTCUB16 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB16 );
		l.DrawLatex(2.5,3e-10,"16-point cubature");
	}

	if( PLOTCUB19 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB19 );
		l.DrawLatex(2.5,3e-10,"19-point cubature");
	}

	if( PLOTCUB33 ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLCUB33 );
		l.DrawLatex(2.6,1.e-16,"33-point cubature");
	}

	if( PLOTNUM ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.6,1.e-16,"Numerical cubature + analytical RWG");
	}

	cPot.Update();

	TCanvas cField("cField","Averaged relative error of triangle field", 0, 0, 960, 760);
	cField.SetMargin(0.16,0.06,0.15,0.06);
	cField.SetLogx();
	cField.SetLogy();

	// multigraph, create plot
	cField.cd();
	mgField->Draw( "apl" );
	mgField->SetTitle( "Averaged error of triangle field" );
	mgField->GetXaxis()->SetTitle( "distance ratio" );
	mgField->GetXaxis()->CenterTitle();
	mgField->GetYaxis()->SetTitle( "relative error" );
	mgField->GetYaxis()->CenterTitle();

	if( (PLOTRWG)|(PLOTANA) ) {
		l.SetTextAngle(29);
		if( PLOTRWG ) l.SetTextColor( COLRWG );
		if( (PLOTANA)&&!(PLOTRWG) ) l.SetTextColor( COLANA );
		if( PLOTRWG ) l.DrawLatex(500,2.0e-9,"Analytical (RWG)");
		if( (PLOTANA)&&!(PLOTRWG) ) l.DrawLatex(500,2.0e-9,"Analytical");
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

	if( PLOTCUB12 ) {
		l.SetTextAngle(-65);
		l.SetTextColor( COLCUB12 );
		l.DrawLatex(2.9,1.e-8,"12-point cubature");
	}

	if( PLOTCUB16 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB16 );
		l.DrawLatex(2.5,3e-10,"16-point cubature");
	}

	if( PLOTCUB19 ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLCUB19 );
		l.DrawLatex(2.5,3e-10,"19-point cubature");
	}

	if( PLOTCUB33 ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLCUB33 );
		l.DrawLatex(2.7,1.2e-16,"33-point cubature");
	}

	if( PLOTNUM ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.7,1.2e-16,"Numerical cubature + analytical RWG");
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
