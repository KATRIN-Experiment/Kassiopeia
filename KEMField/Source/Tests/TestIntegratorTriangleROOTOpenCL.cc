#include <iostream>
#include <cstdlib>

#include "KEMThreeVector.hh"
#include "KSurfaceContainer.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"

#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"

#include "TStyle.h"
#include "TApplication.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TLatex.h"

#define POW2(x) ((x)*(x))

// VALUES
#define NUMTRIANGLES 500   // number of triangles for each Dr step
#define MINDR 2            // minimal distance ratio to be investigated
#define MAXDR 10000        // maximal distance ratio to be investigated
#define STEPSDR 500        // steps between given distance ratio range
#define SEPARATECOMP	   // if this variable has been defined potentials and fields will be computed separately,
						   // hence 'ElectricFieldAndPotential' function won't be used
						   // both options have to produce same values

// ROOT PLOTS AND COLORS (all settings apply for both field and potential)
#define PLOTANA 1
#define PLOTRWG 1
#define PLOTNUM 1

#define COLANA kBlue
#define COLRWG kGreen
#define COLNUM kRed

#define LINEWIDTH 1.

using namespace KEMField;

double IJKLRANDOM;
typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle> KEMTriangle;
void subrn(double *u,int len);
double randomnumber();

void printVec( std::string add, KEMThreeVector input )
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
	KEMThreeVector GetCentroid(){ return fShapeCentroid; }

private:
	double fAverageSideLength;
	KEMThreeVector fShapeCentroid;
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
	std::vector<KEMThreeVector> fPointDirections;

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
		triangle->SetP0( KEMThreeVector(P0[0],P0[1],P0[2]) );
		triangle->SetN1( KEMThreeVector(N1[0],N1[1],N1[2]) );
		triangle->SetN2( KEMThreeVector(N2[0],N2[1],N2[2]) );

		triangle->SetBoundaryValue( 1. );
		triangle->SetSolution( 1. );

		container->push_back( triangle );

		const double costhetaFP = -1.+2.*randomnumber();
		const double sinthetaFP = sqrt( 1. - POW2(costhetaFP) );
		const double phiFP = 2.*M_PI*randomnumber();

		fPointDirections.push_back( KEMThreeVector(
				sinthetaFP*cos(phiFP),
				sinthetaFP*sin(phiFP),
				costhetaFP ) );
	}

	// OpenCL surface container
    KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(*container);
    KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );

    // Bi-Quadrature and OpenCL integrator classes
    KElectrostaticBiQuadratureTriangleIntegrator fQuadIntegrator;
    KOpenCLElectrostaticBoundaryIntegrator intOCLAna {
    	KoclEBIFactory::MakeAnalytic( *oclContainer )};
    KOpenCLElectrostaticBoundaryIntegrator intOCLNum {
    	KoclEBIFactory::MakeNumeric( *oclContainer )};
    KOpenCLElectrostaticBoundaryIntegrator intOCLRwg {
    	KoclEBIFactory::MakeRWG( *oclContainer )};

	// visitor for elements
	TriangleVisitor fTriangleVisitor;

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
	KEMThreeVector fP;

	// field and potential values
	std::pair<KEMThreeVector,double> valQuad;
	std::pair<KEMThreeVector,double> valAna;
	std::pair<KEMThreeVector,double> valRwg;
	std::pair<KEMThreeVector,double> valNum;

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
	plotDrFieldAna->SetTitle( "Relative error of analytical triangle field" );
	plotDrFieldAna->SetDrawOption( "AC" );
	plotDrFieldAna->SetMarkerColor( COLANA );
	plotDrFieldAna->SetLineWidth( LINEWIDTH );
	plotDrFieldAna->SetLineColor( COLANA );
	plotDrFieldAna->SetMarkerSize( 0.2 );
	plotDrFieldAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgField->Add( plotDrFieldAna );

	TGraph* plotDrFieldRwg = new TGraph( kmax+1 );
	plotDrFieldRwg->SetTitle( "Relative error of triangle RWG field" );
	plotDrFieldRwg->SetDrawOption( "same" );
	plotDrFieldRwg->SetMarkerColor( COLRWG );
	plotDrFieldRwg->SetLineWidth( LINEWIDTH );
	plotDrFieldRwg->SetLineColor( COLRWG );
	plotDrFieldRwg->SetMarkerSize( 0.2 );
	plotDrFieldRwg->SetMarkerStyle( 8 );
	if( PLOTRWG ) mgField->Add( plotDrFieldRwg );

	TGraph* plotDrFieldNum = new TGraph( kmax+1 );
	plotDrFieldNum->SetTitle( "Relative error of numerical triangle field" );
	plotDrFieldNum->SetDrawOption( "same" );
	plotDrFieldNum->SetMarkerColor( COLNUM );
	plotDrFieldNum->SetLineWidth( LINEWIDTH );
	plotDrFieldNum->SetLineColor( COLNUM );
	plotDrFieldNum->SetMarkerSize( 0.2 );
	plotDrFieldNum->SetMarkerStyle( 8 );
	if( PLOTNUM )mgField->Add( plotDrFieldNum );

	double relAnaPot( 0. );
	double relRwgPot( 0. );
	double relNumPot( 0. );

	double relAnaField( 0. );
	double relRwgField( 0. );
	double relNumField( 0. );

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

			KEMTriangle* itTri;
			itTri = static_cast<KEMTriangle*>((*it));

#ifdef SEPARATECOMP
			valQuad = std::make_pair( fQuadIntegrator.ElectricField(itTri->GetShape(),fP), fQuadIntegrator.Potential(itTri->GetShape(),fP));
			valAna = std::make_pair(intOCLAna.ElectricField(itTri->GetShape(),fP),intOCLAna.Potential(itTri->GetShape(),fP));
			valNum = std::make_pair(intOCLNum.ElectricField(itTri->GetShape(),fP),intOCLNum.Potential(itTri->GetShape(),fP));
			valRwg = std::make_pair(intOCLRwg.ElectricField(itTri->GetShape(),fP),intOCLRwg.Potential(itTri->GetShape(),fP));
#else
			valQuad = fQuadIntegrator.ElectricFieldAndPotential(itTri->GetShape(),fP);
			valAna = intOCLAna.ElectricFieldAndPotential(itTri->GetShape(),fP);
			valNum = intOCLNum.ElectricFieldAndPotential(itTri->GetShape(),fP);
			valRwg = intOCLRwg.ElectricFieldAndPotential(itTri->GetShape(),fP);
#endif

			// sum for relative error

			relAnaPot += fabs((valAna.second-valQuad.second)/valQuad.second);
			relRwgPot += fabs((valRwg.second-valQuad.second)/valQuad.second);
			relNumPot += fabs((valNum.second-valQuad.second)/valQuad.second);

			const double mag = sqrt(POW2(valQuad.first[0]) + POW2(valQuad.first[1]) + POW2(valQuad.first[2]));

			for( unsigned short i=0; i<3; i++ ) {
				relAnaField += fabs(valAna.first[i]-valQuad.first[i])/(mag);
				relRwgField += fabs(valRwg.first[i]-valQuad.first[i])/(mag);
				relNumField += fabs(valNum.first[i]-valQuad.first[i])/(mag);
			}
		}

		relAnaPot /= Num;
		relRwgPot /= Num;
		relNumPot /= Num;

		relAnaField /= Num;
		relRwgField /= Num;
		relNumField /= Num;

		// save relative error of each integrator
		if( PLOTANA ) plotDrPotAna->SetPoint( k, Dr, relAnaPot );
		if( PLOTRWG ) plotDrPotRwg->SetPoint( k, Dr, relRwgPot );
		if( PLOTNUM ) plotDrPotNum->SetPoint( k, Dr, relNumPot );

		// reset relative error
		relAnaPot = 0.;
		relRwgPot = 0.;
		relNumPot = 0.;

		if( PLOTANA ) plotDrFieldAna->SetPoint( k, Dr, relAnaField );
		if( PLOTRWG ) plotDrFieldRwg->SetPoint( k, Dr, relRwgField );
		if( PLOTNUM ) plotDrFieldNum->SetPoint( k, Dr, relNumField );

		relAnaField = 0.;
		relRwgField = 0.;
		relNumField = 0.;
	} /* distance ratio */

	KEMField::cout << "Computation finished." << KEMField::endl;

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

	if( PLOTANA ) {
		l.SetTextAngle(29);
		l.SetTextColor( COLANA );
		l.DrawLatex(400,1.5e-9,"Analytical");
	}

    if( PLOTRWG ) {
        l.SetTextAngle(29);
        l.SetTextColor( COLRWG );
        l.DrawLatex(500,1.5e-9,"RWG");
    }

	if( PLOTNUM ) {
		l.SetTextAngle(0);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.6,1.e-16,"Numerical (cubature + RWG)");
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

    if( PLOTANA ) {
        l.SetTextAngle(29);
        l.SetTextColor( COLANA );
        l.DrawLatex(400,1.5e-9,"Analytical");
    }

    if( PLOTRWG ) {
        l.SetTextAngle(29);
        l.SetTextColor( COLRWG );
        l.DrawLatex(500,1.5e-9,"RWG");
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
