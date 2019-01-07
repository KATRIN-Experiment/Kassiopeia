#include <iostream>
#include <cstdlib>

#include "KThreeVector_KEMField.hh"
#include "KSurfaceContainer.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh"
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
#define NUMLINES 500       // number of line segments for each Dr step
#define MINDR 2            // minimal distance ratio to be investigated
#define MAXDR 10000        // maximal distance ratio to be investigated
#define STEPSDR 500        // steps between given distance ratio range
#define SEPARATECOMP	   // if this variable has been defined potentials and fields will be computed separately,
						   // hence 'ElectricFieldAndPotential' function won't be used
						   // both options have to produce same values

// ROOT PLOTS AND COLORS (all settings apply for both field and potential)
#define PLOTANA 1
#define PLOTNUM 1

#define COLANA kRed
#define COLNUM kBlue

#define LINEWIDTH 1.

using namespace KEMField;

double IJKLRANDOM;
typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment> KEMLineSegment;
void subrn(double *u,int len);
double randomnumber();

void printVec( std::string add, KThreeVector input )
{
	std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

namespace KEMField{

// visitor for line segment geometry

class LineSegmentVisitor :
		public KSelectiveVisitor<KShapeVisitor,
		KTYPELIST_1(KLineSegment)>
{
public:
	using KSelectiveVisitor<KShapeVisitor,KTYPELIST_1(KLineSegment)>::Visit;

	LineSegmentVisitor(){}

	void Visit(KLineSegment& t) { ProcessLineSegment(t); }

	void ProcessLineSegment(KLineSegment& l)
	{
		fLength = sqrt( POW2(l.GetP1()[0] - l.GetP0()[0]) + POW2(l.GetP1()[1] - l.GetP0()[1]) + POW2(l.GetP1()[2] - l.GetP0()[2]) );

		// centroid
		fShapeCentroid = l.Centroid();
	}

	double GetLength() { return fLength; }
	KThreeVector GetCentroid(){ return fShapeCentroid; }

private:
	double fLength;
	KThreeVector fShapeCentroid;
};

} /* KEMField namespace*/

int main()
{
	// This program determines the accuracy of the line segment integrators for a given distance ratio range.
	// distance ratio = distance to centroid / average side length

	// line segment data
	double P0[3];
	double P1[3];
	double length;

	// assign a unique direction vector for field point to each line segment and save into std::vector
	std::vector<KThreeVector> fPointDirections;

	// 'Num' line segments will be diced in the beginning and added to a surface container
	// This value decides how much 'line segments=field points' will be computed for each distance ratio value

	KSurfaceContainer* container = new KSurfaceContainer();
	const unsigned int Num( NUMLINES ); /* number of line segments */

	for( unsigned int i=0; i<Num; i++ ) {
		IJKLRANDOM = i+1;
		KEMLineSegment* line = new KEMLineSegment();

		// dice line segment geometry, diameter fixed ratio to length
		for( unsigned short l=0; l<3; l++ ) P0[l]=-1.+2.*randomnumber();
		for( unsigned short j=0; j<3; j++ ) P1[j]=-1.+2.*randomnumber();

		// compute further line segment data

		length = sqrt( POW2(P1[0]-P0[0]) + POW2(P1[1]-P0[1]) + POW2(P1[2]-P0[2]) );

		line->SetP0( KThreeVector(P0[0],P0[1],P0[2]) );
		line->SetP1( KThreeVector(P1[0],P1[1],P1[2]) );
		line->SetDiameter( length*0.1 );

		line->SetBoundaryValue( 1. );
		line->SetSolution( 1. );

		container->push_back( line );

		const double costhetaFP = -1.+2.*randomnumber();
		const double sinthetaFP = sqrt( 1. - POW2(costhetaFP) );
		const double phiFP = 2.*M_PI*randomnumber();

		fPointDirections.push_back( KThreeVector(
				sinthetaFP*cos(phiFP),
				sinthetaFP*sin(phiFP),
				costhetaFP ) );
	}

    // OpenCL surface container
    KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(*container);
    KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );

    // Quadrature and OpenCL integrator classes
    KElectrostatic256NodeQuadratureLineSegmentIntegrator fQuadIntegrator;
    KOpenCLElectrostaticBoundaryIntegrator intOCLAna {
    	KoclEBIFactory::MakeAnalytic( *oclContainer )};
    KOpenCLElectrostaticBoundaryIntegrator intOCLNum {
    	KoclEBIFactory::MakeNumeric( *oclContainer )};

	// visitor for elements
	LineSegmentVisitor fLineSegmentVisitor;

	KSurfaceContainer::iterator it;

	// distance ratios
	const double minDr( MINDR );
	const double maxDr( MAXDR );
	double Dr = 0.;
	const unsigned int kmax( STEPSDR);
	const double C = log(maxDr/minDr)/kmax;

	KEMField::cout << "Iterate from dist. ratio " << minDr << " to " << maxDr << " in " << kmax << " steps." << KEMField::endl;
	KEMField::cout << "Taking averaged relative error for " << container->size() << " line segments for each dist. ratio value." << KEMField::endl;

	// field point
	KThreeVector fP;

	std::pair<KThreeVector,double> valQuad256;
	std::pair<KThreeVector,double> valAna;
	std::pair<KThreeVector,double> valNum;

	// plot

	TApplication* fAppWindow = new TApplication("fAppWindow", 0, NULL);

	gStyle->SetCanvasColor( kWhite );
	gStyle->SetLabelOffset( 0.03, "xyz" ); // values
	gStyle->SetTitleOffset( 1.6, "xyz" ); // label

	TMultiGraph *mgPot = new TMultiGraph();

	TGraph* plotDrPotAna = new TGraph( kmax+1 );
	plotDrPotAna->SetTitle( "Relative error of analytical line segment potential" );
	plotDrPotAna->SetDrawOption( "AC" );
	plotDrPotAna->SetMarkerColor( COLANA );
	plotDrPotAna->SetLineWidth( LINEWIDTH );
	plotDrPotAna->SetLineColor( COLANA );
	plotDrPotAna->SetMarkerSize( 0.2 );
	plotDrPotAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgPot->Add( plotDrPotAna );

	TGraph* plotDrPotNum = new TGraph( kmax+1 );
	plotDrPotNum->SetTitle( "Relative error of line segment potential with adjusted numerical integrator" );
	plotDrPotNum->SetDrawOption( "same" );
	plotDrPotNum->SetMarkerColor( COLNUM );
	plotDrPotNum->SetLineWidth( LINEWIDTH );
	plotDrPotNum->SetLineColor( COLNUM );
	plotDrPotNum->SetMarkerSize( 0.2 );
	plotDrPotNum->SetMarkerStyle( 8 );
	if( PLOTNUM ) mgPot->Add( plotDrPotNum );

	TMultiGraph *mgField = new TMultiGraph();

	TGraph* plotDrFieldAna = new TGraph( kmax+1 );
	plotDrFieldAna->SetTitle( "Relative error of analytical line segment field" );
	plotDrFieldAna->SetDrawOption( "AC" );
	plotDrFieldAna->SetMarkerColor( COLANA );
	plotDrFieldAna->SetLineWidth( LINEWIDTH );
	plotDrFieldAna->SetLineColor( COLANA );
	plotDrFieldAna->SetMarkerSize( 0.2 );
	plotDrFieldAna->SetMarkerStyle( 8 );
	if( PLOTANA ) mgField->Add( plotDrFieldAna );

	TGraph* plotDrFieldNum = new TGraph( kmax+1 );
	plotDrFieldNum->SetTitle( "Relative error of triangle field with adjusted numerical integrator" );
	plotDrFieldNum->SetDrawOption( "same" );
	plotDrFieldNum->SetMarkerColor( COLNUM );
	plotDrFieldNum->SetLineWidth( LINEWIDTH );
	plotDrFieldNum->SetLineColor( COLNUM );
	plotDrFieldNum->SetMarkerSize( 0.2 );
	plotDrFieldNum->SetMarkerStyle( 8 );
	if( PLOTNUM ) mgField->Add( plotDrFieldNum );

	double relAnaPot( 0. );
	double relNumPot( 0. );

	double relAnaField( 0. );
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

			(*it)->Accept(fLineSegmentVisitor);

			// assign field point value
			fP = fLineSegmentVisitor.GetCentroid() + fLineSegmentVisitor.GetLength()*Dr*fPointDirections[directionIndex];

			directionIndex++;

			KEMLineSegment* itLine;
            itLine = static_cast<KEMLineSegment*>((*it));

#ifdef SEPARATECOMP
            valQuad256 = std::make_pair( fQuadIntegrator.ElectricField(itLine->GetShape(),fP), fQuadIntegrator.Potential(itLine->GetShape(),fP));
            valAna = std::make_pair(intOCLAna.ElectricField(itLine->GetShape(),fP),intOCLAna.Potential(itLine->GetShape(),fP));
            valNum = std::make_pair(intOCLNum.ElectricField(itLine->GetShape(),fP),intOCLNum.Potential(itLine->GetShape(),fP));
#else
            valQuad256 = fQuadIntegrator.ElectricFieldAndPotential(itLine->GetShape(),fP);
            valAna = intOCLAna.ElectricFieldAndPotential(itLine->GetShape(),fP);
            valNum = intOCLNum.ElectricFieldAndPotential(itLine->GetShape(),fP);
#endif

			// sum for relative error
			relAnaPot += fabs((valAna.second-valQuad256.second)/valQuad256.second);
			relNumPot += fabs((valNum.second-valQuad256.second)/valQuad256.second);

			const double mag = sqrt(POW2(valQuad256.first[0]) + POW2(valQuad256.first[1]) + POW2(valQuad256.first[2]));

			for( unsigned short i=0; i<3; i++ ) {
				relAnaField += fabs(valAna.first[i]-valQuad256.first[i])/(mag);
				relNumField += fabs(valNum.first[i]-valQuad256.first[i])/(mag);
			}
		}

		relAnaPot /= Num;
		relNumPot /= Num;

		relAnaField /= Num;
		relNumField /= Num;

		// save relative error of each integrator
		if( PLOTANA ) plotDrPotAna->SetPoint( k, Dr, relAnaPot );
		if( PLOTNUM ) plotDrPotNum->SetPoint( k, Dr, relNumPot );

		// reset relative error
		relAnaPot = 0.;
		relNumPot = 0.;

		if( PLOTANA ) plotDrFieldAna->SetPoint( k, Dr, relAnaField );
		if( PLOTNUM ) plotDrFieldNum->SetPoint( k, Dr, relNumField );

		relAnaField = 0.;
		relNumField = 0.;
	} /* distance ratio */

    KEMField::cout << "Computation finished." << KEMField::endl;

	TCanvas cPot("cPot","Averaged relative error of line segment potential", 0, 0, 960, 760);
	cPot.SetMargin(0.16,0.06,0.15,0.06);
	cPot.SetLogx();
	cPot.SetLogy();

	// multigraph, create plot
	cPot.cd();

	mgPot->Draw( "apl" );
	mgPot->SetTitle( "Averaged error of line segment potential" );
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
		l.DrawLatex(500,1.5e-9,"Analytical");
	}

	if( PLOTNUM ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.5,3e-10,"Numerical quadrature + analytical");
	}

	cPot.Update();

	TCanvas cField("cField","Averaged relative error of line segment field", 0, 0, 960, 760);
	cField.SetMargin(0.16,0.06,0.15,0.06);
	cField.SetLogx();
	cField.SetLogy();

	// multigraph, create plot
	cField.cd();
	mgField->Draw( "apl" );
	mgField->SetTitle( "Averaged error of line segment field" );
	mgField->GetXaxis()->SetTitle( "distance ratio" );
	mgField->GetXaxis()->CenterTitle();
	mgField->GetYaxis()->SetTitle( "relative error" );
	mgField->GetYaxis()->CenterTitle();

	if( PLOTANA ) {
		l.SetTextAngle(29);
		l.SetTextColor( COLANA );
		l.DrawLatex(500,1.5e-9,"Analytical");
	}

	if( PLOTNUM ) {
		l.SetTextAngle(-69);
		l.SetTextColor( COLNUM );
		l.DrawLatex(2.5,3e-10,"Numerical quadrature + analytical");
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
