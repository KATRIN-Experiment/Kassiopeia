#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KEMThreeVector.hh"

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"

#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"

#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KEMCout.hh"



using namespace KEMField;

void printVec( std::string add, KEMThreeVector input )
{
	std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

void printVecKEM( std::string add, KEMThreeVector input )
{
	KEMField::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << KEMField::endl;
}

int main()
{
	// Boundary integrators and visitors
	KElectrostaticBiQuadratureTriangleIntegrator intTriQuad;
	KElectrostaticAnalyticTriangleIntegrator intTriAna;
	KElectrostaticRWGTriangleIntegrator intTriRwg;

	KElectrostaticBiQuadratureRectangleIntegrator intReQuad;
    KElectrostaticAnalyticRectangleIntegrator intReAna;
    KElectrostaticRWGRectangleIntegrator intReRwg;

	// Triangle
	// --------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tri1 = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();

	KEMThreeVector tri1P0( -1.75, 0.25, 0. );
	tri1->SetP0( tri1P0) ;

	KEMThreeVector tri1P1( 1.75, 0.25, 0. );
	KEMThreeVector tri1N1 = (tri1P1-tri1P0).Unit();
	tri1->SetA( (tri1P1-tri1P0).Magnitude() );
	tri1->SetN1( tri1N1 );

	KEMThreeVector tri1P2( 0., 2.25, 0. );
	KEMThreeVector tri1N2 = (tri1P2-tri1P0).Unit();
	tri1->SetB( (tri1P2-tri1P0).Magnitude() );
	tri1->SetN2( tri1N2 );

	tri1->SetBoundaryValue( 10. ); // electric potential

	// triangle 1 from rectangle
	KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tri2 = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();

	KEMThreeVector tri2P0( -2., -2, 0. );
	tri2->SetP0( tri2P0) ;

	KEMThreeVector tri2P1( 2., -2., 0. );
	KEMThreeVector tri2N1 = (tri2P1-tri2P0).Unit();
	tri2->SetA( (tri2P1-tri2P0).Magnitude() );
	tri2->SetN1( tri2N1 );

	KEMThreeVector tri2P2( 2., 2., 0. );
	KEMThreeVector tri2N2 = (tri2P2-tri2P0).Unit();
	tri2->SetB( (tri2P2-tri2P0).Magnitude() );
	tri2->SetN2( tri2N2 );

	tri2->SetBoundaryValue( 10. ); // electric potential

	// Rectangles
	// ----------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>* rL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>();
	rL->SetA( 4. ); // positive x-direction
	rL->SetB( 4. ); // positive y-direction
	KEMThreeVector rLp0( -2., -2., 0. ); /* P0 */
	rL->SetP0(rLp0);
	KEMThreeVector rLn1( 1., 0., 0. ); /* N1 */
	rL->SetN1( rLn1 );
	KEMThreeVector rLn2( 0., 1., 0. ); /* N2 */
	rL->SetN2( rLn2 );
	//rL->SetSolution(12.); // charge density (electrostatic basis)
	rL->SetBoundaryValue( 10. ); // electric potential

	KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>* rR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>();
	rR->SetA( 1. ); // positive x-direction
	rR->SetB( 2. ); // positive y-direction
	KEMThreeVector rRp0( 0., 0., 0.9 ); /* P0 */
	rR->SetP0(rRp0);
	KEMThreeVector rRn1( 1., 0., 0. ); /* N1 */
	rR->SetN1( rRn1 );
	KEMThreeVector rRn2( 0., 1., 0. ); /* N2 */
	rR->SetN2( rRn2 );
	//rR->SetSolution(12.); // charge density (electrostatic basis)
	rR->SetBoundaryValue( 10. ); // electric potential


	std::cout << std::fixed << std::setprecision(8);

	// RECTANGLE

	KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>* testRect;
	testRect = rL;
	KEMThreeVector reEvalPoint[7];
	std::vector<std::string> textRe;

// points for rL

	textRe.push_back((std::string)("point in left lower corner (P0):\t"));
	reEvalPoint[0] = testRect->GetP0();

	textRe.push_back((std::string)("arbitrary point on surface:\t"));
	reEvalPoint[1] = KEMThreeVector( 0.1, -1.5, 0. );

	textRe.push_back((std::string)("point in center of rectangle:\t"));
	reEvalPoint[2] = testRect->Centroid();

	textRe.push_back((std::string)("point in plane of rectangle but far outside:\t"));
	reEvalPoint[3] = KEMThreeVector( 10., 10., 0. );

	textRe.push_back((std::string)("point on side line of rectangle:\t"));
	reEvalPoint[4] = KEMThreeVector( 0., -2., 0. );

	textRe.push_back((std::string)("arbitrary point:\t"));
	reEvalPoint[5] = KEMThreeVector( 10., -20.5, 33.58 );

	textRe.push_back((std::string)("point on line subdividing the rectangle into two triangles:\t"));
	reEvalPoint[6] = KEMThreeVector( -1, -1., 0. );

	KEMField::cout << "Testing RWG rectangles ..." << KEMField::endl;
	for( unsigned int i=0; i<textRe.size(); i++ ) {
		printVecKEM(textRe[i], reEvalPoint[i]);
		std::cout << "* Potential, Quadrature:  " << intReQuad.Potential(testRect,reEvalPoint[i]) << std::endl;
		std::cout << "* Potential, Analytical:  " << intReAna.Potential(testRect,reEvalPoint[i]) << std::endl;
		std::cout << "* Potential, RWG:         " << intReRwg.Potential(testRect,reEvalPoint[i]) << std::endl;
		printVec("* Electric Field, Quadrature: ", intReQuad.ElectricField(testRect,reEvalPoint[i]));
		printVec("* Electric Field, Analytical: ", intReAna.ElectricField(testRect,reEvalPoint[i]));
		printVec("* Electric Field, RWG:        ", intReRwg.ElectricField(testRect,reEvalPoint[i]));
		std::cout << std::endl;
	}

	// TRIANGLE

	KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* testTri;
	testTri = tri1;
	KEMThreeVector triEvalPoint[6];
	std::vector<std::string> textTri;

// points for tri1

	textTri.push_back((std::string)("Point in left lower corner (P0):\t"));
	triEvalPoint[0] = testTri->GetP0();

	textTri.push_back((std::string)("arbitrary point on surface:\t"));
	triEvalPoint[1] = KEMThreeVector( 0.01, 0.5, 0. );

	textTri.push_back((std::string)("point in center of triangle:\t"));
	triEvalPoint[2] = testTri->Centroid();

	textTri.push_back((std::string)("point in plane of triangle but far outside:\t"));
	triEvalPoint[3] = KEMThreeVector( 10., 1., 0. );

	textTri.push_back((std::string)("point on side line of triangle:\t"));
	triEvalPoint[4] = KEMThreeVector( 0., 0.25, 0. );

	textTri.push_back((std::string)("arbitrary point:\t"));
	triEvalPoint[5] = KEMThreeVector( 10., -20.5, 33.58 );

	KEMField::cout << "Testing RWG triangles ..." << KEMField::endl;
	for( unsigned int i=0; i<textTri.size(); i++ ) {
		printVecKEM(textTri[i], triEvalPoint[i]);
		std::cout << "* Potential, Quadrature:  " << intTriQuad.Potential(testTri,triEvalPoint[i]) << std::endl;
		std::cout << "* Potential, Analytical:  " << intTriAna.Potential(testTri,triEvalPoint[i]) << std::endl;
		std::cout << "* Potential, RWG:         " << intTriRwg.Potential(testTri,triEvalPoint[i]) << std::endl;
		printVec("* Electric Field, Quadrature: ", intTriQuad.ElectricField(testTri,triEvalPoint[i]));
		printVec("* Electric Field, Analytical: ", intTriAna.ElectricField(testTri,triEvalPoint[i]));
		printVec("* Electric Field, RWG:        ", intTriRwg.ElectricField(testTri,triEvalPoint[i]));
		std::cout << std::endl;
	}




	return 0;
}
