#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "KThreeVector_KEMField.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"

#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KElectrostaticCubatureTriangleIntegrator.hh"

#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticCubatureRectangleIntegrator.hh"

#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"
#include "KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh"

using namespace KEMField;

void printVec( std::string add, KThreeVector input )
{
	std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

int main()
{
    // field points

    KPosition test1( 2., 0., 7. );
    KPosition test2a( 0., 0., 0. );
    KPosition test2b( 0., 0.1, 0. ); /* analytic -> nan (field) */
    KPosition test3( 4.5, 2., 500. );

    // evaluation point for computation
    KPosition evalPoint = test1;

	// Triangles
	// ---------

//	KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
//	tL->SetA( 1. ); // positive x-direction
//	tL->SetB( 2. ); // positive y-direction
//	KThreeVector tLp0( 3., 5., -6. ); /* P0 */
//	tL->SetP0(tLp0);
//	KThreeVector tLn1( 1./sqrt(2.), 1./sqrt(2.), 0. ); /* N1 */
//	tL->SetN1( tLn1 );
//	KThreeVector tLn2( 0., 1./sqrt(2.), 1./sqrt(2.) ); /* N2 */
//	tL->SetN2( tLn2 );
//	//tL->SetSolution(1.); // charge density (electrostatic basis)
//	tL->SetBoundaryValue( 100. ); // electric potential

    // triangle for 'test1'

    KPosition P0( 2.0000000000000000, 0.0000000000000000, 10.3361800000000006 );
    KPosition N1( -0.0034906496776776, 0.9999939076638557, 0.0000000000000000 );
    KPosition N2( -0.8157191387677045, 0.1280630217856991, 0.5640940959620033 );
    const double a( 0.0139626012892605 );
    const double b( 0.1022139826116594 );

    // triangles for 'test2a/b'

    // i = 1481254

//    KPosition P0( -0.0065403129230143, 0.0997858923238604, 15.7968499999999992 );
//    KPosition N1( 0.0000000000000000, -0.0000000000000011, -1.0000000000000000 );
//    KPosition N2( 0.9994645874763657, 0.0327190828217764, 0.0000000000000000 );
//    const double a( 0.0127500000000005 );
//    const double b( 0.0065438165643552 );

    // i = 1481255

//    KPosition P0( -0.0000000000000000, 0.1000000000000000, 15.7840999999999987 );
//    KPosition N1(  0.0000000000000000, 0.0000000000000011, 1.0000000000000000 );
//    KPosition N2( -0.9994645874763657, -0.0327190828217764, 0.0000000000000000 );
//    const double a( 0.0127500000000005 );
//    const double b( 0.0065438165643552 );

    // i = 1481256

//    KPosition P0( -0.0000000000000000, 0.1000000000000000, 15.7968499999999992 );
//    KPosition N1(  0.0000000000000000, -0.0000000000000011, -1.0000000000000000 );
//    KPosition N2( 0.9994645874763658, -0.0327190828217743, 0.0000000000000000 );
//    const double a( 0.0127500000000005 );
//    const double b( 0.0065438165643552 );

    // i = 1481257

//    KPosition P0( 0.0065403129230142, 0.0997858923238604, 15.7840999999999987 );
//    KPosition N1( 0.0000000000000000, 0.0000000000000011, 1.0000000000000000 );
//    KPosition N2( -0.9994645874763658, 0.0327190828217743, 0.0000000000000000 );
//    const double a( 0.0127500000000005 );
//    const double b( 0.0065438165643552 );

    // triangles for 'test2'

    KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
    tL->SetA( a ); // positive x-direction
    tL->SetB( b ); // positive y-direction
    tL->SetP0( P0 );
    tL->SetN1( N1 );
    tL->SetN2( N2 );
    tL->SetSolution( 1. ); // charge density (electrostatic basis)
    tL->SetBoundaryValue( 1. ); // electric potential

    // triangle data
	const double tLdata[11] = {tL->GetA(),
			tL->GetB(),
			tL->GetP0().X(),
			tL->GetP0().Y(),
			tL->GetP0().Z(),
			tL->GetN1().X(),
			tL->GetN1().Y(),
			tL->GetN1().Z(),
			tL->GetN2().X(),
			tL->GetN2().Y(),
			tL->GetN2().Z()
	};

	KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
	tR->SetA( 1.133 ); // positive x-direction
	tR->SetB( 2.2323 ); // positive y-direction
	KThreeVector tRp0( 0., 0., 1. ); /* P0 */
	tR->SetP0(tRp0);
	KThreeVector tRn1( 1., 0., 0. ); /* N1 */
	tR->SetN1( tRn1 );
	KThreeVector tRn2( 0., 1., 0. ); /* N2 */
	tR->SetN2( tRn2 );
	//tR->SetSolution(12.); // charge density (electrostatic basis)
	tR->SetBoundaryValue( -100. ); // electric potential


	// Rectangles
	// ----------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>* rL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>();
	rL->SetA( 3. ); // positive x-direction
	rL->SetB( 2. ); // positive y-direction
	KThreeVector rLp0( 0., 0., -0.9 ); /* P0 */
	rL->SetP0(rLp0);
	KThreeVector rLn1( 1., 0., 0. ); /* N1 */
	rL->SetN1( rLn1 );
	KThreeVector rLn2( 0., 1., 0. ); /* N2 */
	rL->SetN2( rLn2 );
	//rL->SetSolution(12.); // charge density (electrostatic basis)
	rL->SetBoundaryValue( -200. ); // electric potential

    // rectangle data
	const double rLdata[11] = {rL->GetA(),
			rL->GetB(),
			rL->GetP0().X(),
			rL->GetP0().Y(),
			rL->GetP0().Z(),
			rL->GetN1().X(),
			rL->GetN1().Y(),
			rL->GetN1().Z(),
			rL->GetN2().X(),
			rL->GetN2().Y(),
			rL->GetN2().Z()
	};

	KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>* rR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>();
	rR->SetA( 1. ); // positive x-direction
	rR->SetB( 2. ); // positive y-direction
	KThreeVector rRp0( 0., 0., 0.9 ); /* P0 */
	rR->SetP0(rRp0);
	KThreeVector rRn1( 1., 0., 0. ); /* N1 */
	rR->SetN1( rRn1 );
	KThreeVector rRn2( 0., 1., 0. ); /* N2 */
	rR->SetN2( rRn2 );
	//rR->SetSolution(12.); // charge density (electrostatic basis)
	rR->SetBoundaryValue( 200. ); // electric potential


	// Line Segments
	// -------------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment>* wL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment>();
	wL->SetP0(KThreeVector(0.1,-1.5,-0.5));
	wL->SetP1(KThreeVector(0.1,1.,-0.5));
	wL->SetDiameter(0.003);
	wL->SetBoundaryValue(-1000);

	const double wLdata[7] = {
			wL->GetP0().X(),
			wL->GetP0().Y(),
			wL->GetP0().Z(),
			wL->GetP1().X(),
			wL->GetP1().Y(),
			wL->GetP1().Z(),
			wL->GetDiameter()
	};

	KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment>* wR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment>();
	wR->SetP0(KThreeVector(0.1,-1.,0.5));
	wR->SetP1(KThreeVector(0.1,1.,0.5));
	wR->SetDiameter(0.003);
	wR->SetBoundaryValue(-1000);

	// Boundary integrators and visitors
	KElectrostaticAnalyticTriangleIntegrator intTriAna;
	KElectrostaticRWGTriangleIntegrator intTriRwg;
	KElectrostaticCubatureTriangleIntegrator intTriCub;
    KElectrostaticBiQuadratureTriangleIntegrator intTriQuad;

    KElectrostaticAnalyticRectangleIntegrator intReAna;
    KElectrostaticRWGRectangleIntegrator intReRwg;
    KElectrostaticCubatureRectangleIntegrator intReCub;
    KElectrostaticBiQuadratureRectangleIntegrator intReQuad;

    KElectrostaticAnalyticLineSegmentIntegrator intLineAna;
    KElectrostaticQuadratureLineSegmentIntegrator intLineNum;
    KElectrostatic256NodeQuadratureLineSegmentIntegrator intLineQuad;

    // ---------------------------------------------------------------------------------

    std::cout << "\n----------" << std::endl;
	std::cout << "Potentials" << std::endl;
	std::cout << "----------" << std::endl;

	std::cout << "TRIANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	std::cout << std::fixed << std::setprecision(16);
	std::cout << "\t CPU:       " << intTriAna.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriAna.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Cubature:" << std::endl;

	// computing Q-points for triangle tL

	double triQ4[12];
	intTriCub.GaussPoints_Tri4P(tLdata,triQ4);
	double triQ7[21];
	intTriCub.GaussPoints_Tri7P(tLdata,triQ7);
	double triQ12[36];
	intTriCub.GaussPoints_Tri12P(tLdata,triQ12);
	double triQ16[48];
	intTriCub.GaussPoints_Tri16P(tLdata,triQ16);
	double triQ19[57];
	intTriCub.GaussPoints_Tri19P(tLdata,triQ19);
	double triQ33[99];
	intTriCub.GaussPoints_Tri33P(tLdata,triQ33);

	// 4-point
	std::cout << "\t CPU, n=4p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 4, triQ4, gTriCub4w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 4, triQ4, gTriCub4w ).second << std::endl;
	// 7-point
	std::cout << "\t CPU, n=7p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 7, triQ7, gTriCub7w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 7, triQ7, gTriCub7w ).second << std::endl;
	// 12-point
	std::cout << "\t CPU,n=12p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 12, triQ12, gTriCub12w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 12, triQ12, gTriCub12w ).second << std::endl;
	// 16-point
	std::cout << "\t CPU,n=16p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 16, triQ16, gTriCub16w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 16, triQ16, gTriCub16w ).second << std::endl;
	// 19-point
	std::cout << "\t CPU,n=19p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 19, triQ19, gTriCub19w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 19, triQ19, gTriCub19w ).second << std::endl;
	// 33-point
	std::cout << "\t CPU,n=33p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 33, triQ33, gTriCub33w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 33, triQ33, gTriCub33w ).second << std::endl;

	std::cout << "* Quadrature:" << std::endl;
	std::cout << "\t CPU:       " << intTriQuad.Potential(tL,evalPoint) << std::endl;
	std::cout << "\t Field+Pot: " << intTriQuad.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* RWG:" << std::endl;
	std::cout << "\t CPU:       " << intTriRwg.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriRwg.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Numerical (cubature+RWG with set distance ratios):" << std::endl;
	std::cout << "\t CPU:       " << intTriCub.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

	std::cout << std::endl;

	std::cout << "RECTANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	std::cout << "\t CPU:       " << intReAna.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intReAna.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Cubature:" << std::endl;

	// computing Q-points for rectangle rL
	double rectQ4[12];
	intReCub.GaussPoints_Rect4P(rLdata,rectQ4);
	double rectQ7[21];
	intReCub.GaussPoints_Rect7P(rLdata,rectQ7);
	double rectQ9[27];
	intReCub.GaussPoints_Rect9P(rLdata,rectQ9);
	double rectQ12[36];
	intReCub.GaussPoints_Rect12P(rLdata,rectQ12);
	double rectQ17[51];
	intReCub.GaussPoints_Rect17P(rLdata,rectQ17);
	double rectQ20[60];
	intReCub.GaussPoints_Rect20P(rLdata,rectQ20);
	double rectQ33[99];
	intReCub.GaussPoints_Rect33P(rLdata,rectQ33);

	// 4-point
	std::cout << "\t CPU, n=4p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 4, rectQ4, gRectCub4w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 4, rectQ4, gRectCub4w ).second << std::endl;
	// 7-point
	std::cout << "\t CPU, n=7p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 7, rectQ7, gRectCub7w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 7, rectQ7, gRectCub7w ).second << std::endl;
	// 9-point
	std::cout << "\t CPU, n=9p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 9, rectQ9, gRectCub9w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 9, rectQ9, gRectCub9w ).second << std::endl;
	// 12-point
	std::cout << "\t CPU,n=12p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 12, rectQ12, gRectCub12w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 12, rectQ12, gRectCub12w ).second << std::endl;
	// 17-point
	std::cout << "\t CPU,n=17p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 17, rectQ17, gRectCub17w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 17, rectQ17, gRectCub17w ).second << std::endl;
	// 20-point
	std::cout << "\t CPU,n=20p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 20, rectQ20, gRectCub20w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 20, rectQ20, gRectCub20w ).second << std::endl;
	// 33-point
	std::cout << "\t CPU,n=33p: " << intReCub.Potential_RectNP( rLdata, evalPoint, 33, rectQ33, gRectCub33w ) << std::endl;
	std::cout << "\t Field+Pot: " << intReCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 33, rectQ33, gRectCub33w ).second << std::endl;

	std::cout << "* Quadrature:" << std::endl;
	std::cout << "\t CPU:       " << intReQuad.Potential(rL->GetShape(), evalPoint) << std::endl;
	std::cout << "\t Field+Pot: " << intReQuad.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* RWG:" << std::endl;
	std::cout << "\t CPU:       " << intReRwg.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intReRwg.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;

	std::cout << std::endl;

	std::cout << "LINE SEGMENT" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	std::cout << "\t CPU:       " << intLineAna.Potential( wL->GetShape(), evalPoint ) << std::endl << std::endl;
	std::cout << "\t Field+Pot: " << intLineAna.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Quadrature:" << std::endl;
	std::cout << "\t CPU, n=2p  " << intLineNum.Potential_nNodes( wLdata, evalPoint, 1, gQuadx2, gQuadw2 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 1, gQuadx2, gQuadw2 ).second << std::endl;

	std::cout << "\t CPU, n=3p  " << intLineNum.Potential_nNodes( wLdata, evalPoint, 2, gQuadx3, gQuadw3 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 2, gQuadx3, gQuadw3 ).second << std::endl;

	std::cout << "\t CPU, n=4p  " << intLineNum.Potential_nNodes( wLdata, evalPoint, 2, gQuadx4, gQuadw4 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 2, gQuadx4, gQuadw4 ).second << std::endl;

	std::cout << "\t CPU, n=6p  " << intLineNum.Potential_nNodes( wLdata, evalPoint, 3, gQuadx6, gQuadw6 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 3, gQuadx6, gQuadw6 ).second << std::endl;

	std::cout << "\t CPU, n=8p  " << intLineNum.Potential_nNodes( wLdata, evalPoint, 4, gQuadx8, gQuadw8 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 4, gQuadx8, gQuadw8 ).second << std::endl;

	std::cout << "\t CPU, n=16p " << intLineNum.Potential_nNodes(wLdata,evalPoint,8, gQuadx16,gQuadw16) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 8, gQuadx16, gQuadw16 ).second << std::endl;

	std::cout << "\t CPU, n=32p " << intLineNum.Potential_nNodes(wLdata, evalPoint, 16, gQuadx32, gQuadw32) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 16, gQuadx32, gQuadw32 ).second << std::endl;

	std::cout << "\n";

	std::cout << "\t CPU, 256p: " << intLineQuad.Potential( wL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineQuad.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).second << std::endl;

	std::cout << std::endl;

	std::cout << "\n---------------" << std::endl;
	std::cout << "Electric Fields" << std::endl;
	std::cout << "---------------" << std::endl;

	std::cout << "TRIANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	printVec("\t CPU:       ", intTriAna.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intTriAna.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);

	std::cout << "* Cubature:" << std::endl;
	// 4-point
	printVec("\t CPU, n=4p: ", intTriCub.ElectricField_TriNP(tLdata,evalPoint,4,triQ4,gTriCub4w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,4,triQ4,gTriCub4w).first);
	// 7-point
	printVec("\t CPU, n=7p: ", intTriCub.ElectricField_TriNP(tLdata,evalPoint,7,triQ7,gTriCub7w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,7,triQ7,gTriCub7w).first);
	// 12-point
	printVec("\t CPU, n=12p:", intTriCub.ElectricField_TriNP(tLdata,evalPoint,12,triQ12,gTriCub12w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,12,triQ12,gTriCub12w).first);
	// 16-point
	printVec("\t CPU, n=16p:", intTriCub.ElectricField_TriNP(tLdata,evalPoint,16,triQ16,gTriCub16w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,16,triQ16,gTriCub16w).first);
	// 19-point
	printVec("\t CPU, n=19p:", intTriCub.ElectricField_TriNP(tLdata,evalPoint,19,triQ19,gTriCub19w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,19,triQ19,gTriCub19w).first);
	// 33-point
	printVec("\t CPU, n=33p:", intTriCub.ElectricField_TriNP(tLdata,evalPoint,33,triQ33,gTriCub33w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,33,triQ33,gTriCub33w).first);

	std::cout << "* Quadrature:" << std::endl;
	printVec("\t CPU:       ", intTriQuad.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intTriQuad.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);

	std::cout << "* RWG:" << std::endl;
	printVec("\t CPU:       ", intTriRwg.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intTriRwg.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);

	std::cout << "* Numerical (cubature+RWG with set distance ratios):" << std::endl;
	printVec("\t CPU:       ", intTriCub.ElectricField( tL->GetShape(), evalPoint) );
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);

	std::cout << std::endl;

	std::cout << "RECTANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	printVec("\t CPU:       ", intReAna.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intReAna.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);

	std::cout << "* Cubature:" << std::endl;
	// 4-point
	printVec("\t CPU, n=4p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,4,rectQ4,gRectCub4w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,4,rectQ4,gRectCub4w).first);
	// 7-point
	printVec("\t CPU, n=7p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,7,rectQ7,gRectCub7w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,7,rectQ7,gRectCub7w).first);
	// 9-point
	printVec("\t CPU, n=9p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,9,rectQ9,gRectCub9w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,9,rectQ9,gRectCub9w).first);
	// 12-point
	printVec("\t CPU,n=12p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,12,rectQ12,gRectCub12w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,12,rectQ12,gRectCub12w).first);
	// 17-point
	printVec("\t CPU,n=17p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,17,rectQ17,gRectCub17w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,17,rectQ17,gRectCub17w).first);
	// 20-point
	printVec("\t CPU,n=20p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,20,rectQ20,gRectCub20w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,20,rectQ20,gRectCub20w).first);
	// 33-point
	printVec("\t CPU,n=33p: ", intReCub.ElectricField_RectNP(rLdata,evalPoint,33,rectQ33,gRectCub33w));
	printVec("\t Field+Pot: ", intReCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,33,rectQ33,gRectCub33w).first);

	std::cout << "* Quadrature:" << std::endl;
	printVec("\t CPU:       ", intReQuad.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intReQuad.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);

	std::cout << "* RWG:" << std::endl;
	printVec("\t CPU:       ", intReRwg.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intReRwg.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);

	std::cout << std::endl;

	std::cout << "LINE SEGMENT" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	printVec("\t CPU:      ", intLineAna.ElectricField(wL->GetShape(),evalPoint));
	printVec("\t Field+Pot:", intLineAna.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).first);
	std::cout << std::endl;

	std::cout << "* Quadrature:" << std::endl;
	printVec("\t CPU,n=2p: ", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 1, gQuadx2, gQuadw2 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 1, gQuadx2, gQuadw2 ).first);

	printVec("\t CPU,n=3p: ", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 2, gQuadx3, gQuadw3 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 2, gQuadx3, gQuadw3 ).first);

	printVec("\t CPU,n=4p: ", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 2, gQuadx4, gQuadw4 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 2, gQuadx4, gQuadw4 ).first);

	printVec("\t CPU,n=6p: ", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 3, gQuadx6, gQuadw6 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 3, gQuadx6, gQuadw6 ).first);

	printVec("\t CPU,n=8p: ", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 4, gQuadx8, gQuadw8 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 4, gQuadx8, gQuadw8 ).first);

	printVec("\t CPU,n=16p:", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 8, gQuadx16, gQuadw16 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 8, gQuadx16, gQuadw16 ).first);

	printVec("\t CPU,n=32p:", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 16, gQuadx32, gQuadw32 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 16, gQuadx32, gQuadw32 ).first);

	std::cout << "\n";

	printVec("\t CPU, 256p:", intLineQuad.ElectricField( wL->GetShape(), evalPoint ));
	printVec("\t Field+Pot:", intLineQuad.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).first);

	return 0;
}
