#include <getopt.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticCubatureTriangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"

#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticCubatureRectangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"

#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"

#include "KEMThreeVector.hh"

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"

#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"

#define PRINTCST
//#define TRI
#define RECT
//#define LINE

using namespace KEMField;

void printVec( std::string add, KEMThreeVector input )
{
	std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

int main()
{
	// Functionality test for OpenCL boundary integrator classes and computation of numeric constants

	std::cout << std::fixed << std::setprecision(16);

#ifdef PRINTCST
	std::cout << "Constants for triangle 7-point cubature:" << std::endl;
	std::cout << "gTriCub7alpha = { ";
	for( unsigned short i=0; i<3; i++ ) std::cout << gTriCub7alpha[i] << " ; ";
	std::cout << " }" << std::endl;
	std::cout << "gTriCub7beta = { ";
	for( unsigned short i=0; i<3; i++ ) std::cout << gTriCub7beta[i] << " ; ";
	std::cout << " }" << std::endl;
	std::cout << "gTriCub7gamma = { ";
	for( unsigned short i=0; i<3; i++ ) std::cout << gTriCub7gamma[i] << " ; ";
	std::cout << " }" << std::endl;
	std::cout << "gTriCub7w = { ";
	for( unsigned short i=0; i<7; i++ ) std::cout << gTriCub7w[i] << "\n\t";
	std::cout << " }\n" << std::endl;

    std::cout << "Constants for triangle 12-point cubature:" << std::endl;
    std::cout << "gTriCub12alpha = { ";
    for( unsigned short i=0; i<4; i++ ) std::cout << gTriCub12alpha[i] << " ; ";
    std::cout << " }" << std::endl;
    std::cout << "gTriCub12beta = { ";
    for( unsigned short i=0; i<4; i++ ) std::cout << gTriCub12beta[i] << " ; ";
    std::cout << " }" << std::endl;
    std::cout << "gTriCub12gamma = { ";
    for( unsigned short i=0; i<4; i++ ) std::cout << gTriCub12gamma[i] << " ; ";
    std::cout << " }" << std::endl;
    std::cout << "gTriCub12w = { ";
    for( unsigned short i=0; i<12; i++ ) std::cout << gTriCub12w[i] << "\n\t";
    std::cout << " }\n" << std::endl;

	std::cout << "Constants for triangle 33-point cubature:" << std::endl;
	std::cout << "gTriCub33alpha = { ";
	for( unsigned short i=0; i<8; i++ ) std::cout << gTriCub33alpha[i] << " ; ";
	std::cout << " }" << std::endl;
	std::cout << "gTriCub33beta = { ";
	for( unsigned short i=0; i<8; i++ ) std::cout << gTriCub33beta[i] << " ; ";
	std::cout << " }" << std::endl;
	std::cout << "gTriCub33gamma = { ";
	for( unsigned short i=0; i<8; i++ ) std::cout << gTriCub33gamma[i] << " ; ";
	std::cout << " }" << std::endl;
	std::cout << "gTriCub33w = { ";
	for( unsigned short i=0; i<33; i++ ) std::cout << gTriCub33w[i] << "\n\t";
	std::cout << " }\n" << std::endl;

	std::cout << "Constants for rectangle 7-point cubature:" << std::endl;
	std::cout << "gRectCub7w = { ";
	for( unsigned short i=0; i<7; i++ ) std::cout << gRectCub7w[i] << "\n\t";
	std::cout << " }\n" << std::endl;

    std::cout << "Constants for rectangle 12-point cubature:" << std::endl;
    std::cout << "gRectCub12w = { ";
    for( unsigned short i=0; i<12; i++ ) std::cout << gRectCub12w[i] << "\n\t";
    std::cout << " }\n" << std::endl;

	std::cout << "Constants for rectangle 33-point cubature:" << std::endl;
	std::cout << "gRectCub33w = { ";
	for( unsigned short i=0; i<33; i++ ) std::cout << gRectCub33w[i] << "\n\t";
	std::cout << " }\n" << std::endl;
#endif

#ifdef TRI
	// Triangles
	// ---------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
	tL->SetA( 1. ); // positive x-direction
	tL->SetB( 2. ); // positive y-direction
	KEMThreeVector tLp0( 3., 5., -6. ); /* P0 */
	tL->SetP0(tLp0);
	KEMThreeVector tLn1( 1./sqrt(2.), 1./sqrt(2.), 0. ); /* N1 */
	tL->SetN1( tLn1 );
	KEMThreeVector tLn2( 0., 1./sqrt(2.), 1./sqrt(2.) ); /* N2 */
	tL->SetN2( tLn2 );
	//tL->SetSolution(1.); // charge density (electrostatic basis)
	tL->SetBoundaryValue( 100. ); // electric potential

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
#endif

#ifdef RECT
	// Rectangles
	// ----------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>* rL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>();
	rL->SetA( 3. ); // positive x-direction
	rL->SetB( 2. ); // positive y-direction
	KEMThreeVector rLp0( 0., 0., -0.9 ); /* P0 */
	rL->SetP0(rLp0);
	KEMThreeVector rLn1( 1., 0., 0. ); /* N1 */
	rL->SetN1( rLn1 );
	KEMThreeVector rLn2( 0., 1., 0. ); /* N2 */
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
#endif

#ifdef LINE
	// Line Segments
	// -------------

	KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment>* wL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment>();
	wL->SetP0(KEMThreeVector(0.1,-1.5,-0.5));
	wL->SetP1(KEMThreeVector(0.1,1.,-0.5));
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
#endif

	// Surface container
	// -----------------

	KSurfaceContainer* surfaceContainer = new KSurfaceContainer();

#ifdef TRI
	surfaceContainer->push_back( tL );
#endif

#ifdef RECT
	surfaceContainer->push_back( rL );
#endif

#ifdef LINE
	surfaceContainer->push_back( wL );
#endif

    KOpenCLData* data = KOpenCLInterface::GetInstance()->GetActiveData();
    KOpenCLSurfaceContainer* oclContainer;
    if( data )
        oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
    else {
        oclContainer = new KOpenCLSurfaceContainer( *surfaceContainer );
        KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
    }

	// Boundary integrators and visitors

#ifdef TRI
	KElectrostaticAnalyticTriangleIntegrator intTriAna;
	KElectrostaticRWGTriangleIntegrator intTriRwg;
	KElectrostaticCubatureTriangleIntegrator intTriCub;
    KElectrostaticBiQuadratureTriangleIntegrator intTriQuad;

	// computing Q-points for triangle tL

	double triQ7[21];
	intTriCub.GaussPoints_Tri7P(tLdata,triQ7);
    double triQ12[36];
    intTriCub.GaussPoints_Tri12P(tLdata,triQ12);
	double triQ33[99];
	intTriCub.GaussPoints_Tri33P(tLdata,triQ33);
#endif

#ifdef RECT
    KElectrostaticAnalyticRectangleIntegrator intRectAna;
    KElectrostaticRWGRectangleIntegrator intRectRwg;
    KElectrostaticCubatureRectangleIntegrator intRectCub;
    KElectrostaticBiQuadratureRectangleIntegrator intRectQuad;

	double rectQ7[21];
	intRectCub.GaussPoints_Rect7P(rLdata,rectQ7);
    double rectQ12[36];
    intRectCub.GaussPoints_Rect12P(rLdata,rectQ12);
	double rectQ33[99];
	intRectCub.GaussPoints_Rect33P(rLdata,rectQ33);
#endif

#ifdef LINE
    KElectrostaticAnalyticLineSegmentIntegrator intLineAna;
    KElectrostaticQuadratureLineSegmentIntegrator intLineNum;
#endif

    KOpenCLElectrostaticBoundaryIntegrator intOCLAna {
    	KoclEBIFactory::MakeAnalytic( *oclContainer )};
    KOpenCLElectrostaticBoundaryIntegrator intOCLNum {
		KoclEBIFactory::MakeNumeric( *oclContainer )};
    KOpenCLElectrostaticBoundaryIntegrator intOCLRwg {
    	KoclEBIFactory::MakeRWG( *oclContainer )};


	// left triangle
	KPosition evalPoint(10.1,0.12,5.);
    KPosition testL( 0.5, 0.5, 8. );
    KPosition test1( 4.5, 2., 8. );
    KPosition test2( 4.5, 2., 80. );
    KPosition test3( 4.5, 2., 500. );

    // ---------------------------------------------------------------------------------

    std::cout << "\n----------" << std::endl;
	std::cout << "Potentials" << std::endl;
	std::cout << "----------" << std::endl;

#ifdef TRI
	std::cout << "TRIANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	std::cout << "\t CPU:       " << intTriAna.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriAna.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLAna.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLAna.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Cubature:" << std::endl;
	std::cout << "\t CPU, n=7p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 7, triQ7, gTriCub7w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 7, triQ7, gTriCub7w ).second << std::endl;
    std::cout << "\t CPU, n=12p:" << intTriCub.Potential_TriNP( tLdata, evalPoint, 12, triQ12, gTriCub12w ) << std::endl;
    std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 12, triQ12, gTriCub12w ).second << std::endl;
	std::cout << "\t CPU,n=33p: " << intTriCub.Potential_TriNP( tLdata, evalPoint, 33, triQ33, gTriCub33w ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriCub.ElectricFieldAndPotential_TriNP( tLdata, evalPoint, 33, triQ33, gTriCub33w ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLNum.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLNum.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* RWG:" << std::endl;
	std::cout << "\t CPU:       " << intTriRwg.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intTriRwg.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLRwg.Potential( tL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLRwg.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).second << std::endl;

#endif

#ifdef RECT
	std::cout << "RECTANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	std::cout << "\t CPU:       " << intRectAna.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intRectAna.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLAna.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLAna.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Cubature:" << std::endl;
	std::cout << "\t CPU, n=7p: " << intRectCub.Potential_RectNP( rLdata, evalPoint, 7, rectQ7, gRectCub7w ) << std::endl;
	std::cout << "\t Field+Pot: " << intRectCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 7, rectQ7, gRectCub7w ).second << std::endl;
    std::cout << "\t CPU,n=12p: " << intRectCub.Potential_RectNP( rLdata, evalPoint, 12, rectQ12, gRectCub12w ) << std::endl;
    std::cout << "\t Field+Pot: " << intRectCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 12, rectQ12, gRectCub12w ).second << std::endl;
	std::cout << "\t CPU,n=33p: " << intRectCub.Potential_RectNP( rLdata, evalPoint, 33, rectQ33, gRectCub33w ) << std::endl;
	std::cout << "\t Field+Pot: " << intRectCub.ElectricFieldAndPotential_RectNP( rLdata, evalPoint, 33, rectQ33, gRectCub33w ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLNum.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLNum.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* RWG:" << std::endl;
	std::cout << "\t CPU:       " << intRectRwg.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intRectRwg.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLRwg.Potential( rL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLRwg.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).second << std::endl;
#endif

#ifdef LINE
	std::cout << "LINE SEGMENT" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	std::cout << "\t CPU:       " << intLineAna.Potential( wL->GetShape(), evalPoint ) << std::endl << std::endl;
	std::cout << "\t Field+Pot: " << intLineAna.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLAna.Potential( wL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLAna.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).second << std::endl;

	std::cout << "* Quadrature:" << std::endl;
	std::cout << "\t CPU, n=4p  " << intLineNum.Potential_nNodes( wLdata, evalPoint, 2, gQuadx4, gQuadw4 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 2, gQuadx4, gQuadw4 ).second << std::endl;
	std::cout << "\t CPU, n=16p " << intLineNum.Potential_nNodes( wLdata,evalPoint, 8, gQuadx16, gQuadw16 ) << std::endl;
	std::cout << "\t Field+Pot: " << intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 8, gQuadx16, gQuadw16 ).second << std::endl;
	std::cout << "\t GPU:       " << intOCLNum.Potential( wL->GetShape(), evalPoint ) << std::endl;
	std::cout << "\t Field+Pot: " << intOCLNum.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).second << std::endl;

#endif

    // ---------------------------------------------------------------------------------

	std::cout << "\n---------------" << std::endl;
	std::cout << "Electric Fields" << std::endl;
	std::cout << "---------------" << std::endl;

#ifdef TRI
	std::cout << "TRIANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	printVec("\t CPU:       ", intTriAna.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intTriAna.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);
	printVec("\t GPU:       ", intOCLAna.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intOCLAna.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);

	std::cout << "* Cubature:" << std::endl;
	printVec("\t CPU, n=7p: ", intTriCub.ElectricField_TriNP(tLdata,evalPoint,7,triQ7,gTriCub7w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,7,triQ7,gTriCub7w).first);
    printVec("\t CPU, n=12p:", intTriCub.ElectricField_TriNP(tLdata,evalPoint,12,triQ12,gTriCub12w));
    printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,12,triQ12,gTriCub12w).first);
	printVec("\t CPU, n=33p:", intTriCub.ElectricField_TriNP(tLdata,evalPoint,33,triQ33,gTriCub33w));
	printVec("\t Field+Pot: ", intTriCub.ElectricFieldAndPotential_TriNP(tLdata,evalPoint,33,triQ33,gTriCub33w).first);
	printVec("\t GPU:       ", intOCLNum.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intOCLNum.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);


	std::cout << "* RWG:" << std::endl;
	printVec("\t CPU:       ", intTriRwg.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intTriRwg.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);
	printVec("\t GPU:       ", intOCLRwg.ElectricField(tL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intOCLRwg.ElectricFieldAndPotential( tL->GetShape(), evalPoint ).first);
#endif

#ifdef RECT
	std::cout << "RECTANGLE" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	printVec("\t CPU:       ", intRectAna.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intRectAna.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);
	printVec("\t GPU:       ", intOCLAna.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intOCLAna.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);

	std::cout << "* Cubature:" << std::endl;
	printVec("\t CPU, n=7p: ", intRectCub.ElectricField_RectNP(rLdata,evalPoint,7,rectQ7,gRectCub7w));
	printVec("\t Field+Pot: ", intRectCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,7,rectQ7,gRectCub7w).first);
    printVec("\t CPU, n=12p:", intRectCub.ElectricField_RectNP(rLdata,evalPoint,12,rectQ12,gRectCub12w));
    printVec("\t Field+Pot: ", intRectCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,12,rectQ12,gRectCub12w).first);
	printVec("\t CPU, n=33p:", intRectCub.ElectricField_RectNP(rLdata,evalPoint,33,rectQ33,gRectCub33w));
	printVec("\t Field+Pot: ", intRectCub.ElectricFieldAndPotential_RectNP(rLdata,evalPoint,33,rectQ33,gRectCub33w).first);
	printVec("\t GPU:       ", intOCLNum.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intOCLNum.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);


	std::cout << "* RWG:" << std::endl;
	printVec("\t CPU:       ", intRectRwg.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intRectRwg.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);
	printVec("\t GPU:       ", intOCLRwg.ElectricField(rL->GetShape(),evalPoint));
	printVec("\t Field+Pot: ", intOCLRwg.ElectricFieldAndPotential( rL->GetShape(), evalPoint ).first);
#endif

#ifdef LINE
	std::cout << "LINE SEGMENT" << std::endl;

	std::cout << "* Analytical:" << std::endl;
	printVec("\t CPU:      ", intLineAna.ElectricField(wL->GetShape(),evalPoint));
	printVec("\t Field+Pot:", intLineAna.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).first);
	printVec("\t GPU:      ", intOCLAna.ElectricField(wL->GetShape(),evalPoint));
	printVec("\t Field+Pot:", intOCLAna.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).first);
	std::cout << std::endl;

	std::cout << "* Quadrature:" << std::endl;
	printVec("\t CPU,n=4p: ", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 2, gQuadx4, gQuadw4 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 2, gQuadx4, gQuadw4 ).first);
	printVec("\t CPU,n=16p:", intLineNum.ElectricField_nNodes(wLdata,evalPoint, 8, gQuadx16, gQuadw16 ));
	printVec("\t Field+Pot:", intLineNum.ElectricFieldAndPotential_nNodes( wLdata, evalPoint, 8, gQuadx16, gQuadw16 ).first);

	printVec("\t GPU:      ", intOCLNum.ElectricField(wL->GetShape(),evalPoint ));
	printVec("\t Field+Pot:", intOCLNum.ElectricFieldAndPotential( wL->GetShape(), evalPoint ).first);
#endif

	return 0;
}
