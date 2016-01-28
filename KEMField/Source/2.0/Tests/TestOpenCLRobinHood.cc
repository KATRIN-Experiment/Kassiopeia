#include <getopt.h>
#include <iostream>
#include <cstdlib>

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"


#include "KElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood.hh"
#include "KRobinHood_OpenCL.hh"
#include "KIterativeStateWriter.hh"
#include "KIterationTracker.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"

using namespace KEMField;


int main(int argc, char* argv[])
{

  KSurfaceContainer surfaceContainer;

  KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
  tL->SetA( 5. ); // positive x-direction
  tL->SetB( 2.5 ); // positive y-direction
  KEMThreeVector tLp0( 0., 0., 8. ); /* P0 */
  tL->SetP0(tLp0);
  KEMThreeVector tLn1( 1., 0., 0. ); /* N1 */
  tL->SetN1( tLn1 );
  KEMThreeVector tLn2( 0., 1., 0. ); /* N2 */
  tL->SetN2( tLn2 );
  //tL->SetSolution(1.); // charge density (electrostatic basis)
  tL->SetBoundaryValue( -10. ); // electric potential

  KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
  tR->SetA( 5. ); // positive x-direction
  tR->SetB( 2.5 ); // positive y-direction
  KEMThreeVector tRp0( 0., 0., 12. ); /* P0 */
  tR->SetP0(tRp0);
  KEMThreeVector tRn1( 1., 0., 0. ); /* N1 */
  tR->SetN1( tRn1 );
  KEMThreeVector tRn2( 0., 1., 0. ); /* N2 */
  tR->SetN2( tRn2 );
  //tR->SetSolution(1.); // charge density (electrostatic basis)
  tR->SetBoundaryValue( 10. ); // electric potential

  KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* t3 = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
  t3->SetA( 5. ); // positive x-direction
  t3->SetB( 2.5 ); // positive y-direction
  KEMThreeVector t3p0( 0., 0., 14. ); /* P0 */
  t3->SetP0(t3p0);
  KEMThreeVector t3n1( 1., 0., 0. ); /* N1 */
  t3->SetN1( t3n1 );
  KEMThreeVector t3n2( 0., 1., 0. ); /* N2 */
  t3->SetN2( t3n2 );
  t3->SetSolution(1.); // charge density (electrostatic basis)
  t3->SetBoundaryValue( -10. ); // electric potential

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KLineSegment>* w = new KSurface<KElectrostaticBasis,
					   KDirichletBoundary,
					   KLineSegment>();

  w->SetP0(KEMThreeVector(-0.457222,0.0504778,-0.51175));
  w->SetP1(KEMThreeVector(-0.463342,0.0511534,-0.515712));
  w->SetDiameter(0.0003);
  w->SetBoundaryValue(-900);

  // left triangle
KPosition t(0.5,0.5,9.);

surfaceContainer.push_back( tL );
surfaceContainer.push_back( tR );
surfaceContainer.push_back( t3 );

// check components of Robin Hood
// check pointers to data arrays
// check streamer


    KOpenCLSurfaceContainer* cudaSurfaceContainer = new KOpenCLSurfaceContainer(surfaceContainer);
    KOpenCLInterface::GetInstance()->SetActiveData( cudaSurfaceContainer );
    
    KOpenCLElectrostaticBoundaryIntegrator integrator( *cudaSurfaceContainer );
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > A(*cudaSurfaceContainer,integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > b(*cudaSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > x(*cudaSurfaceContainer,integrator);
    
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
    
    robinHood.SetTolerance( 1e-6 );
    robinHood.SetResidualCheckInterval( 1 );

    KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >* display = new KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >();
    display->Interval( 1 );
   	robinHood.AddVisitor( display );
                        
    robinHood.Solve(A,x,b);

    //KElectrostaticBoundaryIntegrator* fIntegrator;
    //KIntegratingFieldSolver< KElectrostaticBoundaryIntegrator >* fIntegratingFieldSolver;

    //KOpenCLElectrostaticBoundaryIntegrator* integrator;

    KOpenCLData* data = KOpenCLInterface::GetInstance()->GetActiveData();
//    if( data )
//        cudaSurfaceContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
//    else
//    {
        cudaSurfaceContainer = new KOpenCLSurfaceContainer( surfaceContainer );
        KOpenCLInterface::GetInstance()->SetActiveData( cudaSurfaceContainer );
//    }
    //KOpenCLElectrostaticBoundaryIntegrator* fOCLIntegrator = new KOpenCLElectrostaticBoundaryIntegrator( *oclContainer );
    KIntegratingFieldSolver< KOpenCLElectrostaticBoundaryIntegrator >* fOCLIntegratingFieldSolver
	= new KIntegratingFieldSolver< KOpenCLElectrostaticBoundaryIntegrator >( *cudaSurfaceContainer, integrator );
    fOCLIntegratingFieldSolver->Initialize();


    KPosition P(0,0,16);

    std::cout << fOCLIntegratingFieldSolver->Potential( P ) << std::endl;


  return 0;
}
