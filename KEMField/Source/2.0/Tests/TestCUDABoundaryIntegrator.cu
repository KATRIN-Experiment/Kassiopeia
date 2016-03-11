#include <getopt.h>
#include <iostream>
#include <cstdlib>

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"


#include "KCUDASurfaceContainer.hh"
#include "KCUDAElectrostaticBoundaryIntegrator.hh"


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
  tL->SetSolution(12.); // charge density (electrostatic basis)
  tL->SetBoundaryValue( -9. ); // electric potential

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
//surfaceContainer.push_back( w );



    KCUDASurfaceContainer cudaSurfaceContainer(surfaceContainer);


    //std::cout << cudaSurfaceContainer.size() << std::endl;
	cudaSurfaceContainer.SetMinimumWorkgroupSizeForKernels(256);
    cudaSurfaceContainer.ConstructCUDAObjects();
    KCUDAElectrostaticBoundaryIntegrator integrator( cudaSurfaceContainer );
    std::cout << integrator.Potential(tL,t) << std::endl;
    std::cout << cudaSurfaceContainer.at(0)->GetName() << std::endl;

  return 0;
}
