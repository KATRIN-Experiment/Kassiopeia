#include <getopt.h>
#include <iostream>
#include <cstdlib>

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"


#ifdef KEMFIELD_USE_CUDA
#include "KCUDASurfaceContainer.hh"
#endif


using namespace KEMField;

void DiscretizeInterval( double interval, int nSegments, double power, std::vector< double >& segments );

int main(int argc, char* argv[])
{


int scale=10;
double power=2.;

  KSurfaceContainer surfaceContainer;

  std::vector<double> segments(2*scale);
  DiscretizeInterval(2*1.,2*scale,power,segments);

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
  tL->SetBoundaryValue( 1000. ); // electric potential

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
surfaceContainer.push_back( w );



    KCUDASurfaceContainer cudaSurfaceContainer(surfaceContainer);


    std::cout << cudaSurfaceContainer.size() << std::endl;
cudaSurfaceContainer.SetMinimumWorkgroupSizeForKernels(256);
    cudaSurfaceContainer.ConstructCUDAObjects();


    std::cout << cudaSurfaceContainer.at(0)->GetName() << std::endl;
    std::cout << cudaSurfaceContainer.at(1)->GetName() << std::endl;

  return 0;
}

void DiscretizeInterval( double interval, int nSegments, double power, std::vector< double >& segments )
{
  if( nSegments == 1 )
    segments[ 0 ] = interval;
  else
  {
    double inc1, inc2;
    double mid = interval * .5;
    if( nSegments % 2 == 1 )
    {
      segments[ nSegments / 2 ] = interval / nSegments;
      mid -= interval / (2 * nSegments);
    }

    for( int i = 0; i < nSegments / 2; i++ )
    {
      inc1 = ((double) i) / (nSegments / 2);
      inc2 = ((double) (i + 1)) / (nSegments / 2);

      inc1 = pow( inc1, power );
      inc2 = pow( inc2, power );

      segments[ i ] = segments[ nSegments - (i + 1) ] = mid * (inc2 - inc1);
    }
  }
  return;
}
