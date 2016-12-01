#include <getopt.h>
#include <iostream>
#include <cstdlib>

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"

#include "KElectrostaticNumericBoundaryIntegrator.hh"

#include "KCUDASurfaceContainer.hh"
#include "KCUDAElectrostaticNumericBoundaryIntegrator.hh"

#include "KCUDABoundaryIntegralMatrix.hh"
#include "KCUDABoundaryIntegralVector.hh"
#include "KCUDABoundaryIntegralSolutionVector.hh"
#include "KRobinHood_CUDA.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KCUDAElectrostaticIntegratingFieldSolver.hh"


using namespace KEMField;

int main(int /*argc*/, char** /*argv[]*/)
{
    KPosition ori(0.,0.,14.001);

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
    tL->SetBoundaryValue( 1000. ); // electric potential

    KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
    tR->SetA( 5. ); // positive x-direction
    tR->SetB( 2.5 ); // positive y-direction
    KEMThreeVector tRp0( 0., 0., 12. ); /* P0 */
    tR->SetP0(tRp0);
    KEMThreeVector tRn1( 1., 0., 0. ); /* N1 */
    tR->SetN1( tRn1 );
    KEMThreeVector tRn2( 0., 1., 0. ); /* N2 */
    tR->SetN2( tRn2 );
    //tL->SetSolution(1.); // charge density (electrostatic basis)
    tR->SetBoundaryValue( 1000. ); // electric potential

    KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* t3 = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
    t3->SetA( 5. ); // positive x-direction
    t3->SetB( 2.5 ); // positive y-direction
    KEMThreeVector t3p0( 0., 0., 14. ); /* P0 */
    t3->SetP0(t3p0);
    KEMThreeVector t3n1( 1., 0., 0. ); /* N1 */
    t3->SetN1( t3n1 );
    KEMThreeVector t3n2( 0., 1., 0. ); /* N2 */
    t3->SetN2( t3n2 );
    //t3->SetSolution(1.); // charge density (electrostatic basis)
    t3->SetBoundaryValue( 1000. ); // electric potential

    KSurface<KElectrostaticBasis,
    KDirichletBoundary,
    KLineSegment>* w = new KSurface<KElectrostaticBasis,
    KDirichletBoundary,
    KLineSegment>();

    w->SetP0(KEMThreeVector(-0.457222,0.0504778,-0.51175));
    w->SetP1(KEMThreeVector(-0.463342,0.0511534,-0.515712));
    w->SetDiameter(0.0003);
    w->SetBoundaryValue(-900);


    surfaceContainer.push_back( tL );
    surfaceContainer.push_back( tR );
    surfaceContainer.push_back( t3 );

    KCUDASurfaceContainer* cudaSurfaceContainer = new KCUDASurfaceContainer(surfaceContainer);
    KCUDAInterface::GetInstance()->SetActiveData( cudaSurfaceContainer );


    KCUDAElectrostaticNumericBoundaryIntegrator integrator( *cudaSurfaceContainer );

    // ---------------------------------------------------------------------------------

    // CUDA memcopy needed here in parallel to memcopy in CUDA integrator class in order
    // to guarantee that data will be copied onto GPU constant memory

    // 7-point triangle cubature

    cudaMemcpyToSymbol( cuTriCub7alpha, gTriCub7alpha, sizeof(CU_TYPE)*3 );
    cudaMemcpyToSymbol( cuTriCub7beta, gTriCub7beta, sizeof(CU_TYPE)*3 );
    cudaMemcpyToSymbol( cuTriCub7gamma, gTriCub7gamma, sizeof(CU_TYPE)*3 );
    cudaMemcpyToSymbol( cuTriCub7w, gTriCub7w, sizeof(CU_TYPE)*7 );

    // 12-point triangle cubature

    cudaMemcpyToSymbol( cuTriCub12alpha, gTriCub12alpha, sizeof(CU_TYPE)*4 );
    cudaMemcpyToSymbol( cuTriCub12beta, gTriCub12beta, sizeof(CU_TYPE)*4 );
    cudaMemcpyToSymbol( cuTriCub12gamma, gTriCub12gamma, sizeof(CU_TYPE)*4 );
    cudaMemcpyToSymbol( cuTriCub12w, gTriCub12w, sizeof(CU_TYPE)*12 );

    // 33-point triangle cubature

    cudaMemcpyToSymbol( cuTriCub33alpha, gTriCub33alpha, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuTriCub33beta, gTriCub33beta, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuTriCub33gamma, gTriCub33gamma, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuTriCub33w, gTriCub33w, sizeof(CU_TYPE)*33 );

    // rectangle cubature weights

    cudaMemcpyToSymbol( cuRectCub7w, gRectCub7w, sizeof(CU_TYPE)*7 );
    cudaMemcpyToSymbol( cuRectCub12w, gRectCub12w, sizeof(CU_TYPE)*12 );
    cudaMemcpyToSymbol( cuRectCub33w, gRectCub33w, sizeof(CU_TYPE)*33 );

    // quadrature weights and nodes for line segments

    cudaMemcpyToSymbol( cuLineQuadx4, gQuadx4, sizeof(CU_TYPE)*2 );
    cudaMemcpyToSymbol( cuLineQuadw4, gQuadw4, sizeof(CU_TYPE)*2 );
    cudaMemcpyToSymbol( cuLineQuadx16, gQuadx16, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuLineQuadw16, gQuadw16, sizeof(CU_TYPE)*8 );

    // ---------------------------------------------------------------------------------

    KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<KElectrostaticBasis> > A(*cudaSurfaceContainer,integrator);
    KBoundaryIntegralVector<KCUDABoundaryIntegrator<KElectrostaticBasis> > b(*cudaSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<KElectrostaticBasis> > x(*cudaSurfaceContainer,integrator);

    KRobinHood<KElectrostaticNumericBoundaryIntegrator::ValueType, KRobinHood_CUDA> robinHood;

    robinHood.SetTolerance( 1e-8 );
    robinHood.SetResidualCheckInterval( 1 );

    KIterationDisplay< KElectrostaticNumericBoundaryIntegrator::ValueType >* display = new KIterationDisplay< KElectrostaticNumericBoundaryIntegrator::ValueType >();
    display->Interval( 1 );
    robinHood.AddVisitor( display );

    robinHood.Solve(A,x,b);

    KElectrostaticNumericBoundaryIntegrator integr;
    KIntegratingFieldSolver<KElectrostaticNumericBoundaryIntegrator> solver(surfaceContainer, integr );
    std::cout << "Buffered elements on device: " << cudaSurfaceContainer->GetNBufferedElements() << std::endl;
    std::cout << "Potential at origin (CPU): " << solver.Potential( ori ) << std::endl;

    KCUDAData* data = KCUDAInterface::GetInstance()->GetActiveData();
    //            if( data )
    //                oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
    //            else
    //            {
    cudaSurfaceContainer = new KCUDASurfaceContainer( surfaceContainer );
    KCUDAInterface::GetInstance()->SetActiveData( cudaSurfaceContainer );
    //            }

    KIntegratingFieldSolver< KCUDAElectrostaticNumericBoundaryIntegrator >* fCUDAIntegratingFieldSolver
    = new KIntegratingFieldSolver< KCUDAElectrostaticNumericBoundaryIntegrator >( *cudaSurfaceContainer, integrator );

    fCUDAIntegratingFieldSolver->Initialize();
    //fCUDAIntegratingFieldSolver->ConstructCUDAKernels();
    //fCUDAIntegratingFieldSolver->AssignDeviceMemory();
    std::cout << "Potential at origin (GPU): " << fCUDAIntegratingFieldSolver->Potential( ori ) << std::endl;

    //        }



    return 0;
}
