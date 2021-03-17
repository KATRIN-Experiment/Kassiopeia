#include "KEMConstants.hh"
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KElectrostaticCubatureRectangleIntegrator.hh"
#include "KElectrostaticCubatureTriangleIntegrator.hh"
#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KThreeVector_KEMField.hh"

#include <cstdlib>
#include <ctime>
#include <getopt.h>
#include <iostream>


using namespace KEMField;

void printVec(const std::string& add, KFieldVector input)
{
    std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

clock_t start;

void StartTimer()
{
    start = clock();
}

double Time()
{
    double end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
}


int main()
{
    // Functionality test for boundary integrator classes


    // Triangles
    // ---------

    auto* tL = new KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle>();
    tL->SetA(1.);                   // positive x-direction
    tL->SetB(2.);                   // positive y-direction
    KFieldVector tLp0(3., 5., -6.); /* P0 */
    tL->SetP0(tLp0);
    KFieldVector tLn1(1. / sqrt(2.), 1. / sqrt(2.), 0.); /* N1 */
    tL->SetN1(tLn1);
    KFieldVector tLn2(0., 1. / sqrt(2.), 1. / sqrt(2.)); /* N2 */
    tL->SetN2(tLn2);
    //tL->SetSolution(1.); // charge density (electrostatic basis)
    tL->SetBoundaryValue(100.);  // electric potential

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
                               tL->GetN2().Z()};

    auto* tR = new KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle>();
    tR->SetA(1.133);               // positive x-direction
    tR->SetB(2.2323);              // positive y-direction
    KFieldVector tRp0(0., 0., 1.); /* P0 */
    tR->SetP0(tRp0);
    KFieldVector tRn1(1., 0., 0.); /* N1 */
    tR->SetN1(tRn1);
    KFieldVector tRn2(0., 1., 0.); /* N2 */
    tR->SetN2(tRn2);
    //tR->SetSolution(12.); // charge density (electrostatic basis)
    tR->SetBoundaryValue(-100.);  // electric potential


    // Rectangles
    // ----------

    auto* rL = new KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>();
    rL->SetA(3.);                    // positive x-direction
    rL->SetB(2.);                    // positive y-direction
    KFieldVector rLp0(0., 0., -0.9); /* P0 */
    rL->SetP0(rLp0);
    KFieldVector rLn1(1., 0., 0.); /* N1 */
    rL->SetN1(rLn1);
    KFieldVector rLn2(0., 1., 0.); /* N2 */
    rL->SetN2(rLn2);
    //rL->SetSolution(12.); // charge density (electrostatic basis)
    rL->SetBoundaryValue(-200.);  // electric potential

    auto* rR = new KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>();
    rR->SetA(1.);                   // positive x-direction
    rR->SetB(2.);                   // positive y-direction
    KFieldVector rRp0(0., 0., 0.9); /* P0 */
    rR->SetP0(rRp0);
    KFieldVector rRn1(1., 0., 0.); /* N1 */
    rR->SetN1(rRn1);
    KFieldVector rRn2(0., 1., 0.); /* N2 */
    rR->SetN2(rRn2);
    //rR->SetSolution(12.); // charge density (electrostatic basis)
    rR->SetBoundaryValue(200.);  // electric potential


    // Line Segments
    // -------------

    auto* wL = new KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment>();
    wL->SetP0(KFieldVector(0.1, -1.5, -0.5));
    wL->SetP1(KFieldVector(0.1, 1., -0.5));
    wL->SetDiameter(0.003);
    wL->SetBoundaryValue(-1000);

    auto* wR = new KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment>();
    wR->SetP0(KFieldVector(0.1, -1., 0.5));
    wR->SetP1(KFieldVector(0.1, 1., 0.5));
    wR->SetDiameter(0.003);
    wR->SetBoundaryValue(-1000);


    // Surface container
    // -----------------

    KSurfaceContainer surfaceContainer;
    surfaceContainer.push_back(tL);
    //	surfaceContainer.push_back( tR );
    surfaceContainer.push_back(rL);
    //	surfaceContainer.push_back( rR );
    surfaceContainer.push_back(wL);
    //	surfaceContainer.push_back( wR );
    std::cout << "BoundaryContainer - size: " << surfaceContainer.size() << std::endl;


    // Boundary integrators and visitors
    KElectrostaticAnalyticTriangleIntegrator intTriAna;
    KElectrostaticRWGTriangleIntegrator intTriRwg;
    KElectrostaticCubatureTriangleIntegrator intTriCub;

    KElectrostaticAnalyticRectangleIntegrator intReAna;
    KElectrostaticRWGRectangleIntegrator intReRwg;
    KElectrostaticCubatureRectangleIntegrator intReCub;

    KElectrostaticAnalyticLineSegmentIntegrator intLine;
    KElectrostaticQuadratureLineSegmentIntegrator intQuadLine;


    // left triangle
    KPosition evalPoint(10.1, 0.12, 5.);

    std::cout << "TRIANGLE - Potentials:" << std::endl;

    KPosition testL(0.5, 0.5, 8.);
    KPosition test1(4.5, 2., 8.);
    KPosition test2(4.5, 2., 80.);
    KPosition test3(4.5, 2., 500.);

    long N = 1e7;
    std::cout << "Computing " << N << " points." << std::endl;

    double pot1(0.), pot2(0.);

    (void) pot1;
    (void) pot2;

    /// arrays

    std::cout << "Triangle Rwg" << std::endl;

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        pot2 = intTriRwg.Potential(tL->GetShape(), test1);
    }
    std::cout << "* Time: " << Time() << std::endl;

    ///

    std::cout << "Triangle Cub7" << std::endl;

    double q7[21];

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        intTriCub.GaussPoints_Tri7P(tLdata, q7);
        pot2 = intTriCub.Potential_TriNP(tLdata, test1, 7, q7, gTriCub7w);
    }

    std::cout << "* Time: " << Time() << std::endl;

    ///

    std::cout << "Triangle Cub12" << std::endl;

    double q12[36];

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        intTriCub.GaussPoints_Tri12P(tLdata, q12);
        pot2 = intTriCub.Potential_TriNP(tLdata, test1, 12, q12, gTriCub12w);
    }

    std::cout << "* Time: " << Time() << std::endl;


    ///

    std::cout << "Triangle Analytic" << std::endl;

    StartTimer();
    for (unsigned int i = 0; i < N; i++) {
        test1.Z() = i;
        pot1 = intTriAna.Potential(tL->GetShape(), test1);
    }
    std::cout << "* Time: " << Time() << std::endl;

    ///

    std::cout << "Triangle Rwg" << std::endl;

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        pot2 = intTriRwg.Potential(tL->GetShape(), test1);
    }
    std::cout << "* Time: " << Time() << std::endl;

    ///

    std::cout << "Triangle Cub7" << std::endl;

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        intTriCub.GaussPoints_Tri7P(tLdata, q7);
        pot2 = intTriCub.Potential_TriNP(tLdata, test1, 7, q7, gTriCub7w);
    }

    std::cout << "* Time: " << Time() << std::endl;

    ///

    std::cout << "Triangle Cub12" << std::endl;

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        intTriCub.GaussPoints_Tri12P(tLdata, q12);
        pot2 = intTriCub.Potential_TriNP(tLdata, test1, 12, q12, gTriCub12w);
    }

    std::cout << "* Time: " << Time() << std::endl;

    //

    double q16[48];

    std::cout << "Triangle Cub16" << std::endl;

    StartTimer();
    for (int i = 0; i < N; i++) {
        test1.Z() = i;
        intTriCub.GaussPoints_Tri16P(tLdata, q16);
        pot2 = intTriCub.Potential_TriNP(tLdata, test1, 16, q16, gTriCub16w);
    }

    std::cout << "* Time: " << Time() << std::endl;


    return 0;
}
