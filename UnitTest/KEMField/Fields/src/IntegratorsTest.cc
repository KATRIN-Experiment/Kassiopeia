/*
 * IntegratorsTest.cc
 *
 *  Created on: 27 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestIntegratorRWG.cc
 */

#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KThreeVector_KEMField.hh"

#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

#define ASSERT_NEAR_RELATIVE(val1, val2, rel) \
    ASSERT_NEAR(val1, val2, (rel) * 0.5 * fabs((val1) + (val2)))

#include "KEMFieldTest.hh"

TEST_F(KEMFieldTest, RectangleIntegrator)
{
    // Rectangles
    // ----------

    auto* rL =
        new KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>();
    rL->SetA(4.);                    // positive x-direction
    rL->SetB(4.);                    // positive y-direction
    KFieldVector rLp0(-2., -2., 0.); /* P0 */
    rL->SetP0(rLp0);
    KFieldVector rLn1(1., 0., 0.); /* N1 */
    rL->SetN1(rLn1);
    KFieldVector rLn2(0., 1., 0.); /* N2 */
    rL->SetN2(rLn2);
    //rL->SetSolution(12.); // charge density (electrostatic basis)
    rL->SetBoundaryValue(10.);  // electric potential

    // RECTANGLE

    KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>* testRect = rL;
    std::array<KFieldVector,7> reEvalPoint;

    // points for rL

    reEvalPoint[0] = testRect->GetP0();  // point in left lower corner (P0)
    reEvalPoint[1] = KFieldVector(0.1, -1.5, 0.);  // arbitrary point on surface
    reEvalPoint[2] = testRect->Centroid();  // point in center of rectangle
    reEvalPoint[3] = KFieldVector(10., 10., 0.);  // point in plane of rectangle but far outside
    reEvalPoint[4] = KFieldVector(0., -2., 0.);  // point on side line of rectangle
    reEvalPoint[5] = KFieldVector(10., -20.5, 33.58);  // arbitrary point
    reEvalPoint[6] = KFieldVector(-1, -1., 0.);  // point on line subdividing the rectangle into two triangles

    // Boundary integrators and visitors
    KElectrostaticBiQuadratureRectangleIntegrator intReQuad;
    KElectrostaticAnalyticRectangleIntegrator intReAna;
    KElectrostaticRWGRectangleIntegrator intReRwg;

    std::cout << "Testing RWG rectangles ..." << std::endl;
    for (unsigned int i = 0; i < reEvalPoint.size(); i++) {
        double phi[3];
        KFieldVector E[3];

        phi[0] = intReQuad.Potential(testRect, reEvalPoint[i]);
        phi[1] = intReAna.Potential(testRect, reEvalPoint[i]);
        phi[2] = intReRwg.Potential(testRect, reEvalPoint[i]);

        E[0] = intReQuad.ElectricField(testRect, reEvalPoint[i]);
        E[1] = intReAna.ElectricField(testRect, reEvalPoint[i]);
        E[2] = intReRwg.ElectricField(testRect, reEvalPoint[i]);

        // this case is problematic with quadrature integrator
        if (i != 6) {
            ASSERT_NEAR_RELATIVE(phi[0], phi[1], 0.025);
        }
        ASSERT_NEAR_RELATIVE(phi[1], phi[2], 0.025);

        for (int k = 0; k < 3; k++) {
            // some cases are problematic with quadrature or analytic integrator
            if ((i == 3) || (i == 5)) {
                ASSERT_NEAR_RELATIVE(E[0][k], E[1][k], 0.025);
            }
            if ((i == 1) || (i == 2) || (i == 3) || (i == 5) || (i == 6)) {
                ASSERT_NEAR_RELATIVE(E[1][k], E[2][k], 0.025);
            }
        }
    }
}

TEST_F(KEMFieldTest, TriangleIntegrator)
{
    // Triangle
    // --------

    auto* tri1 =
        new KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle>();

    KFieldVector tri1P0(-1.75, 0.25, 0.);
    tri1->SetP0(tri1P0);

    KFieldVector tri1P1(1.75, 0.25, 0.);
    KFieldVector tri1N1 = (tri1P1 - tri1P0).Unit();
    tri1->SetA((tri1P1 - tri1P0).Magnitude());
    tri1->SetN1(tri1N1);

    KFieldVector tri1P2(0., 2.25, 0.);
    KFieldVector tri1N2 = (tri1P2 - tri1P0).Unit();
    tri1->SetB((tri1P2 - tri1P0).Magnitude());
    tri1->SetN2(tri1N2);

    tri1->SetBoundaryValue(10.);  // electric potential

    // TRIANGLE

    KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle>* testTri = tri1;
    std::array<KFieldVector,6> triEvalPoint;

    triEvalPoint[0] = testTri->GetP0();  // point in left lower corner (P0)
    triEvalPoint[1] = KFieldVector(0.01, 0.5, 0.);  // arbitrary point on surface
    triEvalPoint[2] = testTri->Centroid();  // point in center of triangle
    triEvalPoint[3] = KFieldVector(10., 1., 0.);  // point in plane of triangle but far outside
    triEvalPoint[4] = KFieldVector(0., 0.25, 0.);  // point on side line of triangle
    triEvalPoint[5] = KFieldVector(10., -20.5, 33.58);  // arbitrary point

    // Boundary integrators and visitors
    KElectrostaticBiQuadratureTriangleIntegrator intTriQuad;
    KElectrostaticAnalyticTriangleIntegrator intTriAna;
    KElectrostaticRWGTriangleIntegrator intTriRwg;

    std::cout << "Testing RWG triangles ..." << std::endl;
    for (unsigned int i = 0; i < triEvalPoint.size(); i++) {
        double phi[3];
        KFieldVector E[3];

        phi[0] = intTriQuad.Potential(testTri, triEvalPoint[i]);
        phi[1] = intTriAna.Potential(testTri, triEvalPoint[i]);
        phi[2] = intTriRwg.Potential(testTri, triEvalPoint[i]);

        E[0] = intTriQuad.ElectricField(testTri, triEvalPoint[i]);
        E[1] = intTriAna.ElectricField(testTri, triEvalPoint[i]);
        E[2] = intTriRwg.ElectricField(testTri, triEvalPoint[i]);

        ASSERT_NEAR_RELATIVE(phi[0], phi[1], 0.025);
        ASSERT_NEAR_RELATIVE(phi[1], phi[2], 0.025);

        for (int k = 0; k < 3; k++) {
            // some cases are problematic with quadrature or analytic integrator
            if ((i == 3) || (i == 5)) {
                ASSERT_NEAR_RELATIVE(E[0][k], E[1][k], 0.025);
            }
            if ((i == 3) || (i == 5)) {
                ASSERT_NEAR_RELATIVE(E[1][k], E[2][k], 0.025);
            }
        }
    }
}
