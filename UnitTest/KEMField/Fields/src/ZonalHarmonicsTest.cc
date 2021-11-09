/*
 * ZonalHarmonicsTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestZonalHarmonics.cc
 */

#include "KElectromagnetIntegratingFieldSolver.hh"
#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KElectrostaticZonalHarmonicFieldSolver.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KZonalHarmonicContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

#include "KEMConstants.hh"
#include "KEMTicker.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

class KEMFieldZonalHarmonicsTest : public KEMFieldTest
{
protected:
    void SetUp() override
    {
        KEMFieldTest::SetUp();

        using namespace KGeoBag;

        // first, the magnets

        KPosition origin(0., 0., 0.);
        KDirection x(1., 0., 0.);
        KDirection y(0., 1. / sqrt(2.), 1. / sqrt(2.));
        KDirection z(0., 1. / sqrt(2.), -1. / sqrt(2.));

        KPosition newOrigin = origin + 1. * z;

        unsigned int nDisc = 100;

        double rMin = 1.;
        double rMax = 2.;
        double zMin = 0.;
        double zMax = 1.;
        double current = 1.;

        KCurrentLoop* currentLoop = new KCurrentLoop();
        currentLoop->SetValues(rMin, zMin, current);
        currentLoop->GetCoordinateSystem().SetValues(origin, y, z, x);

        KSolenoid* solenoid = new KSolenoid();
        solenoid->SetValues(rMin, zMin, zMax, current);
        solenoid->GetCoordinateSystem().SetValues(origin, x, y, z);

        KCoil* coil = new KCoil();
        coil->SetValues(rMin, rMax, zMin, zMax, current, nDisc);
        coil->GetCoordinateSystem().SetValues(newOrigin, x, y, z);

        electromagnetContainer.push_back(currentLoop);
        electromagnetContainer.push_back(solenoid);
        electromagnetContainer.push_back(coil);

        // then, the electrodes

        KPosition p0(7., 0., 0.);
        KPosition p1(6., 0., 1.);
        double charge = 1.;

        typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KRing> Ring;

        Ring* ring = new Ring();
        ring->SetValues(p0);
        ring->SetSolution(charge);

        typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KConicSection> ConicSection;

        ConicSection* conicSection = new ConicSection();
        conicSection->SetValues(p0, p1);
        conicSection->SetSolution(1.e-8);

        electrostaticContainer.push_back(ring);
        electrostaticContainer.push_back(conicSection);
    }

    KElectromagnetContainer electromagnetContainer;
    KSurfaceContainer electrostaticContainer;

};

TEST_F(KEMFieldZonalHarmonicsTest, Electromagnet)
{
    // make some direct solvers

    KElectromagnetIntegrator electromagnetIntegrator;
    KIntegratingFieldSolver<KElectromagnetIntegrator> integratingBFieldSolver(electromagnetContainer, electromagnetIntegrator);

    // then, the zonal harmonic stuff

    KZonalHarmonicContainer<KMagnetostaticBasis> electromagnetZHContainer(electromagnetContainer);
    electromagnetZHContainer.GetParameters().SetNBifurcations(3);
    electromagnetZHContainer.ComputeCoefficients();

    KZonalHarmonicFieldSolver<KMagnetostaticBasis> zonalHarmonicBFieldSolver(electromagnetZHContainer, electromagnetIntegrator);

    zonalHarmonicBFieldSolver.Initialize();

    unsigned int nSamples = 1.e3;
    srand(1337);

    KFieldVector deltaA, deltaB;
    KGradient deltaBp;

    for (unsigned int i = 0; i < nSamples; i++) {
        KPosition P;
        KFieldVector A[2], B[2];
        KGradient Bp[2];

        const double range = 8;
        P[0] = -range * .5 + range * ((double) rand()) / RAND_MAX;
        P[1] = -range * .5 + range * ((double) rand()) / RAND_MAX;
        P[2] = -range * .5 + range * ((double) rand()) / RAND_MAX;

        A[0] = integratingBFieldSolver.VectorPotential(P);
        A[1] = zonalHarmonicBFieldSolver.VectorPotential(P);

        B[0] = integratingBFieldSolver.MagneticField(P);
        B[1] = zonalHarmonicBFieldSolver.MagneticField(P);

        Bp[0] = integratingBFieldSolver.MagneticFieldGradient(P);
        Bp[1] = zonalHarmonicBFieldSolver.MagneticFieldGradient(P);

        for (unsigned int j = 0; j < 3; j++) {
            deltaA[j] += (A[1][j] - A[0][j]) / (fabs(A[0][j]) > 1.e-14 ? A[0][j] : 1.);
            deltaB[j] += (B[1][j] - B[0][j]) / (fabs(B[0][j]) > 1.e-14 ? B[0][j] : 1.);
            for (unsigned int k = 0; k < 3; k++)
                deltaBp(j, k) += (Bp[1](j, k) - Bp[0](j, k)) / (fabs(Bp[0](j, k)) > 1.e-14 ? Bp[0](j, k) : 1.);
        }
    }

    for (unsigned int j = 0; j < 3; j++) {
        ASSERT_NEAR(deltaA[j], 0, 1e-6);
        ASSERT_NEAR(deltaB[j], 0, 1e-6);
        for (unsigned int k = 0; k < 3; k++)
            ASSERT_NEAR(deltaBp(j, k), 0, 1e-4);
    }

}

TEST_F(KEMFieldZonalHarmonicsTest, ElectrostaticBoundary)
{
    // make some direct solvers

    KElectrostaticBoundaryIntegrator electrostaticIntegrator{KEBIFactory::MakeDefault()};
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator> integratingEFieldSolver(electrostaticContainer, electrostaticIntegrator);

    // then, the zonal harmonic stuff

    KZonalHarmonicContainer<KElectrostaticBasis> electrostaticZHContainer(electrostaticContainer);
    electrostaticZHContainer.GetParameters().SetNCentralCoefficients(500);
    electrostaticZHContainer.GetParameters().SetNRemoteCoefficients(500);
    electrostaticZHContainer.ComputeCoefficients();

    KZonalHarmonicFieldSolver<KElectrostaticBasis> zonalHarmonicEFieldSolver(electrostaticZHContainer, electrostaticIntegrator);

    zonalHarmonicEFieldSolver.Initialize();

    unsigned int nSamples = 1.e3;
    srand(1337);

    double deltaPhi = 0;
    KFieldVector deltaE;

    for (unsigned int i = 0; i < nSamples; i++) {
        KPosition P;
        double phi[2];
        KFieldVector E[2];

        const double range = 8;
        P[0] = -range * .5 + range * ((double) rand()) / RAND_MAX;
        P[1] = -range * .5 + range * ((double) rand()) / RAND_MAX;
        P[2] = -range * .5 + range * ((double) rand()) / RAND_MAX;

        phi[0] = integratingEFieldSolver.Potential(P);
        phi[1] = zonalHarmonicEFieldSolver.Potential(P);

        E[0] = integratingEFieldSolver.ElectricField(P);
        E[1] = zonalHarmonicEFieldSolver.ElectricField(P);

        deltaPhi += (phi[1] - phi[0]) / (fabs(phi[0]) > 1.e-14 ? phi[0] : 1.);
        for (unsigned int j = 0; j < 3; j++) {
            deltaE[j] += (E[1][j] - E[0][j]) / (fabs(E[0][j]) > 1.e-14 ? E[0][j] : 1.);
        }
    }

    ASSERT_NEAR(deltaPhi, 0, 1e-8);
    for (unsigned int j = 0; j < 3; j++) {
        ASSERT_NEAR(deltaE[j], 0, 1e-8);
    }
}
