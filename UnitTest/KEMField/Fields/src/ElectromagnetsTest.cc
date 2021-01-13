/*
 * ElectromagnetsTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestElectromagnets.cc
 */

#include "KCoil.hh"
#include "KCoilIntegrator.hh"
#include "KCurrentLoop.hh"
#include "KCurrentLoopIntegrator.hh"
#include "KLineCurrent.hh"
#include "KLineCurrentIntegrator.hh"
#include "KSolenoid.hh"
#include "KSolenoidIntegrator.hh"
#include "KElectromagnetContainer.hh"
#include "KElectromagnetIntegratingFieldSolver.hh"
#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KZonalHarmonicTypes.hh"

#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KEMFieldTest.hh"

using namespace KEMField;

TEST_F(KEMFieldTest, Electromagnets_LineCurrent)
{
    // testNumber == 0
    KLineCurrentIntegrator lineCurrentIntegrator;

    KPosition p0(0., 0., -1000.);
    KPosition p1(0., 0., 1000.);
    double current = 1.;
    double dist = 3.;

    KLineCurrent lineCurrent;
    lineCurrent.SetValues(p0, p1, current);

    double B_analytic = KEMConstants::Mu0OverPi * current * .5 / dist;

    ASSERT_NEAR(B_analytic, lineCurrentIntegrator.MagneticField(lineCurrent, KPosition(dist, 0., 0.)).Magnitude(), 1e-12);
}

TEST_F(KEMFieldTest, Electromagnets_CurrentLoop)
{
    // testNumber == 1
    KCurrentLoopIntegrator currentLoopIntegrator;

    double r = 1.;
    double current = 1.;
    double offset = 4.;

    KCurrentLoop currentLoop;
    currentLoop.SetValues(r, 0., current);

    double B_analytic = KEMConstants::Mu0 * current * .5 * (r * r / pow(offset * offset + r * r, 1.5));

    ASSERT_NEAR(B_analytic, currentLoopIntegrator.MagneticField(currentLoop, KPosition(0., 0., offset)).Magnitude(), 1e-12);
}

TEST_F(KEMFieldTest, Electromagnets_LineCurrent_CurrentLoop)
{
    // testNumber == 2
    KLineCurrentIntegrator lineCurrentIntegrator;
    KCurrentLoopIntegrator currentLoopIntegrator;

    unsigned int nDisc = 100;

    double r = 1.;
    double z = 0.;
    double current = 1.;

    KCurrentLoop* currentLoop = new KCurrentLoop();
    currentLoop->SetValues(r, z, current);

    std::vector<KLineCurrent*> lineCurrents;

    KPosition p0, p1;

    for (unsigned int i = 0; i < nDisc; i++) {
        double theta0 = (double(i)) / (nDisc) * (2. * KEMConstants::Pi);
        double theta1 = (i + 1.) / (nDisc) * (2. * KEMConstants::Pi);

        p0[0] = r * cos(theta0);
        p0[1] = r * sin(theta0);
        p0[2] = z;

        p1[0] = r * cos(theta1);
        p1[1] = r * sin(theta1);
        p1[2] = z;

        lineCurrents.push_back(new KLineCurrent());
        lineCurrents.back()->SetValues(p0, p1, current);
    }

    KPosition P(3, 0, 0);
    P[0] = P[1] = P[2] = 0.;

    P[0] = 3.;
    P[2] = 0.;

    KFieldVector B[2], A[2];

    B[0] = currentLoopIntegrator.MagneticField(*currentLoop, P);
    A[0] = currentLoopIntegrator.VectorPotential(*currentLoop, P);

    B[1] = A[1] = KFieldVector::sZero;

    for (unsigned int i = 0; i < nDisc; i++) {
        B[1] += lineCurrentIntegrator.MagneticField(*lineCurrents.at(i), P);
        A[1] += lineCurrentIntegrator.VectorPotential(*lineCurrents.at(i), P);
    }

    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(B[0][i], B[1][i], 1e-10);
        ASSERT_NEAR(A[0][i], A[1][i], 1e-10);
    }

    delete currentLoop;
    for (unsigned int i = 0; i < nDisc; i++)
        delete lineCurrents.at(i);
}

TEST_F(KEMFieldTest, Electromagnets_Solenoid_CurrentLoop)
{
    // testNumber == 3
    KSolenoidIntegrator solenoidIntegrator;
    KCurrentLoopIntegrator currentLoopIntegrator;

    unsigned int nDisc = 500;

    double r = 1.;
    double zMin = 0.;
    double zMax = 1.;
    double current = 1.;

    KSolenoid* solenoid = new KSolenoid();
    solenoid->SetValues(r, zMin, zMax, current);

    std::vector<KCurrentLoop*> currentLoops;

    for (unsigned int i = 0; i < nDisc; i++) {
        double zLoop = zMin + i / (nDisc - 1.) * (zMax - zMin);
        currentLoops.push_back(new KCurrentLoop());
        currentLoops.back()->SetValues(r, zLoop, current / nDisc);
    }

    KPosition P;
    P[0] = P[1] = P[2] = 0.;

    P[0] = 3.;
    P[2] = 0.;

    KFieldVector B[2], A[2];

    B[0] = solenoidIntegrator.MagneticField(*solenoid, P);
    A[0] = solenoidIntegrator.VectorPotential(*solenoid, P);

    B[1] = A[1] = KFieldVector::sZero;

    for (unsigned int i = 0; i < nDisc; i++) {
        B[1] += currentLoopIntegrator.MagneticField(*currentLoops.at(i), P);
        A[1] += currentLoopIntegrator.VectorPotential(*currentLoops.at(i), P);
    }

    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(B[0][i], B[1][i], 1e-10);
        ASSERT_NEAR(A[0][i], A[1][i], 1e-10);
    }

    delete solenoid;
    for (unsigned int i = 0; i < nDisc; i++)
        delete currentLoops.at(i);
}


TEST_F(KEMFieldTest, Electromagnets_Solenoid_Coil)
{
    // testNumber == 4
    KSolenoidIntegrator solenoidIntegrator;
    KCoilIntegrator coilIntegrator;

    unsigned int nDisc = 500;

    double rMin = 1.;
    double rMax = 2.;
    double zMin = 0.;
    double zMax = 1.;
    double current = 1.;

    KCoil* coil = new KCoil();
    coil->SetValues(rMin, rMax, zMin, zMax, current, nDisc);

    std::vector<KSolenoid*> solenoids;

    for (unsigned int i = 0; i < nDisc; i++) {
        double rSolenoid = rMin + i / (nDisc - 1.) * (rMax - rMin);
        solenoids.push_back(new KSolenoid());
        solenoids.back()->SetValues(rSolenoid, zMin, zMax, current / nDisc);
    }

    KPosition P;
    P[0] = P[1] = P[2] = 0.;

    P[0] = 3.;
    P[2] = 0.;

    KFieldVector B[2], A[2];

    B[0] = coilIntegrator.MagneticField(*coil, P);
    A[0] = coilIntegrator.VectorPotential(*coil, P);

    B[1] = A[1] = KFieldVector::sZero;

    for (unsigned int i = 0; i < nDisc; i++) {
        B[1] += solenoidIntegrator.MagneticField(*solenoids.at(i), P);
        A[1] += solenoidIntegrator.VectorPotential(*solenoids.at(i), P);
    }

    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(B[0][i], B[1][i], 1e-10);
        ASSERT_NEAR(A[0][i], A[1][i], 1e-10);
    }

    delete coil;
    for (unsigned int i = 0; i < nDisc; i++)
        delete solenoids.at(i);
}
