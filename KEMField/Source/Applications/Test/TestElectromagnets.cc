#include "KCoil.hh"
#include "KCoilIntegrator.hh"
#include "KCurrentLoop.hh"
#include "KCurrentLoopIntegrator.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KElectromagnetContainer.hh"
#include "KElectromagnetIntegratingFieldSolver.hh"
#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KLineCurrent.hh"
#include "KLineCurrentIntegrator.hh"
#include "KSolenoid.hh"
#include "KSolenoidIntegrator.hh"
#include "KThreeVector_KEMField.hh"
#include "KZonalHarmonicTypes.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    int testNumber = 8;

    KLineCurrentIntegrator lineCurrentIntegrator;
    KCurrentLoopIntegrator currentLoopIntegrator;
    KSolenoidIntegrator solenoidIntegrator;
    KCoilIntegrator coilIntegrator;

    if (testNumber == 0) {
        KPosition p0(0., 0., -1000.);
        KPosition p1(0., 0., 1000.);
        double current = 1.;
        double dist = 3.;

        KLineCurrent lineCurrent;
        lineCurrent.SetValues(p0, p1, current);

        KEMField::cout << "TEST: " << lineCurrentIntegrator.MagneticField(lineCurrent, KPosition(dist, 0., 0.))
                       << KEMField::endl;
        KEMField::cout << "should be " << KEMConstants::Mu0OverPi * current * .5 / dist << KEMField::endl;
    }

    if (testNumber == 1) {
        double r = 1.;
        double current = 1.;
        double offset = 4.;

        KCurrentLoop currentLoop;
        currentLoop.SetValues(r, 0., current);

        KEMField::cout << "TEST: " << currentLoopIntegrator.MagneticField(currentLoop, KPosition(0., 0., offset))
                       << KEMField::endl;
        KEMField::cout << "should be " << KEMConstants::Mu0 * current * .5 * (r * r / pow(offset * offset + r * r, 1.5))
                       << KEMField::endl;
    }

    if (testNumber == 2) {
        unsigned int nDisc = 100;

        double r = 1.;
        double z = 0.;
        double current = 1.;

        auto* currentLoop = new KCurrentLoop();
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

        KPosition P;
        P[0] = P[1] = P[2] = 0.;

        P[0] = 3.;
        P[2] = 0.;

        KEMField::cout << "CURRENT LOOP:  magnetic field at " << P << " is "
                       << currentLoopIntegrator.MagneticField(*currentLoop, P) << KEMField::endl;
        KEMField::cout << "CURRENT LOOP:  vector potential at " << P << " is "
                       << currentLoopIntegrator.VectorPotential(*currentLoop, P) << KEMField::endl;

        KFieldVector B(0., 0., 0.);
        KFieldVector A(0., 0., 0.);

        for (unsigned int i = 0; i < nDisc; i++) {
            B += lineCurrentIntegrator.MagneticField(*lineCurrents.at(i), P);
            A += lineCurrentIntegrator.VectorPotential(*lineCurrents.at(i), P);
        }

        KEMField::cout << "LINE CURRENTS: magnetic field at " << P << " is " << B << KEMField::endl;
        KEMField::cout << "LINE CURRENTS: vector potential at " << P << " is " << A << KEMField::endl;

        delete currentLoop;
        for (unsigned int i = 0; i < nDisc; i++)
            delete lineCurrents.at(i);
    }

    if (testNumber == 3) {
        unsigned int nDisc = 500;

        double r = 1.;
        double zMin = 0.;
        double zMax = 1.;
        double current = 1.;

        auto* solenoid = new KSolenoid();
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

        KEMField::cout << "SOLENOID:  magnetic field at " << P << " is "
                       << solenoidIntegrator.MagneticField(*solenoid, P) << KEMField::endl;
        KEMField::cout << "SOLENOID:  vector potential at " << P << " is "
                       << solenoidIntegrator.VectorPotential(*solenoid, P) << KEMField::endl;

        KFieldVector B(0., 0., 0.);
        KFieldVector A(0., 0., 0.);

        for (unsigned int i = 0; i < nDisc; i++) {
            B += currentLoopIntegrator.MagneticField(*currentLoops.at(i), P);
            A += currentLoopIntegrator.VectorPotential(*currentLoops.at(i), P);
        }

        KEMField::cout << "CURRENT LOOPS: magnetic field at " << P << " is " << B << KEMField::endl;
        KEMField::cout << "CURRENT LOOPS: vector potential at " << P << " is " << A << KEMField::endl;

        delete solenoid;
        for (unsigned int i = 0; i < nDisc; i++)
            delete currentLoops.at(i);
    }

    if (testNumber == 4) {
        unsigned int nDisc = 500;

        double rMin = 1.;
        double rMax = 2.;
        double zMin = 0.;
        double zMax = 1.;
        double current = 1.;

        auto* coil = new KCoil();
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

        KEMField::cout << "COIL:  magnetic field at " << P << " is " << coilIntegrator.MagneticField(*coil, P)
                       << KEMField::endl;
        KEMField::cout << "COIL:  vector potential at " << P << " is " << coilIntegrator.VectorPotential(*coil, P)
                       << KEMField::endl;

        KFieldVector B(0., 0., 0.);
        KFieldVector A(0., 0., 0.);

        for (unsigned int i = 0; i < nDisc; i++) {
            B += solenoidIntegrator.MagneticField(*solenoids.at(i), P);
            A += solenoidIntegrator.VectorPotential(*solenoids.at(i), P);
        }

        KEMField::cout << "SOLENOIDS: magnetic field at " << P << " is " << B << KEMField::endl;
        KEMField::cout << "SOLENOIDS: vector potential at " << P << " is " << A << KEMField::endl;

        delete coil;
        for (unsigned int i = 0; i < nDisc; i++)
            delete solenoids.at(i);
    }

    if (testNumber == 5) {
        KElectromagnetContainer container;

        unsigned int nDisc = 500;

        double rMin = 1.;
        double rMax = 2.;
        double zMin = 0.;
        double zMax = 1.;
        double current = 1.;

        auto* coil = new KCoil();
        coil->SetValues(rMin, rMax, zMin, zMax, current, nDisc);

        auto* solenoid = new KSolenoid();
        solenoid->SetValues(rMin, zMin, zMax, current);

        KEMField::cout << "container size (empty): " << container.size() << KEMField::endl;
        container.push_back(coil);
        KEMField::cout << "container size (one element): " << container.size() << KEMField::endl;
        container.push_back(solenoid);
        KEMField::cout << "container size (two elements): " << container.size() << KEMField::endl;

        KEMField::cout << container << KEMField::endl;

        auto* anotherContainer = new KElectromagnetContainer(container);
        anotherContainer->IsOwner(false);
        KEMField::cout << *anotherContainer << KEMField::endl;
        delete anotherContainer;
        KEMField::cout << container << KEMField::endl;
    }

    if (testNumber == 6) {
        KElectromagnetContainer container;

        unsigned int nDisc = 500;

        double rMin = 1.;
        double rMax = 2.;
        double zMin = 0.;
        double zMax = 1.;
        double current = 1.;

        auto* coil = new KCoil();
        coil->SetValues(rMin, rMax, zMin, zMax, current, nDisc);

        auto* solenoid = new KSolenoid();
        solenoid->SetValues(rMin, zMin, zMax, current);

        container.push_back(coil);
        container.push_back(solenoid);

        KElectromagnetIntegrator integrator;
        KIntegratingFieldSolver<KElectromagnetIntegrator> fieldSolver(container, integrator);

        KPosition P;
        P[0] = P[1] = P[2] = 0.;

        P[0] = 3.;
        P[2] = 0.;

        KEMField::cout << "Magnetic field: " << fieldSolver.MagneticField(P) << KEMField::endl;
        KEMField::cout << "Vector potential: " << fieldSolver.VectorPotential(P) << KEMField::endl;
    }

    if (testNumber == 7) {
        KElectromagnetContainer lineCurrents;
        KElectromagnetContainer currentLoops;

        unsigned int nDisc = 300;

        double r = 1.;
        double z = 0.;
        double current = 1.;

        auto* currentLoop = new KCurrentLoop();
        currentLoop->SetValues(r, z, current);

        currentLoops.push_back(currentLoop);

        KLineCurrent* lineCurrent;

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

            lineCurrent = new KLineCurrent();
            lineCurrent->SetValues(p0, p1, current);
            lineCurrents.push_back(lineCurrent);
        }

        KElectromagnetIntegrator integrator;

        KIntegratingFieldSolver<KElectromagnetIntegrator> lineCurrentFieldSolver(lineCurrents, integrator);
        KIntegratingFieldSolver<KElectromagnetIntegrator> currentLoopFieldSolver(currentLoops, integrator);

        KPosition P;
        P[0] = P[1] = P[2] = 0.;

        P[0] = 3.;
        P[2] = 0.;

        KEMField::cout << "CURRENT LOOP:  magnetic field at " << P << " is " << currentLoopFieldSolver.MagneticField(P)
                       << KEMField::endl;
        KEMField::cout << "CURRENT LOOP:  vector potential at " << P << " is "
                       << currentLoopFieldSolver.VectorPotential(P) << KEMField::endl;

        KEMField::cout << "LINE CURRENTS:  magnetic field at " << P << " is " << lineCurrentFieldSolver.MagneticField(P)
                       << KEMField::endl;
        KEMField::cout << "LINE CURRENTS:  vector potential at " << P << " is "
                       << lineCurrentFieldSolver.VectorPotential(P) << KEMField::endl;
    }

    if (testNumber == 8) {
        unsigned int nDisc = 500;

        double rMin = 1.;
        double rMax = 2.;
        double zMin = 0.;
        double zMax = 1.;
        double current = 1.;

        auto* coil = new KCoil();
        coil->SetValues(rMin, rMax, zMin, zMax, current, nDisc);

        coil->GetCoordinateSystem().SetValues(KPosition(0., 0., 0.),
                                              KDirection(1., 0., 0.),
                                              KDirection(0., 1., 0.),
                                              KDirection(0., 0., 1.));

        auto* coil2 = new KCoil();
        coil2->SetValues(rMin, rMax, zMin - 1, zMax, current, nDisc);

        coil2->GetCoordinateSystem().SetValues(KPosition(0., 0., -4.),
                                               KDirection(1., 0., 0.),
                                               KDirection(0., 1., 0.),
                                               KDirection(0., 0., 1.));

        auto* coil3 = new KCoil();
        coil3->SetValues(rMin, rMax, zMin, zMax, current, nDisc);

        coil3->GetCoordinateSystem().SetValues(KPosition(1., 2., 3.),
                                               KDirection(0., 1., 0.),
                                               KDirection(1. / sqrt(2.), 0., 1. / sqrt(2.)),
                                               KDirection(1. / sqrt(2.), 0., -1. / sqrt(2.)));

        KElectromagnetContainer container;

        container.push_back(coil);
        container.push_back(coil2);
        container.push_back(coil3);

        KElectromagnetIntegrator integrator;

        KIntegratingFieldSolver<KElectromagnetIntegrator> integratingFieldSolver(container, integrator);

        KZonalHarmonicContainer<KMagnetostaticBasis> zonalHarmonicContainer(container);
        zonalHarmonicContainer.GetParameters().SetNBifurcations(3);
        zonalHarmonicContainer.ComputeCoefficients();

        KZonalHarmonicFieldSolver<KMagnetostaticBasis> zonalHarmonicFieldSolver(zonalHarmonicContainer, integrator);
        zonalHarmonicFieldSolver.Initialize();

        KPosition P;
        P[0] = P[1] = P[2] = 0.;

        // P[0] = 15.;
        P[0] = 6.;
        P[1] = 1.;
        P[2] = 3.;

        KEMField::cout << "INTEGRATING:    magnetic field at " << P << " is " << integratingFieldSolver.MagneticField(P)
                       << KEMField::endl;
        KEMField::cout << "INTEGRATING:    vector potential at " << P << " is "
                       << integratingFieldSolver.VectorPotential(P) << KEMField::endl;
        KEMField::cout << "INTEGRATING:    magnetic field gradient at " << P << " is "
                       << integratingFieldSolver.MagneticFieldGradient(P) << KEMField::endl;

        KEMField::cout << "ZONAL HARMONIC: magnetic field at " << P << " is "
                       << zonalHarmonicFieldSolver.MagneticField(P) << KEMField::endl;
        KEMField::cout << "ZONAL HARMONIC: vector potential at " << P << " is "
                       << zonalHarmonicFieldSolver.VectorPotential(P) << KEMField::endl;
        KEMField::cout << "ZONAL HARMONIC: magnetic field gradient at " << P << " is "
                       << zonalHarmonicFieldSolver.MagneticFieldGradient(P) << KEMField::endl;
    }

    return 0;
}
