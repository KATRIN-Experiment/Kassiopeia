#include "KCoil.hh"
#include "KCurrentLoop.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KElectromagnetContainer.hh"
#include "KLineCurrent.hh"
#include "KSolenoid.hh"
#include "KThreeVector_KEMField.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKElectromagnetViewer.hh"
#endif

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
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

    KElectromagnetContainer container;

    container.push_back(coil);
    container.push_back(solenoid);

#ifdef KEMFIELD_USE_VTK
    KEMVTKElectromagnetViewer viewer(container);

    viewer.GenerateGeometryFile();
    viewer.ViewGeometry();
#endif

    return 0;
}
