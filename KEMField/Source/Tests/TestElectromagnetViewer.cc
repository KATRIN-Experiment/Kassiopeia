#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include "KEMThreeVector.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KLineCurrent.hh"
#include "KCurrentLoop.hh"
#include "KSolenoid.hh"
#include "KCoil.hh"

#include "KElectromagnetContainer.hh"

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

   KCoil* coil = new KCoil();
   coil->SetValues(rMin,rMax,zMin,zMax,current,nDisc);

   KSolenoid* solenoid = new KSolenoid();
   solenoid->SetValues(rMin,zMin,zMax,current);

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
