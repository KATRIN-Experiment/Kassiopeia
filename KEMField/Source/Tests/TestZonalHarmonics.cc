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
#include "KEMTicker.hh"

#include "KMD5HashGenerator.hh"

#include "KSurface.hh"

#include "KBinaryDataStreamer.hh"
#include "KSADataStreamer.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KElectromagnetIntegratingFieldSolver.hh"

#include "KElectrostaticZonalHarmonicFieldSolver.hh"
#include "KElectromagnetZonalHarmonicFieldSolver.hh"

#include "KZonalHarmonicContainer.hh"

#include "KEMThreeMatrix.hh"

using namespace KEMField;

template <class Solver>
void ComputeEField(Solver&,
		   KPosition&,
		   KEMThreeVector&);

template <class Solver>
void ComputeBField(Solver&,
		   KPosition&,
		   KEMThreeVector&);

int main(int /*argc*/, char** /*argv*/)
{
  // first, the magnets

  KPosition origin(0.,0.,0.);
  KDirection x(1.,0.,0.);
  KDirection y(0.,1./sqrt(2.),1./sqrt(2.));
  KDirection z(0.,1./sqrt(2.),-1./sqrt(2.));

  KPosition newOrigin = origin + 1.*z;

  unsigned int nDisc = 500;

  double rMin = 1.;
  double rMax = 2.;
  double zMin = 0.;
  double zMax = 1.;
  double current = 1.;

  KCurrentLoop* currentLoop = new KCurrentLoop();
  currentLoop->SetValues(rMin,zMin,current);
  currentLoop->GetCoordinateSystem().SetValues(origin,y,z,x);

  KSolenoid* solenoid = new KSolenoid();
  solenoid->SetValues(rMin,zMin,zMax,current);
  solenoid->GetCoordinateSystem().SetValues(origin,x,y,z);

  KCoil* coil = new KCoil();
  coil->SetValues(rMin,rMax,zMin,zMax,current,nDisc);
  coil->GetCoordinateSystem().SetValues(newOrigin,x,y,z);

  KElectromagnetContainer electromagnetContainer;

  electromagnetContainer.push_back(currentLoop);
  electromagnetContainer.push_back(solenoid);
  electromagnetContainer.push_back(coil);

  // then, the electrodes

  KPosition p0(7.,0.,0.);
  KPosition p1(6.,0.,1.);
  double charge = 1.;

  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KRing> Ring;

  Ring* ring = new Ring();
  ring->SetValues(p0);
  ring->SetSolution(charge);

  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KConicSection> ConicSection;

  ConicSection* conicSection = new ConicSection();
  conicSection->SetValues(p0,p1);
  conicSection->SetSolution(1.e-8);

  KSurfaceContainer electrostaticContainer;
  electrostaticContainer.push_back(ring);
  electrostaticContainer.push_back(conicSection);

  // make some direct solvers

  KElectromagnetIntegrator electromagnetIntegrator;
  KIntegratingFieldSolver<KElectromagnetIntegrator>
    integratingBFieldSolver(electromagnetContainer,electromagnetIntegrator);

  KElectrostaticBoundaryIntegrator electrostaticIntegrator {KEBIFactory::MakeDefault()};
  KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>
    integratingEFieldSolver(electrostaticContainer,electrostaticIntegrator);

  // then, the zonal harmonic stuff

  // KBinaryDataStreamer bDS;
  // KSADataStreamer bDS;

  // {
  //   KZonalHarmonicContainer<KZHElectromagnet> electromagnetZHContainer(electromagnetContainer);
  //   electromagnetZHContainer.ComputeCoefficients();

  //   bDS.open("testZHMagnets.kbd","overwrite");
  //   bDS << electromagnetZHContainer;
  //   bDS.close();
  // }

  KZonalHarmonicContainer<KMagnetostaticBasis> electromagnetZHContainer(electromagnetContainer);
  electromagnetZHContainer.GetParameters().SetNBifurcations(3);
  electromagnetZHContainer.ComputeCoefficients();

  // bDS.open("testZHMagnets.kbd","read");
  // bDS >> electromagnetZHContainer;
  // bDS.close();

  // std::remove("testZHMagnets.kbd");

  KZonalHarmonicFieldSolver<KMagnetostaticBasis> zonalHarmonicBFieldSolver(electromagnetZHContainer,electromagnetIntegrator);

  zonalHarmonicBFieldSolver.Initialize();

  // {
  //   KZonalHarmonicContainer<KZHElectrostaticSurface> electrostaticZHContainer(electrostaticContainer);
  //   electrostaticZHContainer.ComputeCoefficients();

  //   bDS.open("testZHElectrodes.kbd","overwrite");
  //   bDS << electrostaticZHContainer;
  //   bDS.close();
  // }

  KZonalHarmonicContainer<KElectrostaticBasis> electrostaticZHContainer(electrostaticContainer);
  electrostaticZHContainer.GetParameters().SetNCentralCoefficients(500);
  electrostaticZHContainer.GetParameters().SetNRemoteCoefficients(500);
  electrostaticZHContainer.ComputeCoefficients();

  // bDS.open("testZHElectrodes.kbd","read");
  // bDS >> electrostaticZHContainer;
  // bDS.close();

  // std::remove("testZHElectrodes.kbd");

  KZonalHarmonicFieldSolver<KElectrostaticBasis> zonalHarmonicEFieldSolver(electrostaticZHContainer,electrostaticIntegrator);

  zonalHarmonicEFieldSolver.Initialize();

    // std::cout<<"******************************************************************************"<<std::endl;
    // KEMField::cout<<"Performing Data Display test."<<KEMField::endl;

    // KEMField::cout<<electromagnetZHContainer<<KEMField::endl;

    // KEMField::cout<<"Data Display test completed."<<KEMField::endl;
    // std::cout<<"******************************************************************************\n"<<std::endl;

  // finally, sample some points

  KPosition P;

  double phi[2];
  KEMThreeVector A[2],B[2],E[2];
  KGradient Bp[2];

  KEMThreeVector E_Numeric;
  KEMThreeVector B_Numeric;

  double deltaPhi = 0.;
  KEMThreeVector deltaA,deltaB,deltaE;
  KGradient deltaBp;

  double deltaPhi_av = 0.;
  KEMThreeVector deltaA_av,deltaB_av,deltaE_av;
  KGradient deltaBp_av;
  double deltaPhi2_av = 0.;
  KEMThreeVector deltaA2_av,deltaB2_av,deltaE2_av;
  KGradient deltaBp2_av;
  double deltaPhi_min = 0.;
  KEMThreeVector deltaA_min,deltaB_min,deltaE_min;
  KGradient deltaBp_min;
  double deltaPhi_max = 0.;
  KEMThreeVector deltaA_max,deltaB_max,deltaE_max;
  KGradient deltaBp_max;

  unsigned int nSamples = 1.e3;

  srand((unsigned)time(0));

  KTicker ticker;

  KEMField::cout<<"Sampling "<<nSamples<<" points."<<KEMField::endl;

  ticker.StartTicker(nSamples);

  const double range = 8;

  for (unsigned int i=0;i<nSamples;i++)
  {
    P[0] = -range*.5 + range*((double)rand())/RAND_MAX;
    P[1] = -range*.5 + range*((double)rand())/RAND_MAX;
    P[2] = -range*.5 + range*((double)rand())/RAND_MAX;

    // P[0] = P[1] = 0.;
    // P[2] = 0.;

    A[0] = integratingBFieldSolver.VectorPotential(P);
    A[1] = zonalHarmonicBFieldSolver.VectorPotential(P);

    B[0] = integratingBFieldSolver.MagneticField(P);
    B[1] = zonalHarmonicBFieldSolver.MagneticField(P);

    Bp[0] = integratingBFieldSolver.MagneticFieldGradient(P);
    Bp[1] = zonalHarmonicBFieldSolver.MagneticFieldGradient(P);

    phi[0] = integratingEFieldSolver.Potential(P);
    phi[1] = zonalHarmonicEFieldSolver.Potential(P);

    E[0] = integratingEFieldSolver.ElectricField(P);
    E[1] = zonalHarmonicEFieldSolver.ElectricField(P);

    // ComputeEField(integratingEFieldSolver,P,E_Numeric);

    // std::cout<<"Phi: "<<phi[0]<<" "<<phi[1]<<std::endl;
    // std::cout<<"Ex: "<<E[0][0]<<" "<<E[1][0]<<" "<<E_Numeric[0]<<" "<<(E[0]-E_Numeric)[0]<<" "<<(E[1]-E_Numeric)[0]<<std::endl;
    // std::cout<<"Ey: "<<E[0][1]<<" "<<E[1][1]<<" "<<E_Numeric[1]<<" "<<(E[0]-E_Numeric)[1]<<" "<<(E[1]-E_Numeric)[1]<<std::endl;
    // std::cout<<"Ez: "<<E[0][2]<<" "<<E[1][2]<<" "<<E_Numeric[2]<<" "<<(E[0]-E_Numeric)[2]<<" "<<(E[1]-E_Numeric)[2]<<std::endl;

    // ComputeEField(zonalHarmonicEFieldSolver,P,E_Numeric);

    // std::cout<<"Ex: "<<E[0][0]<<" "<<E[1][0]<<" "<<E_Numeric[0]<<" "<<(E[0]-E_Numeric)[0]<<" "<<(E[1]-E_Numeric)[0]<<std::endl;
    // std::cout<<"Ey: "<<E[0][1]<<" "<<E[1][1]<<" "<<E_Numeric[1]<<" "<<(E[0]-E_Numeric)[1]<<" "<<(E[1]-E_Numeric)[1]<<std::endl;
    // std::cout<<"Ez: "<<E[0][2]<<" "<<E[1][2]<<" "<<E_Numeric[2]<<" "<<(E[0]-E_Numeric)[2]<<" "<<(E[1]-E_Numeric)[2]<<std::endl;

    // std::cout<<""<<std::endl;

    // std::cout<<"Ax: "<<A[0][0]<<" "<<A[1][0]<<" "<<(A[0][0]-A[1][0])<<" "<<(A[0][0]-A[1][0])/A[0][0]<<std::endl;

    // ComputeBField(integratingBFieldSolver,P,B_Numeric);

    // std::cout<<"Bx: "<<B[0][0]<<" "<<B[1][0]<<" "<<B_Numeric[0]<<" "<<(B[0]-B_Numeric)[0]<<" "<<(B[1]-B_Numeric)[0]<<std::endl;
    // std::cout<<"By: "<<B[0][1]<<" "<<B[1][1]<<" "<<B_Numeric[1]<<" "<<(B[0]-B_Numeric)[1]<<" "<<(B[1]-B_Numeric)[1]<<std::endl;
    // std::cout<<"Bz: "<<B[0][2]<<" "<<B[1][2]<<" "<<B_Numeric[2]<<" "<<(B[0]-B_Numeric)[2]<<" "<<(B[1]-B_Numeric)[2]<<std::endl;

    // ComputeBField(zonalHarmonicBFieldSolver,P,B_Numeric);

    // std::cout<<"Bx: "<<B[0][0]<<" "<<B[1][0]<<" "<<B_Numeric[0]<<" "<<(B[0]-B_Numeric)[0]<<" "<<(B[1]-B_Numeric)[0]<<std::endl;
    // std::cout<<"By: "<<B[0][1]<<" "<<B[1][1]<<" "<<B_Numeric[1]<<" "<<(B[0]-B_Numeric)[1]<<" "<<(B[1]-B_Numeric)[1]<<std::endl;
    // std::cout<<"Bz: "<<B[0][2]<<" "<<B[1][2]<<" "<<B_Numeric[2]<<" "<<(B[0]-B_Numeric)[2]<<" "<<(B[1]-B_Numeric)[2]<<std::endl;

    // std::cout<<""<<std::endl;

    for (unsigned int j=0;j<3;j++)
    {
      deltaA[j] = (A[1][j]-A[0][j])/(fabs(A[0][j]) > 1.e-14 ? A[0][j] : 1.);
      deltaB[j] = (B[1][j]-B[0][j])/(fabs(B[0][j]) > 1.e-14 ? B[0][j] : 1.);
      deltaE[j] = (E[1][j]-E[0][j])/(fabs(E[0][j]) > 1.e-14 ? E[0][j] : 1.);
      for (unsigned int k=0;k<3;k++)
	deltaBp(j,k) = (Bp[1](j,k)-Bp[0](j,k))/(fabs(Bp[0](j,k)) > 1.e-14 ? Bp[0](j,k) : 1.);
    }

    // std::cout<<"deltaAx: "<<deltaA[0]<<std::endl;

    deltaPhi = (phi[1]-phi[0])/(fabs(phi[0]) > 1.e-14 ? phi[0] : 1.);

    deltaA_av += deltaA;
    deltaB_av += deltaB;
    deltaBp_av += deltaBp;
    deltaPhi_av += deltaPhi;
    deltaE_av += deltaE;

    for (unsigned int j=0;j<3;j++)
    {
      deltaA2_av[j] += deltaA[j]*deltaA[j];
      deltaB2_av[j] += deltaB[j]*deltaB[j];
      deltaE2_av[j] += deltaE[j]*deltaE[j];

      for (unsigned int k=0;k<3;k++)
	deltaBp2_av(j,k) += deltaBp(j,k)*deltaBp(j,k);

      if (i==0 || fabs(deltaA[j])<fabs(deltaA_min[j]))
	deltaA_min[j] = deltaA[j];
      if (i==0 || fabs(deltaB[j])<fabs(deltaB_min[j]))
	deltaB_min[j] = deltaB[j];
      if (i==0 || fabs(deltaE[j])<fabs(deltaE_min[j]))
	deltaE_min[j] = deltaE[j];

      for (unsigned int k=0;k<3;k++)
	if (i==0 || fabs(deltaBp(j,k))<fabs(deltaBp_min(j,k)))
	  deltaBp_min(j,k) = deltaBp(j,k);

      if (i==0 || fabs(deltaA[j])>fabs(deltaA_max[j]))
	deltaA_max[j] = deltaA[j];
      if (i==0 || fabs(deltaB[j])>fabs(deltaB_max[j]))
	deltaB_max[j] = deltaB[j];
      if (i==0 || fabs(deltaE[j])>fabs(deltaE_max[j]))
	deltaE_max[j] = deltaE[j];

      for (unsigned int k=0;k<3;k++)
	if (i==0 || fabs(deltaBp(j,k))>fabs(deltaBp_max(j,k)))
	  deltaBp_max(j,k) = deltaBp(j,k);
    }

    deltaPhi2_av += deltaPhi*deltaPhi;

    if (i==0 || fabs(deltaPhi)<fabs(deltaPhi_min))
      deltaPhi_min = deltaPhi;
    if (i==0 || fabs(deltaPhi)>fabs(deltaPhi_max))
      deltaPhi_max = deltaPhi;

    ticker.Tick(i);
  }

  ticker.EndTicker();

  deltaA_av/=nSamples;
  deltaB_av/=nSamples;
  deltaBp_av/=nSamples;
  deltaE_av/=nSamples;
  deltaPhi_av/=nSamples;

  deltaA2_av/=nSamples;
  deltaB2_av/=nSamples;
  deltaBp2_av/=nSamples;
  deltaE2_av/=nSamples;
  deltaPhi2_av/=nSamples;

  std::string index[3] = {"_x","_y","_z"};
  std::string indexx[3][3] = {{"_xx","_xy","_xz"},
			      {"_yx","_yy","_yz"},
			      {"_zx","_zy","_zz"}};
  unsigned int nPrecision = 6;

  std::cout<<"Value\tMean\t\tSigma\t\tMax\t\tMin"<<std::endl;
  for (unsigned int i=0;i<3;i++)
    std::cout<<std::setprecision(nPrecision)<<std::scientific<<"A"<<index[i]<<"\t"<<deltaA_av[i]<<"\t"<<sqrt(deltaA2_av[i] - deltaA_av[i]*deltaA_av[i])<<"\t"<<deltaA_max[i]<<"\t"<<deltaA_min[i]<<std::endl;
  for (unsigned int i=0;i<3;i++)
    std::cout<<std::setprecision(nPrecision)<<std::scientific<<"B"<<index[i]<<"\t"<<deltaB_av[i]<<"\t"<<sqrt(deltaB2_av[i] - deltaB_av[i]*deltaB_av[i])<<"\t"<<deltaB_max[i]<<"\t"<<deltaB_min[i]<<std::endl;
  for (unsigned int i=0;i<3;i++)
    std::cout<<std::setprecision(nPrecision)<<std::scientific<<"E"<<index[i]<<"\t"<<deltaE_av[i]<<"\t"<<sqrt(deltaE2_av[i] - deltaE_av[i]*deltaE_av[i])<<"\t"<<deltaE_max[i]<<"\t"<<deltaE_min[i]<<std::endl;
    std::cout<<std::setprecision(nPrecision)<<std::scientific<<"Phi"<<"\t"<<deltaPhi_av<<"\t"<<sqrt(deltaPhi2_av - deltaPhi_av*deltaPhi_av)<<"\t"<<deltaPhi_max<<"\t"<<deltaPhi_min<<std::endl;
  for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++)
      std::cout<<std::setprecision(nPrecision)<<std::scientific<<"Bp"<<indexx[i][j]<<"\t"<<deltaBp_av(i,j)<<"\t"<<sqrt(deltaBp2_av(i,j) - deltaBp_av(i,j)*deltaBp_av(i,j))<<"\t"<<deltaBp_max(i,j)<<"\t"<<deltaBp_min(i,j)<<std::endl;

  return 0;
}

template <class Solver>
void ComputeEField(Solver& s,
		   KPosition& P,
		   KEMThreeVector& E)
{
  static KDirection axis[3] = {KDirection(1.,0.,0.),
			       KDirection(0.,1.,0.),
			       KDirection(0.,0.,1.)};
  static double eps = 1.e-6;

  for (unsigned int i=0;i<3;i++)
  {
    E[i] = -(s.Potential(P+eps*axis[i]) - s.Potential(P-eps*axis[i]))/(2.*eps);
  }
}

template <class Solver>
void ComputeBField(Solver& s,
		   KPosition& P,
		   KEMThreeVector& B)
{
  static KDirection axis[3] = {KDirection(1.,0.,0.),
			       KDirection(0.,1.,0.),
			       KDirection(0.,0.,1.)};
  static double eps = 1.e-6;

  KEMThreeVector partialA[3];

  for (unsigned int i=0;i<3;i++)
    partialA[i] = (s.VectorPotential(P+eps*axis[i]) -
		   s.VectorPotential(P-eps*axis[i]))/(2.*eps);

  for (unsigned int i=0;i<3;i++)
  {
    B[i] = partialA[(i+1)%3][(i+2)%3] - partialA[(i+2)%3][(i+1)%3];
  }
}

