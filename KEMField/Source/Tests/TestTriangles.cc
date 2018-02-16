#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include "KSurfaceTypes.hh"
#include "KSurface.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#include "KSurfaceContainer.hh"
#include "KBoundaryIntegralMatrix.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
  // This test produces a rectangle with randomly selected side lengths,
  // position, and orientation.  It then reproduces this rectangle using two
  // right triangles, and tests the potential at randomly selected points in
  // space 1.e6 times, and outputs the errors.

  // define the rectangle that will be defined 4 ways

  int nTest = 1.e6;

  srand((unsigned)time(0));

  double a = 1 + 5.*((double)rand())/RAND_MAX;
  double b = a+1;
  while (b>a)
    b = 1 + 5.*((double)rand())/RAND_MAX;

  a = b = 1.;

  KPosition p0(-4. + 8.*((double)rand())/RAND_MAX,
	       -4. + 8.*((double)rand())/RAND_MAX,
	       -4. + 8.*((double)rand())/RAND_MAX);

  p0[0] = 0;
  p0[1] = 0;
  p0[2] = 0;

  KDirection randomVec(-1. + 2.*((double)rand())/RAND_MAX,
		       -1. + 2.*((double)rand())/RAND_MAX,
		       -1. + 2.*((double)rand())/RAND_MAX);

  randomVec[0] = 1;
  randomVec[1] = 0;
  randomVec[2] = 0;

  randomVec = randomVec.Unit();

  KDirection n1 = randomVec;
  KDirection n2;

  KDirection randomVec2(-1. + 2.*((double)rand())/RAND_MAX,
			-1. + 2.*((double)rand())/RAND_MAX,
			-1. + 2.*((double)rand())/RAND_MAX);

  randomVec2[0] = 0;
  randomVec2[1] = 1;
  randomVec2[2] = 0;

  randomVec2 = randomVec2.Unit();

  // solve for n2 using Gramm-Schmidt procedure
  n2 = (randomVec2 - randomVec.Dot(randomVec2)*randomVec).Unit();

  std::cout<<"Rectangle to be tested: "<<std::endl;
  std::cout<<"   a,b: "<<a<<" "<<b<<std::endl;
  std::cout<<"   p0: ("<<p0[0]<<","<<p0[1]<<","<<p0[2]<<")"<<std::endl;
  std::cout<<"   n1: ("<<n1[0]<<","<<n1[1]<<","<<n1[2]<<")"<<std::endl;
  std::cout<<"   n2: ("<<n2[0]<<","<<n2[1]<<","<<n2[2]<<")"<<std::endl;
  std::cout<<""<<std::endl;
  std::cout<<"This rectangle will be decomposed into the following two triangles: "<<std::endl;

  double a_tri1 = a;
  double b_tri1 = b;
  KPosition p0_tri1 = p0;
  KDirection n1_tri1 = n1;
  KDirection n2_tri1 = n2;

  std::cout<<"Triangle 1: "<<std::endl;
  std::cout<<"   a,b: "<<a_tri1<<" "<<b_tri1<<std::endl;
  std::cout<<"   p0: ("<<p0_tri1[0]<<","<<p0_tri1[1]<<","<<p0_tri1[2]<<")"<<std::endl;
  std::cout<<"   n1: ("<<n1_tri1[0]<<","<<n1_tri1[1]<<","<<n1_tri1[2]<<")"<<std::endl;
  std::cout<<"   n2: ("<<n2_tri1[0]<<","<<n2_tri1[1]<<","<<n2_tri1[2]<<")"<<std::endl;
  std::cout<<""<<std::endl;

  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle>
    KEMRectangle;
  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>
    KEMTriangle;
  typedef KSurface<KElectrostaticBasis,KNeumannBoundary,KTriangle>
    KEMTriangle2;

  KEMTriangle* t1 = new KEMTriangle();
  t1->SetA(a_tri1);
  t1->SetB(b_tri1);
  t1->SetP0(p0_tri1);
  t1->SetN1(n1_tri1);
  t1->SetN2(n2_tri1);
  t1->SetBoundaryValue(1.);

  double a_tri2 = a;
  double b_tri2 = b;
  KPosition p0_tri2 = p0 + a*n1 + b*n2;
  KDirection n1_tri2 = -1.*n1;
  KDirection n2_tri2 = -1.*n2;

  std::cout<<"Triangle 2: "<<std::endl;
  std::cout<<"   a,b: "<<a_tri2<<" "<<b_tri2<<std::endl;
  std::cout<<"   p0: ("<<p0_tri2[0]<<","<<p0_tri2[1]<<","<<p0_tri2[2]<<")"<<std::endl;
  std::cout<<"   n1: ("<<n1_tri2[0]<<","<<n1_tri2[1]<<","<<n1_tri2[2]<<")"<<std::endl;
  std::cout<<"   n2: ("<<n2_tri2[0]<<","<<n2_tri2[1]<<","<<n2_tri2[2]<<")"<<std::endl;
  std::cout<<""<<std::endl;

  KEMTriangle2* t2 = new KEMTriangle2();
  t2->SetA(a_tri2);
  t2->SetB(b_tri2);
  t2->SetP0(p0_tri2);
  t2->SetN1(n1_tri2);
  t2->SetN2(n2_tri2);
  t2->SetNormalBoundaryFlux(2.);

  double a_tri3 = a;
  double b_tri3 = sqrt(a*a+b*b);
  KPosition p0_tri3 = p0;
  KDirection n1_tri3 = n1;
  KDirection n2_tri3 = (n1*a + n2*b).Unit();

  std::cout<<"Triangle 3: "<<std::endl;
  std::cout<<"   a,b: "<<a_tri3<<" "<<b_tri3<<std::endl;
  std::cout<<"   p0: ("<<p0_tri3[0]<<","<<p0_tri3[1]<<","<<p0_tri3[2]<<")"<<std::endl;
  std::cout<<"   n1: ("<<n1_tri3[0]<<","<<n1_tri3[1]<<","<<n1_tri3[2]<<")"<<std::endl;
  std::cout<<"   n2: ("<<n2_tri3[0]<<","<<n2_tri3[1]<<","<<n2_tri3[2]<<")"<<std::endl;
  std::cout<<""<<std::endl;

  KEMTriangle* t3 = new KEMTriangle();
  t3->SetA(a_tri3);
  t3->SetB(b_tri3);
  t3->SetP0(p0_tri3);
  t3->SetN1(n1_tri3);
  t3->SetN2(n2_tri3);
  t3->SetBoundaryValue(1.);

  double b_tri4 = sqrt(a*a+b*b);
  double a_tri4 = b;
  KPosition p0_tri4 = p0;
  KDirection n1_tri4 = n2;
  KDirection n2_tri4 = n2_tri3;

  std::cout<<"Triangle 4: "<<std::endl;
  std::cout<<"   a,b: "<<a_tri4<<" "<<b_tri4<<std::endl;
  std::cout<<"   p0: ("<<p0_tri4[0]<<","<<p0_tri4[1]<<","<<p0_tri4[2]<<")"<<std::endl;
  std::cout<<"   n1: ("<<n1_tri4[0]<<","<<n1_tri4[1]<<","<<n1_tri4[2]<<")"<<std::endl;
  std::cout<<"   n2: ("<<n2_tri4[0]<<","<<n2_tri4[1]<<","<<n2_tri4[2]<<")"<<std::endl;
  std::cout<<""<<std::endl;

  KEMTriangle2* t4 = new KEMTriangle2();
  t4->SetA(a_tri4);
  t4->SetB(b_tri4);
  t4->SetP0(p0_tri4);
  t4->SetN1(n1_tri4);
  t4->SetN2(n2_tri4);
  t4->SetNormalBoundaryFlux(2.);

  double a_rect1 = a;
  double b_rect1 = b;
  KPosition p0_rect1 = p0;
  KDirection n1_rect1 = n1;
  KDirection n2_rect1 = n2;

  KEMRectangle* r1 = new KEMRectangle();
  r1->SetA(a_rect1);
  r1->SetB(b_rect1);
  r1->SetP0(p0_rect1);
  r1->SetN1(n1_rect1);
  r1->SetN2(n2_rect1);
  r1->SetBoundaryValue(1.);

  KPosition P(0,0,0);
  KEMThreeVector field_1;
  KEMThreeVector field_2;

  double max[4] = {0,0,0,0};
  double min[4] = {1.e10,1.e10,1.e10,1.e10};
  double average[4] = {0,0,0,0};

  KSurfaceContainer sC;
  sC.push_back(t1);
  sC.push_back(t2);
  // sC.push_back(t3);
  // sC.push_back(t4);

  KElectrostaticBoundaryIntegrator integrator {KEBIFactory::MakeDefault()};
  KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(sC,integrator);

  for (unsigned int i=0;i<A.Dimension();i++)
    for (unsigned int j=0;j<A.Dimension();j++)
      std::cout<<"A("<<i<<","<<j<<"): "<<A(i,j)<<std::endl;

  for (int i=0;i<nTest;i++)
  {
    P[0] = -10. + 20.*((double)rand())/RAND_MAX;
    P[1] = -10. + 20.*((double)rand())/RAND_MAX;
    P[2] = -10. + 20.*((double)rand())/RAND_MAX;
    // P[0] = P[1] = 0.;
    // P[2] = 1.;
    // P[2] = 1.e-9;

    double phi_1;
    double phi_2;
    // integrator_tri.Evaluate(t1,rdummy,phi_1);
    // integrator_tri.Evaluate(t2,rdummy,phi_2);
    // phi_1 += phi_2;
    // integrator_rect.Evaluate(r1,rdummy,phi_2);

    phi_1 = integrator.Potential(t3,P);
    phi_2 = integrator.Potential(t4,P);
    phi_1 += phi_2;
    phi_2 = integrator.Potential(r1,P);

    field_1 = integrator.ElectricField(t3,P);
    field_2 = integrator.ElectricField(t4,P);
    field_1 += field_2;
    field_2 = integrator.ElectricField(r1,P);


    double value = fabs(phi_1 - phi_2)/phi_2*100;

    // if (value>1.)
    //   std::cout<<"Problem!"<<std::endl;
    // else
    //   std::cout<<"All clear!"<<std::endl;

    if (!(std::isinf(value)))
    {
      average[0] += fabs(value);

      if (max[0]<fabs(value))
	max[0] = fabs(value);
      if (min[0]>fabs(value))
	min[0] = fabs(value);
    }

    for (unsigned int j=0;j<3;j++)
    {
      value = fabs(field_1[j]-field_2[j])/field_2[j]*100.;
      average[j+1] += fabs(value);

      if (max[j+1]<fabs(value))
	max[j+1] = fabs(value);
      if (min[j+1]>fabs(value))
	min[j+1] = fabs(value);
    }

    if ((i*100)%(nTest)==0)
    {
      std::cout<<"\r";
      std::cout<<(int)100*i/(nTest)<<" %";
      std::cout.flush();
    }
  }
  std::cout<<"\r";

  for (int i=0;i<4;i++)
    average[i]/=nTest;

  std::cout<<""<<std::endl;
  std::cout<<"Accuracy Summary (2 triangles vs 1 rectangle): "<<std::endl;
  std::cout<<"\t Average \t Max \t\t Min"<<std::endl;
  std::cout<<std::setprecision(6)<<"Phi:\t "<<average[0]<<" \t "<<max[0]<<" \t "<<min[0]<<std::endl;
  std::cout<<std::setprecision(6)<<"Ex:\t "<<average[1]<<" \t "<<max[1]<<" \t "<<min[1]<<std::endl;
  std::cout<<std::setprecision(6)<<"Ey:\t "<<average[2]<<" \t "<<max[2]<<" \t "<<min[2]<<std::endl;
  std::cout<<std::setprecision(6)<<"Ez:\t "<<average[3]<<" \t "<<max[3]<<" \t "<<min[3]<<std::endl;
  std::cout<<""<<std::endl;

  //delete t1;
  //delete t2;
  delete r1;

  return 0;
}
