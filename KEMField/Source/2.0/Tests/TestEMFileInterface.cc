#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>

#include "KEMFileInterface.hh"

#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KEMCout.hh"

#include "KMD5HashGenerator.hh"

#include "KBinaryDataStreamer.hh"

using namespace KEMField;

int main()
{
  std::set<std::string> fileList = KEMFileInterface::GetInstance()->FileList(".");

  std::cout<<"Files in this directory:"<<std::endl;
  for (std::set<std::string>::iterator it=fileList.begin();it!=fileList.end();++it)
    std::cout<<*it<<std::endl;

  std::cout<<""<<std::endl;

  double a = 1.5;
  double b = 1.3;
  KEMThreeVector p0(0.,0.,0.);
  KEMThreeVector n1(1./sqrt(2.),1./sqrt(2.),0.);
  KEMThreeVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

  double dirichletValue = 10.2;

  double chargeDensity = 4.8;

  KSurface<KElectrostaticBasis,
  	   KDirichletBoundary,
  	   KTriangle>* t = new KSurface<KElectrostaticBasis,
  					KDirichletBoundary,
  					KTriangle>();

  std::cout<<"\nOriginal element:\n"<<std::endl;

  t->SetA(a);
  t->SetB(b);
  t->SetP0(p0);
  t->SetN1(n1);
  t->SetN2(n2);

  t->SetBoundaryValue(dirichletValue);

  t->SetSolution(chargeDensity);

  // KEMField::cout<<*t<<KEMField::endl;

  KMD5HashGenerator* hashGenerator_noSolution = new KMD5HashGenerator();
  hashGenerator_noSolution->Omit(Type2Type<KElectrostaticBasis>());

  KEMFileInterface::GetInstance()->Write("test.kbd",*t,"a_triangle");
  KEMFileInterface::GetInstance()->Inspect("test.kbd");
  KEMFileInterface::GetInstance()->Write("test.kbd",*t,"b_triangle");
  KEMFileInterface::GetInstance()->Inspect("test.kbd");

  delete t;

  t = new KSurface<KElectrostaticBasis,
		   KDirichletBoundary,
		   KTriangle>();

  KEMFileInterface::GetInstance()->Read("test.kbd",*t,"b_triangle");

  // KEMField::cout<<*t<<KEMField::endl;

  t->SetBoundaryValue(200.);

  KEMFileInterface::GetInstance()->Overwrite("test.kbd",*t,"a_triangle");

  delete t;

  t = new KSurface<KElectrostaticBasis,
		   KDirichletBoundary,
		   KTriangle>();

  KEMFileInterface::GetInstance()->Read("test.kbd",*t,"a_triangle");

  // KEMField::cout<<*t<<KEMField::endl;

  std::remove("test.kbd");

  std::cout<<"making a directory: "<<std::endl;
  KEMFileInterface::GetInstance()->CreateDirectory("tmpdir");
  if (KEMFileInterface::GetInstance()->DirectoryExists("tmpdir"))
    std::cout<<"Directory created"<<std::endl;
  if (KEMFileInterface::GetInstance()->RemoveDirectory("tmpdir"))
    std::cout<<"Directory removed"<<std::endl;

  return 0;
}
