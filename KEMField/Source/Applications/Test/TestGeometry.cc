#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>

#include "KDataDisplay.hh"
#include "KFundamentalTypeCounter.hh"

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KSerializer.hh"
#include "KEMFileInterface.hh"

#include "KMD5HashGenerator.hh"

#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KEMCout.hh"

using namespace KEMField;

int main()
{
  double a = 1.5;
  double b = 1.3;
  KThreeVector p0(0.,0.,0.);
  KThreeVector n1(1./sqrt(2.),1./sqrt(2.),0.);
  KThreeVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

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

  KEMField::cout<<"A:  "<<t->GetA()<<KEMField::endl;
  KEMField::cout<<"B:  "<<t->GetB()<<KEMField::endl;
  KEMField::cout<<"P0: "<<t->GetP0()<<KEMField::endl;
  KEMField::cout<<"N1: "<<t->GetN1()<<KEMField::endl;
  KEMField::cout<<"N2: "<<t->GetN2()<<KEMField::endl;
  KEMField::cout<<"Dirichlet boundary value: "<<t->GetBoundaryValue()<<KEMField::endl;
  KEMField::cout<<"Charge density: "<<t->GetSolution()<<KEMField::endl;
  KEMField::cout<<""<<KEMField::endl;

  KEMField::cout<<"Using the print method:"<<KEMField::endl;
  KEMField::cout<<*t<<KEMField::endl;

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KRectangle>* r = new KSurface<KElectrostaticBasis,
					KDirichletBoundary,
					KRectangle>();

  r->SetA(a);
  r->SetB(b);
  r->SetP0(p0);
  r->SetN1(n1);
  r->SetN2(n2);

  r->SetBoundaryValue(dirichletValue);

  r->SetSolution(chargeDensity);

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KRectangle>* rClone = r->Clone();

  std::cout<<"Testing the equality operator: "<<std::endl;
  if (*rClone == *r)
    std::cout<<"   == passed."<<std::endl;
  rClone->SetBoundaryValue(dirichletValue+1.);
  if (*rClone != *r)
    std::cout<<"   != passed."<<std::endl;
  delete rClone;

  KMD5HashGenerator hashGenerator;
  hashGenerator.Omit(Type2Type<KElectrostaticBasis>());
  std::cout<<"Old Hash for rectangle:  "<<hashGenerator.GenerateHash(*r)<<std::endl;
  r->SetSolution(chargeDensity+1.);
  std::cout<<"Same Hash for rectangle: "<<hashGenerator.GenerateHash(*r)<<std::endl;
  r->SetBoundaryValue(dirichletValue+1.);
  std::cout<<"New Hash for rectangle:  "<<hashGenerator.GenerateHash(*r)<<std::endl;
  r->SetBoundaryValue(dirichletValue);
  std::cout<<"Old Hash for rectangle:  "<<hashGenerator.GenerateHash(*r)<<std::endl;

  KSurface<KElectrostaticBasis,
	   KRobinBoundary,
	   KLineSegment>* w = new KSurface<KElectrostaticBasis,
					   KRobinBoundary,
					   KLineSegment>();

  w->SetP0(KThreeVector(0.,1.,0.));
  w->SetP1(KThreeVector(1.,0.,0.));
  w->SetDiameter(1.e-4);
  w->SetNormalBoundaryFlux(3.3);
  w->SetSolution(12.6);

  KFundamentalTypeCounter fundamentalTypeCounter;
  fundamentalTypeCounter << *w;
  std::cout<<"There are "<<fundamentalTypeCounter.NumberOfTypes()<<" things in "<<w->Name()<<std::endl;
  std::cout<<"There are "<<fundamentalTypeCounter.NumberOfType<double>()<<" doubles in "<<w->Name()<<std::endl;

  KSurfaceContainer surfaceContainer;

  std::cout<<"At the start, there are "<<surfaceContainer.size()<<" surfaces in the container"<<std::endl;

  KSurface<KElectrostaticBasis,
  	   KDirichletBoundary,
  	   KTriangle>* another_t = t->Clone();
    another_t->SetP0(KPosition(1,0,0));

  surfaceContainer.push_back(t->Clone());
  surfaceContainer.push_back(another_t);
  surfaceContainer.push_back(r->Clone());
  surfaceContainer.push_back(w->Clone());

  KEMField::cout<<"Now there are "<<surfaceContainer.size()<<" surfaces in the container"<<KEMField::endl;

  KEMField::cout<<"There are "<<surfaceContainer.size<KDirichletBoundary,KTriangle>()<<" dirichlet triangles in the container"<<KEMField::endl;

  delete t;
  t = 0;

  KSurfacePrimitive* sP = 0;
  KEMField::cout<<"Pulling from the container"<<KEMField::endl;
  sP = surfaceContainer.at(0);

  t = (KSurface<KElectrostaticBasis,
  		KDirichletBoundary,
  		KTriangle>*)sP;

  KEMField::cout<<"There are "<<surfaceContainer.size<KDirichletBoundary>()<<" dirichlet elements in the container"<<KEMField::endl;
  KEMField::cout<<"There are "<<surfaceContainer.size<KTriangle>()<<" triangles in the container"<<KEMField::endl;

  KEMField::cout<<"Pulled from container:"<<KEMField::endl;
  KEMField::cout<<"A:  "<<t->GetA()<<KEMField::endl;
  KEMField::cout<<"B:  "<<t->GetB()<<KEMField::endl;
  KEMField::cout<<"P0: "<<t->GetP0()<<KEMField::endl;
  KEMField::cout<<"N1: "<<t->GetN1()<<KEMField::endl;
  KEMField::cout<<"N2: "<<t->GetN2()<<KEMField::endl;
  KEMField::cout<<"Dirichlet boundary value: "<<t->GetBoundaryValue()<<KEMField::endl;
  KEMField::cout<<"Charge density: "<<t->GetSolution()<<KEMField::endl;
  std::cout<<""<<std::endl;

  {
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing Data Display test."<<KEMField::endl;

    KEMField::cout<<surfaceContainer<<KEMField::endl;

    KEMField::cout<<"Data Display test completed."<<KEMField::endl;
    std::cout<<"******************************************************************************\n"<<std::endl;
  }

  {
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing Metadata Streamer test."<<KEMField::endl;

    KMetadataStreamer mDS;
    mDS.open("testFile.smd","overwrite");
    mDS << surfaceContainer;
    KEMField::cout<<"\n"<<mDS.StringifyMetadata()<<std::endl;
    mDS.close();

    std::remove("testFile.smd");

    KEMField::cout<<"Metadata Streamer test completed."<<KEMField::endl;  
    std::cout<<"******************************************************************************"<<std::endl;
  }

  {
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing Binary Data Streamer test."<<KEMField::endl;

    KBinaryDataStreamer bDS;
    bDS.open("testFile.kbd","overwrite");
    bDS << surfaceContainer;
    bDS.close();

    KSurfaceContainer anotherSurfaceContainer;
    bDS.open("testFile.kbd","read");
    bDS >> anotherSurfaceContainer;
    bDS.close();

    if (surfaceContainer == anotherSurfaceContainer)
      KEMField::cout<<"Binary streamer passed!"<<KEMField::endl;
    else
      KEMField::cout<<"Binary streamer failed!"<<KEMField::endl;
    std::cout<<""<<std::endl;

    std::remove("testFile.kbd");

    KEMField::cout<<"Binary Data Streamer test completed."<<KEMField::endl;  
    std::cout<<"******************************************************************************"<<std::endl;
  }

  {
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing KSA Data Streamer test."<<KEMField::endl;

    KSADataStreamer saS;
    saS.open("testFile.zksa","overwrite");
    saS << surfaceContainer;
    saS.close();

    KSurfaceContainer anotherSurfaceContainer;
    saS.open("testFile.zksa","read");
    saS >> anotherSurfaceContainer;
    saS.close();

    if (surfaceContainer == anotherSurfaceContainer)
      KEMField::cout<<"KSA streamer passed!"<<KEMField::endl;
    else
      KEMField::cout<<"KSA streamer failed!"<<KEMField::endl;
    std::cout<<""<<std::endl;

    std::remove("testFile.zksa");

    KEMField::cout<<"KSA Data Streamer test completed."<<KEMField::endl;  
    std::cout<<"******************************************************************************"<<std::endl;
  }

  {
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing Serializer test."<<KEMField::endl;

    // KSerializer<KBinaryDataStreamer> serializer;
    KSerializer<KSADataStreamer> serializer;

    serializer.open("TestSerialization","overwrite");
    serializer << surfaceContainer;
    serializer.close();

    KSurfaceContainer anotherSurfaceContainer;
    serializer.open("TestSerialization","read");
    serializer >> anotherSurfaceContainer;

    if (surfaceContainer == anotherSurfaceContainer)
      KEMField::cout<<"Serializer passed!"<<KEMField::endl;
    else
      KEMField::cout<<"Serializer failed!"<<KEMField::endl;
    std::cout<<""<<std::endl;

    std::stringstream s;
    s << "TestSerialization"<<serializer.GetMetadataStreamer().GetFileSuffix();
    std::remove(s.str().c_str());
    s.clear();s.str("");
    s << "TestSerialization"<<serializer.GetDataStreamer().GetFileSuffix();
    std::remove(s.str().c_str());

    KEMField::cout<<"Serializer test completed."<<KEMField::endl;  
    std::cout<<"******************************************************************************"<<std::endl;
  }

  {
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing File Interface test."<<KEMField::endl;

    KEMFileInterface::GetInstance()->
      Write("testFile.kbd",surfaceContainer,"a_surface_container");

    KSurfaceContainer anotherSurfaceContainer;

    KEMFileInterface::GetInstance()->
      Read("testFile.kbd",anotherSurfaceContainer,"a_surface_container");

    if (surfaceContainer == anotherSurfaceContainer)
      KEMField::cout<<"File passed!"<<KEMField::endl;
    else
      KEMField::cout<<"File failed!"<<KEMField::endl;
    std::cout<<""<<std::endl;

    std::remove("testFile.kbd");

    KEMField::cout<<"File Interface test completed."<<KEMField::endl;  
    std::cout<<"******************************************************************************"<<std::endl;
  }

  {
    KSurfaceContainer::iterator it;
    std::cout<<"******************************************************************************"<<std::endl;
    KEMField::cout<<"Performing partially specialized iterator test."<<KEMField::endl;

    KEMField::cout<<"Line Segment elements:"<<KEMField::endl;
    for (it=surfaceContainer.begin<KLineSegment>();
    	 it!=surfaceContainer.end<KLineSegment>();it++)
      KEMField::cout<<*(*it)<<KEMField::endl;

    std::cout<<"\n........\n\n"<<std::endl;

    KEMField::cout<<"Dirichlet elements:"<<KEMField::endl;
    for (it=surfaceContainer.begin<KDirichletBoundary>();
    	 it!=surfaceContainer.end<KDirichletBoundary>();it++)
      KEMField::cout<<*(*it)<<KEMField::endl;

    std::cout<<"\n........\n\n"<<std::endl;

    KEMField::cout<<"Electrostatic elements:"<<KEMField::endl;
    for (it=surfaceContainer.begin<KElectrostaticBasis>();
    	 it!=surfaceContainer.end<KElectrostaticBasis>();it++)
      KEMField::cout<<*(*it)<<KEMField::endl;

    std::cout<<""<<std::endl;

    KEMField::cout<<"Partially specialized iterator test completed."<<KEMField::endl;
    std::cout<<"******************************************************************************"<<std::endl;
  }

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KTriangle>* t2 = t->Clone();

  KEMField::cout<<"Cloned:\n"<<KEMField::endl;
  KEMField::cout<<"A:  "<<t2->GetA()<<KEMField::endl;
  KEMField::cout<<"B:  "<<t2->GetB()<<KEMField::endl;
  KEMField::cout<<"P0: "<<t2->GetP0()<<KEMField::endl;
  KEMField::cout<<"N1: "<<t2->GetN1()<<KEMField::endl;
  KEMField::cout<<"N2: "<<t2->GetN2()<<KEMField::endl;
  KEMField::cout<<"Dirichlet boundary value: "<<t2->GetBoundaryValue()<<KEMField::endl;
  KEMField::cout<<"Charge density: "<<t2->GetSolution()<<KEMField::endl;
  std::cout<<""<<std::endl;

  KEMField::cout<<"trying KEMField::cout"<<KEMField::endl;
  KEMField::cout << (*t2) << KEMField::endl;
  KEMField::cout<<"done"<<KEMField::endl;
  std::cout<<""<<std::endl;

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KTriangle>* t3 = new KSurface<KElectrostaticBasis,
					 KDirichletBoundary,
					 KTriangle>();

  KSurfacePrimitive* surfacePrim = t3;

  KSABuffer buf;
  buf.Clear();
  buf << *t;
  // buf >> *t3;
  buf >> *surfacePrim;

  KEMField::cout<<"Streaming:\n"<<KEMField::endl;
  KEMField::cout<<"A:  "<<t3->GetA()<<KEMField::endl;
  KEMField::cout<<"B:  "<<t3->GetB()<<KEMField::endl;
  KEMField::cout<<"P0: "<<t3->GetP0()<<KEMField::endl;
  KEMField::cout<<"N1: "<<t3->GetN1()<<KEMField::endl;
  KEMField::cout<<"N2: "<<t3->GetN2()<<KEMField::endl;
  KEMField::cout<<"Dirichlet boundary value: "<<t3->GetBoundaryValue()<<KEMField::endl;
  KEMField::cout<<"Charge density: "<<t3->GetSolution()<<KEMField::endl;
  std::cout<<""<<std::endl;

  surfaceContainer.clear();
}
