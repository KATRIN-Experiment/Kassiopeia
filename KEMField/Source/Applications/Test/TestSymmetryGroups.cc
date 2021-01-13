#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KEMCout.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGConicalWireArraySurface.hh"
#include "KGDiscreteRotationalMesher.hh"
#include "KRobinHood.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#endif

using namespace KEMField;

int main(int argc, char** argv)
{
    int test = 0;
    if (argc > 1)
        test = atoi(argv[1]);

    if (test == 0) {
        double a = 1.5;
        double b = 1.3;
        KFieldVector p0(0., 0., -1.);
        KFieldVector n1(1. / sqrt(2.), 1. / sqrt(2.), 0.);
        KFieldVector n2(1. / sqrt(2.), -1. / sqrt(2.), 0.);

        double dirichletValue = 10.2;

        double chargeDensity = 4.8;

        auto* tg = new KSurface<KElectrostaticBasis, KDirichletBoundary, KSymmetryGroup<KRectangle>>();

        KRectangle* t = tg->NewElement();

        t->SetA(a);
        t->SetB(b);
        t->SetP0(p0);
        t->SetN1(n1);
        t->SetN2(n2);

        tg->SetBoundaryValue(dirichletValue);
        tg->SetSolution(chargeDensity);

        KEMField::cout << "Before symmetry: " << KEMField::endl;
        KEMField::cout << *tg << KEMField::endl;

        tg->AddReflectionThroughPlane(KFieldVector(0., 0., 0.), KFieldVector(0., 0., -1.));

        KEMField::cout << "After symmetry: " << KEMField::endl;
        KEMField::cout << *tg << KEMField::endl;
    }
    else if (test == 1) {
#ifdef KEMFIELD_USE_KGEOBAG
        auto* wireArray = new KGeoBag::KGConicalWireArray();
        wireArray->SetR1(1.);
        wireArray->SetZ1(0.);
        wireArray->SetR2(1.5);
        wireArray->SetZ2(1.);
        wireArray->SetNWires(100);
        wireArray->SetDiameter(.003);
        wireArray->SetNDisc(30);

        auto* wASurface = new KGeoBag::KGConicalWireArraySurface(wireArray);
        auto* wireArraySurface = new KGeoBag::KGSurface(wASurface);
        wireArraySurface->SetName("conicalWireArray");
        wireArraySurface->MakeExtension<KGeoBag::KGDiscreteRotationalMesh>();
        wireArraySurface->MakeExtension<KGeoBag::KGElectrostaticDirichlet>();
        wireArraySurface->AsExtension<KGeoBag::KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        auto* mesher = new KGeoBag::KGDiscreteRotationalMesher();
        mesher->SetAxialCount(100);
        wireArraySurface->AcceptNode(mesher);

        KSurfaceContainer surfaceContainer;
        KGeoBag::KGBEMDiscreteRotationalMeshConverter geometryConverter(surfaceContainer);
        wireArraySurface->AcceptNode(&geometryConverter);

        KEMField::cout << "solving for " << surfaceContainer.size() << " elements" << KEMField::endl;

        KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
        robinHood.Solve(A, x, b);

#ifdef KEMFIELD_USE_VTK
        KEMVTKViewer viewer(surfaceContainer);
        viewer.GenerateGeometryFile("conicalWireArray.vtp");
        viewer.ViewGeometry();
#endif

        delete wireArray;
#endif
    }

    return 0;
}
