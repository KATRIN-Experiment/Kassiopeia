#include "KGDiscreteRotationalMesher.hh"

#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshRing.hh"
#include "KGDiscreteRotationalAreaMesher.hh"

#include <iostream>

namespace KGeoBag
{

KGDiscreteRotationalMesher::KGDiscreteRotationalMesher() : fCurrentElements(nullptr), fAxialAngle(0.), fAxialCount(100)
{}
KGDiscreteRotationalMesher::~KGDiscreteRotationalMesher() = default;

void KGDiscreteRotationalMesher::MeshAxialSurface(KGExtendedSurface<KGAxialMesh>* aSurface)
{
    for (auto& it : *aSurface->Elements())
        AddAxialMeshElement(it);
}

void KGDiscreteRotationalMesher::MeshAxialSpace(KGExtendedSpace<KGAxialMesh>* aSpace)
{
    for (auto& it : *aSpace->Elements())
        AddAxialMeshElement(it);
}

void KGDiscreteRotationalMesher::AddAxialMeshElement(KGAxialMeshElement* e)
{
    if (auto* l = dynamic_cast<KGAxialMeshLoop*>(e))
        AddAxialMeshLoop(l);
    else if (auto* r = dynamic_cast<KGAxialMeshRing*>(e))
        AddAxialMeshRing(r);
}

void KGDiscreteRotationalMesher::AddAxialMeshLoop(KGAxialMeshLoop* l)
{
    KTransformation transform;
    transform.SetRotationAxisAngle(fAxialAngle, 0., 0.);

    if (fabs((l->GetP0()[1]) - (l->GetP1()[1])) > 1.e-10) {

        KThreeVector P00((l->GetP0()[1]), 0., l->GetP0()[0]);
        KThreeVector P01((l->GetP0()[1]) * cos(2. * katrin::KConst::Pi() / fAxialCount),
                         (l->GetP0()[1]) * sin(2. * katrin::KConst::Pi() / fAxialCount),
                         l->GetP0()[0]);

        KThreeVector P10((l->GetP1()[1]), 0., l->GetP1()[0]);
        KThreeVector P11((l->GetP1()[1]) * cos(2. * katrin::KConst::Pi() / fAxialCount),
                         (l->GetP1()[1]) * sin(2. * katrin::KConst::Pi() / fAxialCount),
                         l->GetP1()[0]);


        KGMeshTriangle singleTriangle1(P00, P01, P11);
        KGMeshTriangle singleTriangle2(P10, P00, P11);


        singleTriangle1.Transform(transform);
        singleTriangle2.Transform(transform);
        auto* t1 = new KGDiscreteRotationalMeshTriangle(singleTriangle1);
        auto* t2 = new KGDiscreteRotationalMeshTriangle(singleTriangle2);
        t1->NumberOfElements(fAxialCount);
        t2->NumberOfElements(fAxialCount);
        fCurrentElements->push_back(t1);
        fCurrentElements->push_back(t2);
    }
    else {
        KThreeVector P0((l->GetP0()[1]), 0., l->GetP0()[0]);
        KThreeVector P1((l->GetP0()[1]) * cos(2. * katrin::KConst::Pi() / fAxialCount),
                        (l->GetP0()[1]) * sin(2. * katrin::KConst::Pi() / fAxialCount),
                        l->GetP0()[0]);

        KGMeshRectangle singleRectangle(fabs(l->GetP1()[0] - l->GetP0()[0]),
                                        (P1 - P0).Magnitude(),
                                        P0,
                                        KThreeVector(0., 0., 1.),
                                        (P1 - P0).Unit());

        singleRectangle.Transform(transform);
        auto* r = new KGDiscreteRotationalMeshRectangle(singleRectangle);
        r->NumberOfElements(fAxialCount);
        fCurrentElements->push_back(r);
    }
}

void KGDiscreteRotationalMesher::VisitSurface(KGSurface* aSurface)
{
    KGExtendedSurface<KGDiscreteRotationalMesh>* discRotMesh = aSurface->AsExtension<KGDiscreteRotationalMesh>();
    if (!discRotMesh)
        std::cerr << "KGDiscreteRotationalMesh assumes that extension is already present.\n";

    fCurrentElements = discRotMesh->Elements();

    KGExtendedSurface<KGAxialMesh>* axialMesh = aSurface->AsExtension<KGAxialMesh>();
    if (axialMesh)
        MeshAxialSurface(axialMesh);
    else {
        auto* areaMesher = new KGDiscreteRotationalAreaMesher();
        areaMesher->SetMeshElementOutput(fCurrentElements);
        aSurface->AcceptNode(areaMesher);
        delete areaMesher;
    }
}

void KGDiscreteRotationalMesher::VisitSpace(KGSpace* aSpace)
{
    KGExtendedSpace<KGDiscreteRotationalMesh>* discRotMesh = aSpace->AsExtension<KGDiscreteRotationalMesh>();
    if (!discRotMesh)
        std::cerr << "KGDiscreteRotationalMesh assumes that extension is already present.\n";

    fCurrentElements = discRotMesh->Elements();

    KGExtendedSpace<KGAxialMesh>* axialMesh = aSpace->AsExtension<KGAxialMesh>();
    if (axialMesh)
        MeshAxialSpace(axialMesh);
}


void KGDiscreteRotationalMesher::AddAxialMeshRing(KGAxialMeshRing* r)
{
    KTransformation transform;
    transform.SetRotationAxisAngle(fAxialAngle, 0., 0.);

    KGMeshWire singleWire(KThreeVector(r->GetP0()[1], 0., r->GetP0()[0]),
                          KThreeVector(r->GetP0()[1] * cos(2. * katrin::KConst::Pi() / fAxialCount),
                                       r->GetP0()[1] * sin(2. * katrin::KConst::Pi() / fAxialCount),
                                       r->GetP0()[0]),
                          r->GetD());
    singleWire.Transform(transform);
    auto* w = new KGDiscreteRotationalMeshWire(singleWire);
    w->NumberOfElements(fAxialCount);
    fCurrentElements->push_back(w);
}
}  // namespace KGeoBag
