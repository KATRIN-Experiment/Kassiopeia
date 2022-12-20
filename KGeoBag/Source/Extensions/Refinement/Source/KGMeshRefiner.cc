#include "KGMeshRefiner.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

using katrin::KThreeVector;

namespace KGeoBag
{

KGMeshRefiner::KGMeshRefiner() :
    fMaxLength(0.), fMaxArea(0.), fMaxAspect(0.), fMaxNumRefinements(1)
{}

void KGMeshRefiner::VisitExtendedSurface(KGExtendedSurface<KGMesh>* meshSurface)
{
    if (meshSurface->HasExtension<KGMesh>()) {
        // first, fill container vector with deformed mesh elements (and delete
        // the old mesh elements as we go)
        for (auto & it : *(meshSurface->Elements())) {
            AddRefined(it, fMaxNumRefinements);
            delete it;
        }

        auto nAddedElements = fRefinedVector.size() - meshSurface->Elements()->size();
        if (nAddedElements > 0)
            coremsg(eNormal) << "KGMeshRefiner added " << nAddedElements << " elements to surface <" << meshSurface->GetName() << ">" << eom;

        // then, add the deformed mesh elements to the surface
        meshSurface->Elements()->clear();
        for (auto & it : fRefinedVector) {
            meshSurface->Elements()->push_back(it);
        }

        fRefinedVector.clear();
    }
}

void KGMeshRefiner::AddRefined(KGMeshElement* e, int maxDepth)
{
    if (auto* r = dynamic_cast<KGMeshRectangle*>(e))
        AddRefined(r, maxDepth);
    else if (auto* t = dynamic_cast<KGMeshTriangle*>(e))
        AddRefined(t, maxDepth);
    else if (auto* w = dynamic_cast<KGMeshWire*>(e))
        AddRefined(w, maxDepth);
}

void KGMeshRefiner::AddRefined(KGMeshRectangle* r, int maxDepth)
{
    if (maxDepth > 0) {
        KThreeVector p0 = r->GetP0();
        KThreeVector p1 = r->GetP0() + r->GetA() * r->GetN1();
        KThreeVector p2 = p1 + r->GetB() * r->GetN2();
        KThreeVector p3 = r->GetP0() + r->GetB() * r->GetN2();

        double d01 = (p1 - p0).Magnitude();
        double d03 = (p3 - p0).Magnitude();
        double dmax = fmax(d01, d03);

        if ((fMaxLength > 0 && dmax > fMaxLength) || (fMaxArea > 0 && r->Area() > fMaxArea) || (fMaxAspect > 0 && r->Aspect() > fMaxAspect)) {
            KThreeVector q1, q2;
            if (d01 >= d03) {
                q1 = p0 + 0.5 * (p1 - p0);
                q2 = p3 + 0.5 * (p2 - p3);
                AddRefined(new KGMeshRectangle(p0, q1, q2, p3), maxDepth-1);
                AddRefined(new KGMeshRectangle(q1, p1, p2, q2), maxDepth-1);
                return;
            }
            else if (d03 >= d01) {
                q1 = p0 + 0.5 * (p3 - p0);
                q2 = p1 + 0.5 * (p2 - p1);
                AddRefined(new KGMeshRectangle(p0, p1, q2, q1), maxDepth-1);
                AddRefined(new KGMeshRectangle(q2, p2, p3, q1), maxDepth-1);
                return;
            }
        }
    }

    fRefinedVector.push_back(new KGMeshRectangle(*r));
}

void KGMeshRefiner::AddRefined(KGMeshTriangle* t, int maxDepth)
{
    if (maxDepth > 0) {
        KThreeVector p0 = t->GetP0();
        KThreeVector p1 = t->GetP0() + t->GetA() * t->GetN1();
        KThreeVector p2 = t->GetP0() + t->GetB() * t->GetN2();

        double d01 = (p1 - p0).Magnitude();
        double d02 = (p2 - p0).Magnitude();
        double d12 = (p2 - p1).Magnitude();
        double dmax = fmax(d01, fmax(d02, d12));

        if ((fMaxLength > 0 && dmax > fMaxLength) || (fMaxArea > 0 && t->Area() > fMaxArea) || (fMaxAspect > 0 && t->Aspect() > fMaxAspect)) {
            // refine area by splitting into two triangles
            KThreeVector q;
            if (d12 >= d01 && d12 >= d02) {
                q = p1 + 0.5 * (p2 - p1);
                AddRefined(new KGMeshTriangle(p0, p1, q), maxDepth-1);
                AddRefined(new KGMeshTriangle(p0, q, p2), maxDepth-1);
                return;
            }
            else if (d01 >= d02 && d01 >= d12) {
                q = p0 + 0.5 * (p1 - p0);
                AddRefined(new KGMeshTriangle(p0, q, p2), maxDepth-1);
                AddRefined(new KGMeshTriangle(q, p1, p2), maxDepth-1);
                return;
            }
            else if (d02 >= d01 && d02 >= d12) {
                q = p0 + 0.5 * (p2 - p0);
                AddRefined(new KGMeshTriangle(p0, p1, q), maxDepth-1);
                AddRefined(new KGMeshTriangle(q, p1, p2), maxDepth-1);
                return;
            }
        }
    }

    fRefinedVector.push_back(new KGMeshTriangle(*t));
}

void KGMeshRefiner::AddRefined(KGMeshWire* w, int maxDepth)
{
    if (maxDepth > 0) {
        KThreeVector p0 = w->GetP0();
        KThreeVector p1 = w->GetP1();

        double d = (p1 - p0).Magnitude();

        if ((fMaxLength > 0 && d > fMaxLength) || (fMaxArea > 0 && w->Area() > fMaxArea) || (fMaxAspect > 0 && w->Aspect() > fMaxAspect)) {
            KThreeVector q;
            q = p0 + 0.5 * (p1 - p0);
            AddRefined(new KGMeshWire(p0, q, w->GetDiameter()), maxDepth-1);
            AddRefined(new KGMeshWire(q, p1, w->GetDiameter()), maxDepth-1);
            return;
        }
    }

    fRefinedVector.push_back(new KGMeshWire(*w));
}
}  // namespace KGeoBag
