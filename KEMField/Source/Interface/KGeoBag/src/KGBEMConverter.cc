#include "KGBEMConverter.hh"

#include "KEMCoreMessage.hh"
using KEMField::kem_cout;

#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshRing.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

using KEMField::kem_cout;

#include "KThreeVector_KEMField.hh"
using KEMField::KPosition;
using KEMField::KDirection;

using katrin::KThreeVector;
using katrin::KAxis;

#include <cstddef>

namespace KGeoBag
{

KGBEMConverter::KGBEMConverter() :
    fSurfaceContainer(nullptr),
    fMinimumArea(0.),
    fMaximumAspectRatio(1e100),
    fVerbosity(0),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit),
    fCurrentOrigin(KThreeVector::sZero),
    fCurrentXAxis(KThreeVector::sXUnit),
    fCurrentYAxis(KThreeVector::sYUnit),
    fCurrentZAxis(KThreeVector::sZUnit),
    fCurrentElement(nullptr)

{}
KGBEMConverter::~KGBEMConverter()
{
    Clear();
}

void KGBEMConverter::Clear()
{
    //cout << "clearing content" << endl;

    for (auto& triangle : fTriangles) {
        delete triangle;
    }
    fTriangles.clear();

    for (auto& rectangle : fRectangles) {
        delete rectangle;
    }
    fRectangles.clear();

    for (auto& lineSegment : fLineSegments) {
        delete lineSegment;
    }
    fLineSegments.clear();

    for (auto& conicSection : fConicSections) {
        delete conicSection;
    }
    fConicSections.clear();

    for (auto& ring : fRings) {
        delete ring;
    }
    fRings.clear();

    for (auto& symmetricTriangle : fSymmetricTriangles) {
        delete symmetricTriangle;
    }
    fSymmetricTriangles.clear();

    for (auto& symmetricRectangle : fSymmetricRectangles) {
        delete symmetricRectangle;
    }
    fSymmetricRectangles.clear();

    for (auto& symmetricLineSegment : fSymmetricLineSegments) {
        delete symmetricLineSegment;
    }
    fSymmetricLineSegments.clear();

    for (auto& symmetricConicSection : fSymmetricConicSections) {
        delete symmetricConicSection;
    }
    fSymmetricConicSections.clear();

    for (auto& symmetricRing : fSymmetricRings) {
        delete symmetricRing;
    }
    fSymmetricRings.clear();

    return;
}

void KGBEMConverter::SetSystem(const KThreeVector& anOrigin, const KThreeVector& anXAxis, const KThreeVector& aYAxis,
                               const KThreeVector& aZAxis)
{
    fOrigin = anOrigin;
    fXAxis = anXAxis;
    fYAxis = aYAxis;
    fZAxis = aZAxis;
    fAxis.SetPoints(anOrigin, anOrigin + fZAxis);
    return;
}
const KThreeVector& KGBEMConverter::GetOrigin() const
{
    return fOrigin;
}
const KThreeVector& KGBEMConverter::GetXAxis() const
{
    return fXAxis;
}
const KThreeVector& KGBEMConverter::GetYAxis() const
{
    return fYAxis;
}
const KThreeVector& KGBEMConverter::GetZAxis() const
{
    return fZAxis;
}
const KAxis& KGBEMConverter::GetAxis() const
{
    return fAxis;
}

KThreeVector KGBEMConverter::GlobalToInternalPosition(const KThreeVector& aVector)
{
    KThreeVector tPosition(aVector - fOrigin);
    return KThreeVector(tPosition.Dot(fXAxis), tPosition.Dot(fYAxis), tPosition.Dot(fZAxis));
}
KThreeVector KGBEMConverter::GlobalToInternalVector(const KThreeVector& aVector)
{
    const KThreeVector& tVector(aVector);
    return KThreeVector(tVector.Dot(fXAxis), tVector.Dot(fYAxis), tVector.Dot(fZAxis));
}
KThreeVector KGBEMConverter::InternalToGlobalPosition(const KThreeVector& aVector)
{
    KThreeVector tPosition(aVector.X(), aVector.Y(), aVector.Z());
    return KThreeVector(fOrigin + tPosition.X() * fXAxis + tPosition.Y() * fYAxis + tPosition.Z() * fZAxis);
}
KThreeVector KGBEMConverter::InternalToGlobalVector(const KThreeVector& aVector)
{
    KThreeVector tVector(aVector.X(), aVector.Y(), aVector.Z());
    return KThreeVector(tVector.X() * fXAxis + tVector.Y() * fYAxis + tVector.Z() * fZAxis);
}

void KGBEMConverter::VisitSurface(KGSurface* aSurface)
{
    Clear();

    //cout << "visiting surface <" << aSurface->GetName() << ">..." << endl;

    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();
    fCurrentAxis.SetPoints(fCurrentOrigin, fCurrentOrigin + fCurrentZAxis);

    DispatchSurface(aSurface);

    return;
}
void KGBEMConverter::VisitSpace(KGSpace* aSpace)
{
    Clear();

    //cout << "visiting space <" << aSpace->GetName() << ">..." << endl;

    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();
    fCurrentAxis.SetPoints(fCurrentOrigin, fCurrentOrigin + fCurrentZAxis);

    DispatchSpace(aSpace);

    return;
}

KPosition KGBEMConverter::LocalToInternal(const KThreeVector& aVector)
{
    KThreeVector tGlobalVector(fCurrentOrigin + aVector.X() * fCurrentXAxis + aVector.Y() * fCurrentYAxis +
                               aVector.Z() * fCurrentZAxis);
    KThreeVector tInternalVector((tGlobalVector - fOrigin).Dot(fXAxis),
                                 (tGlobalVector - fOrigin).Dot(fYAxis),
                                 (tGlobalVector - fOrigin).Dot(fZAxis));
    return KPosition(tInternalVector.X(), tInternalVector.Y(), tInternalVector.Z());
}
KPosition KGBEMConverter::LocalToInternal(const katrin::KTwoVector& aVector)
{
    KThreeVector tGlobalVector = fCurrentOrigin + fCurrentZAxis * aVector.Z();
    katrin::KTwoVector tInternalVector((tGlobalVector - fOrigin).Dot(fZAxis), aVector.R());
    return KPosition(tInternalVector.R(), 0., tInternalVector.Z());
}

KGBEMMeshConverter::KGBEMMeshConverter() = default;
KGBEMMeshConverter::KGBEMMeshConverter(KEMField::KSurfaceContainer& aContainer)
{
    fSurfaceContainer.reset(&aContainer);
}
KGBEMMeshConverter::KGBEMMeshConverter(std::shared_ptr<KEMField::KSurfaceContainer> aContainer)
{
    fSurfaceContainer = aContainer;
}
KGBEMMeshConverter::~KGBEMMeshConverter() = default;

void KGBEMMeshConverter::DispatchSurface(KGSurface* aSurface)
{
    fCurrentElement = aSurface;
    Add(aSurface->AsExtension<KGMesh>());
    fCurrentElement = nullptr;
    return;
}
void KGBEMMeshConverter::DispatchSpace(KGSpace* aSpace)
{
    fCurrentElement = aSpace;
    Add(aSpace->AsExtension<KGMesh>());
    fCurrentElement = nullptr;
    return;
}

bool KGBEMMeshConverter::Add(KGMeshData* aData)
{
    KGMeshElement* tMeshElement;
    KGMeshTriangle* tMeshTriangle;
    KGMeshRectangle* tMeshRectangle;
    KGMeshWire* tMeshWire;

    Triangle* tTriangle;
    Rectangle* tRectangle;
    LineSegment* tLineSegment;

    if (aData != nullptr) {
        size_t tIgnored = 0;
        for (auto& tElementIt : *aData->Elements()) {
            tMeshElement = tElementIt;

            if (!tMeshElement)
                continue;

            tMeshTriangle = dynamic_cast<KGMeshTriangle*>(tMeshElement);
            if ((tMeshTriangle != nullptr) && (tMeshTriangle->Area() > fMinimumArea) &&
                (tMeshTriangle->Aspect() < fMaximumAspectRatio)) {
                tTriangle = new Triangle();
                tTriangle->SetName(tTriangle->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetPath() + ">") : ""));
                tTriangle->SetTagsFrom(fCurrentElement);
                tTriangle->SetValues(LocalToInternal(tMeshTriangle->GetP0()),
                                     LocalToInternal(tMeshTriangle->GetP1()),
                                     LocalToInternal(tMeshTriangle->GetP2()));
                fTriangles.push_back(tTriangle);
                continue;
            }

            tMeshRectangle = dynamic_cast<KGMeshRectangle*>(tMeshElement);
            if ((tMeshRectangle != nullptr) && (tMeshRectangle->Area() > fMinimumArea) &&
                (tMeshRectangle->Aspect() < fMaximumAspectRatio)) {
                tRectangle = new Rectangle();
                tRectangle->SetName(tRectangle->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetPath() + ">") : ""));
                tRectangle->SetTagsFrom(fCurrentElement);
                tRectangle->SetValues(LocalToInternal(tMeshRectangle->GetP0()),
                                      LocalToInternal(tMeshRectangle->GetP1()),
                                      LocalToInternal(tMeshRectangle->GetP2()),
                                      LocalToInternal(tMeshRectangle->GetP3()));
                fRectangles.push_back(tRectangle);
                continue;
            }

            tMeshWire = dynamic_cast<KGMeshWire*>(tMeshElement);
            if ((tMeshWire != nullptr) && (tMeshWire->Area() > fMinimumArea) &&
                (tMeshWire->Aspect() < fMaximumAspectRatio)) {
                tLineSegment = new LineSegment();
                tLineSegment->SetName(tLineSegment->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetPath() + ">") : ""));
                tLineSegment->SetTagsFrom(fCurrentElement);
                tLineSegment->SetValues(LocalToInternal(tMeshWire->GetP0()),
                                        LocalToInternal(tMeshWire->GetP1()),
                                        tMeshWire->GetDiameter());
                if (tMeshWire->Aspect() < 1.) {
                    kem_cout(eWarning) << "Attention at line segment at P0=" << (KThreeVector)(tLineSegment->GetP0())
                                       << ": Length < Diameter" << eom;
                    kem_cout(eNormal)
                        << "Wires are discretized too finely for the approximation of linear charge density to hold valid."
                        << ret << "Convergence problems of the Robin Hood charge density solver are expected." << ret
                        << "To avoid invalid elements, reduce mesh count and / or mesh power." << eom;
                }
                fLineSegments.push_back(tLineSegment);
                continue;
            }

            //kem_cout(eDebug) << "ignored mesh element of type " << tMeshElement->Name() << eom;
            tIgnored++;
        }
        if (tIgnored)
            kem_cout(eInfo) << "could not add " << tIgnored << " mesh elements that were not triangles, rectangles or wires"
                       << eom;
    }

    // clear current element
    fCurrentElement = nullptr;

    return true;
}

KGBEMAxialMeshConverter::KGBEMAxialMeshConverter() = default;
KGBEMAxialMeshConverter::KGBEMAxialMeshConverter(KEMField::KSurfaceContainer& aContainer)
{
    fSurfaceContainer.reset(&aContainer);
}
KGBEMAxialMeshConverter::KGBEMAxialMeshConverter(std::shared_ptr<KEMField::KSurfaceContainer> aContainer)
{
     fSurfaceContainer = aContainer;
}
KGBEMAxialMeshConverter::~KGBEMAxialMeshConverter() = default;

void KGBEMAxialMeshConverter::DispatchSurface(KGSurface* aSurface)
{
    fCurrentElement = aSurface;
    if (!Add(aSurface->AsExtension<KGAxialMesh>()))
        coremsg(eWarning) << "not adding surface <" << aSurface->GetPath() << "> since it is not coaxial" << eom;
    fCurrentElement = nullptr;
    return;
}
void KGBEMAxialMeshConverter::DispatchSpace(KGSpace* aSpace)
{
    fCurrentElement = aSpace;
    if (!Add(aSpace->AsExtension<KGAxialMesh>()))
        coremsg(eWarning) << "not adding space <" << aSpace->GetPath() << "> since it is not coaxial" << eom;
    fCurrentElement = nullptr;
    return;
}

bool KGBEMAxialMeshConverter::Add(KGAxialMeshData* aData)
{
    KGAxialMeshElement* tAxialMeshElement;
    KGAxialMeshLoop* tAxialMeshLoop;
    KGAxialMeshRing* tAxialMeshRing;

    ConicSection* tConicSection;
    Ring* tRing;

    if (aData != nullptr) {
        //cout << "adding axial mesh surface..." << endl;

        if (fAxis.EqualTo(fCurrentAxis) == false) {
            //cout << "...internal origin is <" << fOrigin << ">" << endl;
            //cout << "...internal z axis is <" << fZAxis << ">" << endl;
            //cout << "...current origin is <" << fCurrentOrigin << ">" << endl;
            //cout << "...current z axis is <" << fCurrentZAxis << ">" << endl;
            //cout << "...axes do not match!" << endl;
            return false;
        }

        size_t tIgnored = 0;
        for (auto& tElementIt : *aData->Elements()) {
            tAxialMeshElement = tElementIt;

            if (!tAxialMeshElement)
                continue;

            tAxialMeshLoop = dynamic_cast<KGAxialMeshLoop*>(tAxialMeshElement);
            if ((tAxialMeshLoop != nullptr) && (tAxialMeshLoop->Area() > fMinimumArea)) {
                tConicSection = new ConicSection();
                tConicSection->SetName(tConicSection->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetPath() + ">") : ""));
                tConicSection->SetTagsFrom(fCurrentElement);
                tConicSection->SetValues(LocalToInternal(tAxialMeshLoop->GetP0()),
                                         LocalToInternal(tAxialMeshLoop->GetP1()));
                fConicSections.push_back(tConicSection);
                continue;
            }

            tAxialMeshRing = dynamic_cast<KGAxialMeshRing*>(tAxialMeshElement);
            if ((tAxialMeshRing != nullptr) && (tAxialMeshRing->Area() > fMinimumArea)) {
                tRing = new Ring();
                tRing->SetName(tRing->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetPath() + ">") : ""));
                tRing->SetTagsFrom(fCurrentElement);
                tRing->SetValues(LocalToInternal(tAxialMeshRing->GetP0()));
                fRings.push_back(tRing);
                continue;
            }

            //kem_cout(eDebug) << "ignored mesh element of type " << tAxialMeshElement->Name() << eom;
            tIgnored++;
        }

        if (tIgnored)
            kem_cout(eInfo) << "could not add " << tIgnored << " axial mesh elements that were not loops or rings" << eom;

        //cout << "...added <" << fConicSections.size() + fRings.size() << "> components." << endl;
    }

    // clear current element
    fCurrentElement = nullptr;

    return true;
}

KGBEMDiscreteRotationalMeshConverter::KGBEMDiscreteRotationalMeshConverter() = default;
KGBEMDiscreteRotationalMeshConverter::KGBEMDiscreteRotationalMeshConverter(KEMField::KSurfaceContainer& aContainer)
{
    fSurfaceContainer.reset(&aContainer);
}
KGBEMDiscreteRotationalMeshConverter::KGBEMDiscreteRotationalMeshConverter(std::shared_ptr<KEMField::KSurfaceContainer> aContainer)
{
     fSurfaceContainer = aContainer;
}
KGBEMDiscreteRotationalMeshConverter::~KGBEMDiscreteRotationalMeshConverter() = default;

void KGBEMDiscreteRotationalMeshConverter::DispatchSurface(KGSurface* aSurface)
{
    fCurrentElement = aSurface;
    if (!Add(aSurface->AsExtension<KGDiscreteRotationalMesh>()))
        coremsg(eWarning) << "not adding surface <" << aSurface->GetPath() << "> since it is not coaxial" << eom;
    fCurrentElement = nullptr;
    return;
}
void KGBEMDiscreteRotationalMeshConverter::DispatchSpace(KGSpace* aSpace)
{
    fCurrentElement = aSpace;
    if (!Add(aSpace->AsExtension<KGDiscreteRotationalMesh>()))
        coremsg(eWarning) << "not adding space <" << aSpace->GetPath() << "> since it is not coaxial" << eom;
    fCurrentElement = nullptr;
    return;
}

bool KGBEMDiscreteRotationalMeshConverter::Add(KGDiscreteRotationalMeshData* aData)
{
    KGDiscreteRotationalMeshElement* tMeshElement;
    KGDiscreteRotationalMeshRectangle* tMeshRectangle;
    KGDiscreteRotationalMeshTriangle* tMeshTriangle;
    KGDiscreteRotationalMeshWire* tMeshWire;

    SymmetricTriangle* tTriangles;
    SymmetricRectangle* tRectangles;
    SymmetricLineSegment* tLineSegments;

    KPosition tCenter;
    KDirection tDirection;

    if (aData != nullptr) {
        //cout << "adding axial mesh surface..." << endl;

        if (fAxis.EqualTo(fCurrentAxis) == false) {
            // improve the hell out of this
            //cout << "...axes do not match!" << endl;
            return false;
        }

        tCenter.SetComponents(fAxis.GetCenter().X(), fAxis.GetCenter().Y(), fAxis.GetCenter().Z());
        tDirection.SetComponents(fAxis.GetDirection().X(), fAxis.GetDirection().Y(), fAxis.GetDirection().Z());

        size_t tIgnored = 0;
        for (auto& tElementIt : *aData->Elements()) {
            tMeshElement = tElementIt;

            if (!tMeshElement)
                continue;

            tMeshTriangle = dynamic_cast<KGDiscreteRotationalMeshTriangle*>(tMeshElement);
            if ((tMeshTriangle != nullptr) && (tMeshTriangle->Area() > fMinimumArea) &&
                (tMeshTriangle->Aspect() < fMaximumAspectRatio)) {
                tTriangles = new SymmetricTriangle();
                tTriangles->SetName(tTriangles->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetName() + ">") : ""));
                tTriangles->SetTagsFrom(fCurrentElement);
                tTriangles->NewElement()->SetValues(LocalToInternal(tMeshTriangle->Element().GetP0()),
                                                    LocalToInternal(tMeshTriangle->Element().GetP1()),
                                                    LocalToInternal(tMeshTriangle->Element().GetP2()));
                tTriangles->AddRotationsAboutAxis(tCenter, tDirection, tMeshTriangle->NumberOfElements());
                fSymmetricTriangles.push_back(tTriangles);
                continue;
            }

            tMeshRectangle = dynamic_cast<KGDiscreteRotationalMeshRectangle*>(tMeshElement);
            if ((tMeshRectangle != nullptr) && (tMeshRectangle->Area() > fMinimumArea) &&
                (tMeshRectangle->Aspect() < fMaximumAspectRatio)) {
                tRectangles = new SymmetricRectangle();
                tRectangles->SetName(tRectangles->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetName() + ">") : ""));
                tRectangles->SetTagsFrom(fCurrentElement);
                tRectangles->NewElement()->SetValues(LocalToInternal(tMeshRectangle->Element().GetP0()),
                                                     LocalToInternal(tMeshRectangle->Element().GetP1()),
                                                     LocalToInternal(tMeshRectangle->Element().GetP2()),
                                                     LocalToInternal(tMeshRectangle->Element().GetP3()));
                tRectangles->AddRotationsAboutAxis(tCenter, tDirection, tMeshRectangle->NumberOfElements());
                fSymmetricRectangles.push_back(tRectangles);
                continue;
            }

            tMeshWire = dynamic_cast<KGDiscreteRotationalMeshWire*>(tMeshElement);
            if ((tMeshWire != nullptr) && (tMeshWire->Area() > fMinimumArea) &&
                (tMeshWire->Aspect() < fMaximumAspectRatio)) {
                tLineSegments = new SymmetricLineSegment();
                tLineSegments->SetName(tLineSegments->Name() + (fCurrentElement ? ("<" + fCurrentElement->GetName() + ">") : ""));
                tLineSegments->SetTagsFrom(fCurrentElement);
                tLineSegments->NewElement()->SetValues(LocalToInternal(tMeshWire->Element().GetP0()),
                                                       LocalToInternal(tMeshWire->Element().GetP1()),
                                                       tMeshWire->Element().GetDiameter());
                tLineSegments->AddRotationsAboutAxis(tCenter, tDirection, tMeshWire->NumberOfElements());
                fSymmetricLineSegments.push_back(tLineSegments);
                continue;
            }

            //kem_cout(eDebug) << "ignored mesh element of type " << tMeshElement->Name() << eom;
            tIgnored++;
        }

        if (tIgnored)
            kem_cout(eInfo) << "could not add " << tIgnored
                       << " discrete-rotational mesh elements that were not triangles, rectangles or wires" << eom;
    }

    // clear current element
    fCurrentElement = nullptr;

    return true;
}
}  // namespace KGeoBag
