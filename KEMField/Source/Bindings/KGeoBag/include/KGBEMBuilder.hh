#ifndef KGELECTRODEBUILDER_HH_
#define KGELECTRODEBUILDER_HH_

#include "KGBEM.hh"
#include "KTagged.h"

namespace KGeoBag
{

template<class BasisPolicy, class BoundaryPolicy> class KGBEMAttributor;

template<class BasisPolicy>
class KGBEMAttributor<BasisPolicy, KDirichletBoundary> :
    public katrin::KTagged,
    public KGBEMData<BasisPolicy, KDirichletBoundary>
{
  public:
    KGBEMAttributor() : fSurfaces() {}
    ~KGBEMAttributor() override
    {
        KGExtendedSurface<KGBEM<BasisPolicy, KDirichletBoundary>>* tBEMSurface;
        for (auto& surface : fSurfaces) {
            tBEMSurface = surface->template MakeExtension<KGBEM<BasisPolicy, KDirichletBoundary>>();
            tBEMSurface->SetName(this->GetName());
            tBEMSurface->SetTags(this->GetTags());
            tBEMSurface->SetBoundaryValue(this->GetBoundaryValue());
        }
        KGExtendedSpace<KGBEM<BasisPolicy, KDirichletBoundary>>* tBEMSpace;
        for (auto& space : fSpaces) {
            tBEMSpace = space->template MakeExtension<KGBEM<BasisPolicy, KDirichletBoundary>>();
            tBEMSpace->SetName(this->GetName());
            tBEMSpace->SetTags(this->GetTags());
            tBEMSpace->SetBoundaryValue(this->GetBoundaryValue());
        }
    }

  public:
    void AddSurface(KGSurface* aSurface)
    {
        fSurfaces.push_back(aSurface);
    }
    void AddSpace(KGSpace* aSpace)
    {
        fSpaces.push_back(aSpace);
    }

  private:
    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;
};

template<class BasisPolicy>
class KGBEMAttributor<BasisPolicy, KNeumannBoundary> :
    public katrin::KTagged,
    public KGBEMData<BasisPolicy, KNeumannBoundary>
{
  public:
    KGBEMAttributor() : fSurfaces() {}
    ~KGBEMAttributor() override
    {
        KGExtendedSurface<KGBEM<BasisPolicy, KNeumannBoundary>>* tBEMSurface;
        for (auto& fSurface : fSurfaces) {
            tBEMSurface = fSurface->template MakeExtension<KGBEM<BasisPolicy, KNeumannBoundary>>();
            tBEMSurface->SetName(this->GetName());
            tBEMSurface->SetTags(this->GetTags());
            tBEMSurface->SetNormalBoundaryFlux(this->GetNormalBoundaryFlux());
        }
        KGExtendedSpace<KGBEM<BasisPolicy, KNeumannBoundary>>* tBEMSpace;
        for (auto& fSpace : fSpaces) {
            tBEMSpace = fSpace->template MakeExtension<KGBEM<BasisPolicy, KNeumannBoundary>>();
            tBEMSpace->SetName(this->GetName());
            tBEMSpace->SetTags(this->GetTags());
            tBEMSpace->SetNormalBoundaryFlux(this->GetNormalBoundaryFlux());
        }
    }

  public:
    void AddSurface(KGSurface* aSurface)
    {
        fSurfaces.push_back(aSurface);
    }
    void AddSpace(KGSpace* aSpace)
    {
        fSpaces.push_back(aSpace);
    }

  private:
    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;
};

typedef KGBEMAttributor<KElectrostaticBasis, KDirichletBoundary> KGElectrostaticDirichletAttributor;
using KGElectrostaticNeumannAttributor = KGBEMAttributor<KElectrostaticBasis, KNeumannBoundary>;
using KGMagnetostaticDirichletAttributor = KGBEMAttributor<KMagnetostaticBasis, KDirichletBoundary>;
using KGMagnetostaticNeumannAttributor = KGBEMAttributor<KMagnetostaticBasis, KNeumannBoundary>;

}  // namespace KGeoBag

#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGElectrostaticDirichletAttributor> KGElectrostaticDirichletBuilder;

template<> inline bool KGElectrostaticDirichletBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace KGeoBag;
    using namespace std;

    if (aContainer->GetName() == "name") {
        fObject->SetName(aContainer->AsString());
        return true;
    }
    if (aContainer->GetName() == "value") {
        fObject->SetBoundaryValue(aContainer->AsReference<double>());
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            coremsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        vector<KGSpace*>::const_iterator tSpaceIt;
        KGSpace* tSpace;
        const vector<KGSurface*>* tSurfaces;
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSpaces.size() == 0) {
            coremsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            tSurfaces = tSpace->GetBoundaries();
            for (tSurfaceIt = tSurfaces->begin(); tSurfaceIt != tSurfaces->end(); tSurfaceIt++) {
                tSurface = *tSurfaceIt;
                fObject->AddSurface(tSurface);
            }
        }
        return true;
    }
    return false;
}

using KGElectrostaticNeumannBuilder = KComplexElement<KGeoBag::KGElectrostaticNeumannAttributor>;

template<> inline bool KGElectrostaticNeumannBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace KGeoBag;
    using namespace std;

    if (aContainer->GetName() == "name") {
        fObject->SetName(aContainer->AsString());
        return true;
    }
    if (aContainer->GetName() == "flux") {
        fObject->SetNormalBoundaryFlux(aContainer->AsReference<double>());
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            coremsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return false;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        vector<KGSpace*>::const_iterator tSpaceIt;
        KGSpace* tSpace;
        const vector<KGSurface*>* tSurfaces;
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSpaces.size() == 0) {
            coremsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return false;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            tSurfaces = tSpace->GetBoundaries();
            for (tSurfaceIt = tSurfaces->begin(); tSurfaceIt != tSurfaces->end(); tSurfaceIt++) {
                tSurface = *tSurfaceIt;
                fObject->AddSurface(tSurface);
            }
        }
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
