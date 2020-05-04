#ifndef KGMESHDEFORMERBUILDER_HH_
#define KGMESHDEFORMERBUILDER_HH_

#include "KComplexElement.hh"
#include "KGMeshDeformer.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGMeshDeformer> KGMeshDeformerBuilder;

template<> inline bool KGMeshDeformerBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "surfaces") {
        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsReference<string>());
        vector<KGSurface*>::iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            coremsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsReference<string>() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            tSurface->AcceptNode(fObject);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsReference<string>());
        vector<KGSpace*>::iterator tSpaceIt;
        KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            coremsg(eWarning) << "no spaces found for specifier <" << aContainer->AsReference<string>() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            tSpace->AcceptTree(fObject);
        }
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
