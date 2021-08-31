#ifndef KGGEOMETRYPRINTERBUILDER_HH_
#define KGGEOMETRYPRINTERBUILDER_HH_

#include "KComplexElement.hh"
#include "KGGeometryPrinter.hh"
#include "KGBindingsMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGGeometryPrinter> KGGeometryPrinterBuilder;

template<> inline bool KGGeometryPrinterBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace KGeoBag;
    using namespace std;

    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KGGeometryPrinter::SetFile);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KGGeometryPrinter::SetPath);
        return true;
    }
    if (aContainer->GetName() == "write_json") {
        aContainer->CopyTo(fObject, &KGGeometryPrinter::SetWriteJSON);
        return true;
    }
    if (aContainer->GetName() == "write_xml") {
        aContainer->CopyTo(fObject, &KGGeometryPrinter::SetWriteXML);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        if (aContainer->AsString().size() == 0) {
            return true;
        }

        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            bindmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        if (aContainer->AsString().size() == 0) {
            return true;
        }

        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        vector<KGSpace*>::const_iterator tSpaceIt;
        KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            bindmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddSpace(tSpace);
        }
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
