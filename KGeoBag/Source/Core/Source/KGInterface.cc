#include "KGCore.hh"

#include <boost/algorithm/string.hpp>
#include <utility>
#include "KStringUtils.h"

using namespace std;

namespace KGeoBag
{

KGInterface* KGInterface::sInstance = nullptr;

const char KGInterface::sSeparator[] = "\t ,;";
const char KGInterface::sNest = '/';
const char KGInterface::sTag = '@';
const char KGInterface::sRecurse = ':';
const char KGInterface::sWildcard = '#';

KGInterface* KGInterface::GetInstance()
{
    if (sInstance == nullptr) {
        sInstance = new KGInterface();
    }

    return sInstance;
}
KGInterface* KGInterface::DeleteInstance()
{
    if (sInstance != nullptr) {
        delete sInstance;
        sInstance = nullptr;
    }

    return sInstance;
}

KGInterface::KGInterface()
{
    fRoot = new KGSpace();
    fRoot->SetName("(interface)");
}
KGInterface::~KGInterface()
{
    delete fRoot;
}

void KGInterface::InstallSpace(KGSpace* aSpace)
{
    fRoot->AddChildSpace(aSpace);
    return;
}
void KGInterface::InstallSurface(KGSurface* aSurface)
{
    fRoot->AddChildSurface(aSurface);
    return;
}

vector<KGSurface*> KGInterface::RetrieveSurfaces()
{
    coremsg_debug("retrieving all surfaces..." << eom);

    vector<KGSurface*> tAccumulator;
    RetrieveSurfacesByWildcard(tAccumulator, fRoot, -1);

    coremsg_debug("...done" << eom);

    return tAccumulator;
}
vector<KGSurface*> KGInterface::RetrieveSurfaces(const string& aSpecifier)
{
    coremsg_debug("retrieving surfaces for <" << aSpecifier << ">..." << eom);

    vector<KGSurface*> tAccumulator;
    RetrieveSurfacesBySpecifier(tAccumulator, fRoot, aSpecifier);

    coremsg_debug("...done" << eom);

    return tAccumulator;
}
KGSurface* KGInterface::RetrieveSurface(const string& aSpecifier)
{
    coremsg_debug("retrieving surface for <" << aSpecifier << ">..." << eom);

    vector<KGSurface*> tAccumulator;
    RetrieveSurfacesBySpecifier(tAccumulator, fRoot, aSpecifier);

    coremsg_debug("...done" << eom);

    if (tAccumulator.empty()) {
        coremsg(eWarning) << "no surfaces registered for path <" << aSpecifier << ">" << eom;
        return nullptr;
    }
    if (tAccumulator.size() != 1) {
        coremsg(eWarning) << "multiple surfaces registered for path <" << aSpecifier << ">" << eom;
        return nullptr;
    }

    return (*tAccumulator.begin());
}

vector<KGSpace*> KGInterface::RetrieveSpaces()
{
    coremsg_debug("retrieving all spaces..." << eom);

    vector<KGSpace*> tAccumulator;
    RetrieveSpacesByWildcard(tAccumulator, fRoot, -1);

    coremsg_debug("...done" << eom);

    return tAccumulator;
}
vector<KGSpace*> KGInterface::RetrieveSpaces(const string& aSpecifier)
{
    coremsg_debug("retrieving spaces for <" << aSpecifier << ">..." << eom);

    vector<KGSpace*> tAccumulator;
    RetrieveSpacesBySpecifier(tAccumulator, fRoot, aSpecifier);

    coremsg_debug("...done" << eom);

    return tAccumulator;
}
KGSpace* KGInterface::RetrieveSpace(const string& aSpecifier)
{
    coremsg_debug("retrieving space for <" << aSpecifier << ">..." << eom);

    vector<KGSpace*> tAccumulator;
    RetrieveSpacesBySpecifier(tAccumulator, fRoot, aSpecifier);

    coremsg_debug("...done" << eom);

    if (tAccumulator.empty()) {
        coremsg(eWarning) << "no spaces registered for path <" << aSpecifier << ">" << eom;
        return nullptr;
    }
    if (tAccumulator.size() != 1) {
        coremsg(eWarning) << "multiple spaces registered for path <" << aSpecifier << ">" << eom;
        return nullptr;
    }

    return (*tAccumulator.begin());
}

void KGInterface::RetrieveSurfacesBySpecifier(vector<KGSurface*>& anAccumulator, KGSpace* aNode, string aSpecifier)
{
    vector<string> tPathList;
    boost::split(tPathList, aSpecifier, boost::is_any_of(sSeparator));

    if (aSpecifier.find_first_of(sTag) == 0){
        coremsg(eWarning) << "Path definition for surfaces just contains a tag, which can make it ambiguous: <" << aSpecifier << ">. Please specify a distinct geometry path!" << eom;
    }

    coremsg_debug("  retrieving surfaces for specifier <" << aSpecifier << "> at <" << aNode->GetName() << ">" << eom);

    for (auto& tPath : tPathList) {
        coremsg_debug("  retrieving final surfaces for path <" << tPath << ">" << eom);

        RetrieveSurfacesByPath(anAccumulator, aNode, tPath);
    }
}

void KGInterface::RetrieveSpacesBySpecifier(vector<KGSpace*>& anAccumulator, KGSpace* aNode, string aSpecifier)
{
    vector<string> tPathList;
    boost::split(tPathList, aSpecifier, boost::is_any_of(sSeparator));

    if (aSpecifier.find_first_of(sTag) == 0){
        coremsg(eWarning) << "Path definition for spaces just contains a tag, which can make it ambiguous: <" << aSpecifier << ">. Please specify a distinct geometry path!" << eom;
    }

    coremsg_debug("  retrieving spaces for specifier <" << aSpecifier << "> at <" << aNode->GetName() << ">" << eom);

    for (auto& tPath : tPathList) {
        coremsg_debug("  retrieving final spaces for path <" << tPath << ">" << eom);

        RetrieveSpacesByPath(anAccumulator, aNode, tPath);
    }
}

void KGInterface::RetrieveSurfacesByPath(vector<KGSurface*>& anAccumulator, KGSpace* aNode, const string& aPath)
{
    size_t tNestPos = aPath.find_first_of(sNest);
    string tHead = aPath.substr(0, tNestPos);
    string tTail = aPath.substr(tNestPos + 1);

    coremsg_debug("  retrieving surfaces for path <" << aPath << "> at <" << aNode->GetName() << ">" << eom);

    if (tNestPos == string::npos) {
        if (tHead.find_first_of(sTag) == 0) {
            string tTag;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tTag = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tTag = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving final surfaces for tag <" << tTag << "> with recursion <" << tRecursion << ">"
                                                                  << eom);

            RetrieveSurfacesByTag(anAccumulator, aNode, tTag, tRecursion);

            return;
        }

        if (tHead.find_first_of(sWildcard) == 0) {
            string tWildcard;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tWildcard = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tWildcard = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving final surfaces for wildcard with recursion <" << tRecursion << ">" << eom);

            RetrieveSurfacesByWildcard(anAccumulator, aNode, tRecursion);

            return;
        }

        const string& tName = tHead;

        coremsg_debug("  retrieving final surfaces for name <" << tName << ">" << eom);

        RetrieveSurfacesByName(anAccumulator, aNode, tName);

        return;
    }
    else {
        vector<KGSpace*> tAccumulator;

        if (tHead.find_first_of(sTag) == 0) {
            string tTag;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tTag = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tTag = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving spaces for tag <" << tTag << "> with recursion <" << tRecursion << ">" << eom);

            RetrieveSpacesByTag(tAccumulator, aNode, tTag, tRecursion);

            for (auto& tIt : tAccumulator) {
                RetrieveSurfacesByPath(anAccumulator, tIt, tTail);
            }

            return;
        }

        if (tHead.find_first_of(sWildcard) == 0) {
            string tWildcard;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tWildcard = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tWildcard = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving spaces for wildcard with recursion <" << tRecursion << ">" << eom);

            RetrieveSpacesByWildcard(tAccumulator, aNode, tRecursion);

            for (auto& tIt : tAccumulator) {
                RetrieveSurfacesByPath(anAccumulator, tIt, tTail);
            }

            return;
        }

        const string& tName = tHead;

        coremsg_debug("  retrieving spaces for name <" << tName << ">" << eom);

        RetrieveSpacesByName(tAccumulator, aNode, tName);

        for (auto& tIt : tAccumulator) {
            RetrieveSurfacesByPath(anAccumulator, tIt, tTail);
        }

        return;
    }
}

void KGInterface::RetrieveSpacesByPath(vector<KGSpace*>& anAccumulator, KGSpace* aNode, const string& aPath)
{
    size_t tNestPos = aPath.find_first_of(sNest);
    string tHead = aPath.substr(0, tNestPos);
    string tTail = aPath.substr(tNestPos + 1);

    coremsg_debug("  retrieving spaces for path <" << aPath << "> at <" << aNode->GetName() << ">" << eom);

    if (tNestPos == string::npos) {
        if (tHead.find_first_of(sTag) == 0) {
            string tTag;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tTag = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tTag = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving final spaces for tag <" << tTag << "> with recursion <" << tRecursion << ">"
                                                                << eom);

            RetrieveSpacesByTag(anAccumulator, aNode, tTag, tRecursion);

            return;
        }

        if (tHead.find_first_of(sWildcard) == 0) {
            string tWildcard;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tWildcard = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tWildcard = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving final spaces for wildcard with recursion <" << tRecursion << ">" << eom);

            RetrieveSpacesByWildcard(anAccumulator, aNode, tRecursion);

            return;
        }

        const string& tName = tHead;

        coremsg_debug("  retrieving final spaces for name <" << tName << ">" << eom);

        RetrieveSpacesByName(anAccumulator, aNode, tName);
        return;
    }
    else {
        vector<KGSpace*> tAccumulator;

        if (tHead.find_first_of(sTag) == 0) {
            string tTag;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tTag = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tTag = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving spaces for tag <" << tTag << "> with recursion <" << tRecursion << ">" << eom);

            RetrieveSpacesByTag(tAccumulator, aNode, tTag, tRecursion);

            for (auto& tIt : tAccumulator) {
                RetrieveSpacesByPath(anAccumulator, tIt, tTail);
            }

            return;
        }

        if (tHead.find_first_of(sWildcard) == 0) {
            string tWildcard;
            int tRecursion;

            size_t tRecursePos = tHead.find_first_of(sRecurse);
            if (tRecursePos != string::npos) {
                tWildcard = tHead.substr(1, tRecursePos - 1);
                stringstream tConverter(tHead.substr(tRecursePos + 1));
                tConverter >> tRecursion;
            }
            else {
                tWildcard = tHead.substr(1);
                tRecursion = -1;
            }

            coremsg_debug("  retrieving spaces for wildcard with recursion <" << tRecursion << ">" << eom);

            RetrieveSpacesByWildcard(tAccumulator, aNode, tRecursion);

            for (auto& tIt : tAccumulator) {
                RetrieveSpacesByPath(anAccumulator, tIt, tTail);
            }

            return;
        }

        const string& tName = tHead;

        coremsg_debug("  retrieving spaces for name <" << tName << ">" << eom);

        RetrieveSpacesByName(tAccumulator, aNode, tName);

        for (auto& tIt : tAccumulator) {
            RetrieveSpacesByPath(anAccumulator, tIt, tTail);
        }

        return;
    }
}

void KGInterface::RetrieveSurfacesByName(vector<KGSurface*>& anAccumulator, KGSpace* aNode, const string& aName)
{
    KGSurface* tBoundary;
    vector<KGSurface*>::const_iterator tBoundaryIt;
    for (tBoundaryIt = aNode->GetBoundaries()->begin(); tBoundaryIt != aNode->GetBoundaries()->end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        if (tBoundary->HasName(aName)) {
            coremsg_debug("    boundary surface of <" << aNode->GetName() << "> has matching name <" << aName << ">"
                                                      << eom) anAccumulator.push_back(tBoundary);
        }
    }

    KGSurface* tSurface;
    vector<KGSurface*>::const_iterator tSurfaceIt;
    for (tSurfaceIt = aNode->GetChildSurfaces()->begin(); tSurfaceIt != aNode->GetChildSurfaces()->end();
         tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        if (tSurface->HasName(aName)) {
            coremsg_debug("    child surface of <" << aNode->GetName() << "> has matching name <" << aName << ">"
                                                   << eom) anAccumulator.push_back(tSurface);
        }
    }

    return;
}
void KGInterface::RetrieveSpacesByName(vector<KGSpace*>& anAccumulator, KGSpace* aNode, const string& aName)
{
    KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = aNode->GetChildSpaces()->begin(); tSpaceIt != aNode->GetChildSpaces()->end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        if (tSpace->HasName(aName)) {
            coremsg_debug("    child space of <" << aNode->GetName() << "> has matching name <" << aName << ">" << eom)
                anAccumulator.push_back(tSpace);
        }
    }

    return;
}

void KGInterface::RetrieveSurfacesByTag(vector<KGSurface*>& anAccumulator, KGSpace* aNode, const string& aTag,
                                        int aDepth)
{
    if (aDepth == 0) {
        return;
    }

    KGSurface* tBoundary;
    vector<KGSurface*>::const_iterator tBoundaryIt;
    for (tBoundaryIt = aNode->GetBoundaries()->begin(); tBoundaryIt != aNode->GetBoundaries()->end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        if (tBoundary->HasTag(aTag)) {
            coremsg_debug("    boundary surface of <" << aNode->GetName() << "> has matching tag <" << aTag << ">"
                                                      << eom) anAccumulator.push_back(tBoundary);
        }
    }

    KGSurface* tSurface;
    vector<KGSurface*>::const_iterator tSurfaceIt;
    for (tSurfaceIt = aNode->GetChildSurfaces()->begin(); tSurfaceIt != aNode->GetChildSurfaces()->end();
         tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        if (tSurface->HasTag(aTag)) {
            coremsg_debug("    child surface of <" << aNode->GetName() << "> has matching tag <" << aTag << ">" << eom)
                anAccumulator.push_back(tSurface);
        }
    }

    KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = aNode->GetChildSpaces()->begin(); tSpaceIt != aNode->GetChildSpaces()->end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        RetrieveSurfacesByTag(anAccumulator, tSpace, aTag, aDepth - 1);
    }

    return;
}
void KGInterface::RetrieveSpacesByTag(vector<KGSpace*>& anAccumulator, KGSpace* aNode, const string& aTag, int aDepth)
{
    if (aDepth == 0) {
        return;
    }

    KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = aNode->GetChildSpaces()->begin(); tSpaceIt != aNode->GetChildSpaces()->end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        if (tSpace->HasTag(aTag)) {
            coremsg_debug("    child space of <" << aNode->GetName() << "> has matching tag <" << aTag << ">" << eom)
                anAccumulator.push_back(tSpace);
        }
    }

    for (tSpaceIt = aNode->GetChildSpaces()->begin(); tSpaceIt != aNode->GetChildSpaces()->end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        RetrieveSpacesByTag(anAccumulator, tSpace, aTag, aDepth - 1);
    }

    return;
}

void KGInterface::RetrieveSurfacesByWildcard(vector<KGSurface*>& anAccumulator, KGSpace* aNode, int aDepth)
{
    if (aDepth == 0) {
        return;
    }

    KGSurface* tBoundary;
    vector<KGSurface*>::const_iterator tBoundaryIt;
    for (tBoundaryIt = aNode->GetBoundaries()->begin(); tBoundaryIt != aNode->GetBoundaries()->end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        anAccumulator.push_back(tBoundary);
    }

    KGSurface* tSurface;
    vector<KGSurface*>::const_iterator tSurfaceIt;
    for (tSurfaceIt = aNode->GetChildSurfaces()->begin(); tSurfaceIt != aNode->GetChildSurfaces()->end();
         tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        anAccumulator.push_back(tSurface);
    }

    KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = aNode->GetChildSpaces()->begin(); tSpaceIt != aNode->GetChildSpaces()->end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        RetrieveSurfacesByWildcard(anAccumulator, tSpace, aDepth - 1);
    }

    return;
}
void KGInterface::RetrieveSpacesByWildcard(vector<KGSpace*>& anAccumulator, KGSpace* aNode, int aDepth)
{
    if (aDepth == 0) {
        return;
    }

    KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = aNode->GetChildSpaces()->begin(); tSpaceIt != aNode->GetChildSpaces()->end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        anAccumulator.push_back(tSpace);
        RetrieveSpacesByWildcard(anAccumulator, tSpace, aDepth - 1);
    }

    return;
}

KGSpace* KGInterface::Root() const
{
    return fRoot;
}
}  // namespace KGeoBag
