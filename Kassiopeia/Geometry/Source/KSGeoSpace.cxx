#include "KSGeoSpace.h"

#include "KSGeoSide.h"
#include "KSGeoSurface.h"
#include "KSGeometryMessage.h"

#include <limits>

using namespace std;

namespace Kassiopeia
{

KSGeoSpace::KSGeoSpace() : fContents(), fCommands() {}
KSGeoSpace::KSGeoSpace(const KSGeoSpace& aCopy) : KSComponent(), fContents(aCopy.fContents), fCommands(aCopy.fCommands)
{}
KSGeoSpace* KSGeoSpace::Clone() const
{
    return new KSGeoSpace(*this);
}
KSGeoSpace::~KSGeoSpace() {}

void KSGeoSpace::Enter() const
{
    geomsg_debug("enter geo space <" << this->GetName() << ">" << eom) vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->Activate();
    }
    return;
}
void KSGeoSpace::Exit() const
{
    geomsg_debug("exit geo space <" << this->GetName() << ">" << eom) vector<KSCommand*>::reverse_iterator tCommandIt;
    for (tCommandIt = fCommands.rbegin(); tCommandIt != fCommands.rend(); tCommandIt++) {
        (*tCommandIt)->Deactivate();
    }
    return;
}

bool KSGeoSpace::Outside(const KThreeVector& aPoint) const
{
    bool tOutside;
    vector<KGSpace*>::const_iterator tSpace;

    for (tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++) {
        tOutside = (*tSpace)->Outside(aPoint);
        if (tOutside == true) {
            return true;
        }
    }

    return false;
}

KThreeVector KSGeoSpace::Point(const KThreeVector& aPoint) const
{
    double tDistance;
    KThreeVector tPoint;
    vector<KGSpace*>::const_iterator tSpace;

    double tNearestDistance = std::numeric_limits<double>::max();
    KThreeVector tNearestPoint;

    for (tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++) {
        tPoint = (*tSpace)->Point(aPoint);
        tDistance = (tPoint - aPoint).Magnitude();
        if (tDistance < tNearestDistance) {
            tNearestDistance = tDistance;
            tNearestPoint = tPoint;
        }
    }

    return tNearestPoint;
}
KThreeVector KSGeoSpace::Normal(const KThreeVector& aPoint) const
{
    double tDistance;
    KThreeVector tPoint;
    vector<KGSpace*>::const_iterator tSpace;

    double tNearestDistance = std::numeric_limits<double>::max();
    const KGSpace* tNearestSpace = nullptr;

    for (tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++) {
        tPoint = (*tSpace)->Point(aPoint);
        tDistance = (tPoint - aPoint).Magnitude();
        if (tDistance < tNearestDistance) {
            tNearestDistance = tDistance;
            tNearestSpace = (*tSpace);
        }
    }

    if (tNearestSpace != nullptr)
        return tNearestSpace->Normal(aPoint);

    geomsg(eWarning) << "geo space <" << GetName() << "> could not find a nearest space to position " << aPoint << eom;
    return KThreeVector::sInvalid;
}

void KSGeoSpace::AddContent(KGSpace* aSpace)
{
    vector<KGSpace*>::iterator tSpace;
    for (tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++) {
        if ((*tSpace) == aSpace) {
            geomsg(eWarning) << "space <" << aSpace->GetName() << "> was already added to geo space <"
                             << this->GetName() << ">" << eom;
            return;
        }
    }
    fContents.push_back(aSpace);
    return;
}
void KSGeoSpace::RemoveContent(KGSpace* aSpace)
{
    vector<KGSpace*>::iterator tSpace;
    for (tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++) {
        if ((*tSpace) == aSpace) {
            fContents.erase(tSpace);
            return;
        }
    }
    geomsg(eWarning) << "can not remove space <" << aSpace->GetName() << ">, as it was not found in geo space <"
                     << this->GetName() << ">" << eom;
    return;
}
vector<KGSpace*> KSGeoSpace::GetContent()
{
    return fContents;
}

void KSGeoSpace::AddCommand(KSCommand* anCommand)
{
    vector<KSCommand*>::iterator tCommand;
    for (tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++) {
        if ((*tCommand) == anCommand) {
            geomsg(eWarning) << "command <" << anCommand->GetName() << "> was already added to geo space <"
                             << this->GetName() << ">" << eom;
            return;
        }
    }
    fCommands.push_back(anCommand);
    return;
}
void KSGeoSpace::RemoveCommand(KSCommand* anCommand)
{
    vector<KSCommand*>::iterator tCommand;
    for (tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++) {
        if ((*tCommand) == anCommand) {
            fCommands.erase(tCommand);
            return;
        }
    }
    geomsg(eWarning) << "can not remove command <" << anCommand->GetName() << ">, as it was not found in geo space <"
                     << this->GetName() << ">" << eom;
    return;
}

void KSGeoSpace::InitializeComponent()
{
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->GetParent()->Initialize();
        (*tCommandIt)->GetChild()->Initialize();
    }

    vector<KSSpace*>::iterator tKSSpaceIt;
    for (tKSSpaceIt = fSpaces.begin(); tKSSpaceIt != fSpaces.end(); tKSSpaceIt++) {
        (*tKSSpaceIt)->Initialize();
    }
    vector<KSSurface*>::iterator tKSSurfaceIt;
    for (tKSSurfaceIt = fSurfaces.begin(); tKSSurfaceIt != fSurfaces.end(); tKSSurfaceIt++) {
        (*tKSSurfaceIt)->Initialize();
    }
    vector<KSSide*>::iterator tKSSideIt;
    for (tKSSideIt = fSides.begin(); tKSSideIt != fSides.end(); tKSSideIt++) {
        (*tKSSideIt)->Initialize();

        //check if navigation sides are really geometric boundaries of volumes
        auto* tGeoSide = dynamic_cast<KSGeoSide*>((*tKSSideIt));
        if (tGeoSide != nullptr) {
            vector<KGSurface*> tSideSurfaces = tGeoSide->GetContent();

            for (vector<KGSurface*>::const_iterator tSideSurface = tSideSurfaces.begin();
                 tSideSurface != tSideSurfaces.end();
                 tSideSurface++) {

                for (vector<KGSpace*>::const_iterator tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++) {
                    bool tValid = false;

                    const vector<KGSurface*>* tSurfaces = (*tSpace)->GetBoundaries();

                    for (auto tSurface = tSurfaces->begin(); tSurface != tSurfaces->end(); tSurface++) {
                        if (*(tSideSurface) == *(tSurface)) {
                            tValid = true;
                        }
                    }

                    if (tValid == false) {
                        geomsg(eError) << "geo_side <" << (*tSideSurface)->GetName()
                                       << "> is not a geometic boundary of space <" << (*tSpace)->GetName()
                                       << ">, check your navigation configuration" << eom;
                    }
                }
            }
        }
    }

    return;
}
void KSGeoSpace::DeinitializeComponent()
{
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->GetParent()->Deinitialize();
        (*tCommandIt)->GetChild()->Deinitialize();
    }

    vector<KSSpace*>::iterator tGeoSpaceIt;
    for (tGeoSpaceIt = fSpaces.begin(); tGeoSpaceIt != fSpaces.end(); tGeoSpaceIt++) {
        (*tGeoSpaceIt)->Deinitialize();
    }
    vector<KSSurface*>::iterator tGeoSurfaceIt;
    for (tGeoSurfaceIt = fSurfaces.begin(); tGeoSurfaceIt != fSurfaces.end(); tGeoSurfaceIt++) {
        (*tGeoSurfaceIt)->Deinitialize();
    }
    vector<KSSide*>::iterator tGeoSideIt;
    for (tGeoSideIt = fSides.begin(); tGeoSideIt != fSides.end(); tGeoSideIt++) {
        (*tGeoSideIt)->Deinitialize();
    }

    return;
}

}  // namespace Kassiopeia
