#include "KSGeoSide.h"

#include "KSGeoSpace.h"
#include "KSGeometryMessage.h"

#include <limits>

using namespace std;
using KGeoBag::KGSurface;
using katrin::KThreeVector;

namespace Kassiopeia
{

KSGeoSide::KSGeoSide() : fOutsideParent(nullptr), fInsideParent(nullptr), fContents() {}
KSGeoSide::KSGeoSide(const KSGeoSide& aCopy) :
    KSComponent(aCopy),
    fOutsideParent(nullptr),
    fInsideParent(nullptr),
    fContents(aCopy.fContents)
{}
KSGeoSide* KSGeoSide::Clone() const
{
    return new KSGeoSide(*this);
}
KSGeoSide::~KSGeoSide() = default;

void KSGeoSide::On() const
{
    geomsg_debug("on geo side <" << this->GetName() << ">" << eom);
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->Activate();
    }
    return;
}
void KSGeoSide::Off() const
{
    geomsg_debug("off geo side <" << this->GetName() << ">" << eom);
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->Deactivate();
    }
    return;
}

KThreeVector KSGeoSide::Point(const KThreeVector& aPoint) const
{
    double tDistance;
    KThreeVector tPoint;
    vector<KGSurface*>::const_iterator tSide;

    double tNearestDistance = std::numeric_limits<double>::max();
    KThreeVector tNearestPoint;

    for (tSide = fContents.begin(); tSide != fContents.end(); tSide++) {
        tPoint = (*tSide)->Point(aPoint);
        tDistance = (tPoint - aPoint).Magnitude();
        if (tDistance < tNearestDistance) {
            tNearestDistance = tDistance;
            tNearestPoint = tPoint;
        }
    }

    return tNearestPoint;
}
KThreeVector KSGeoSide::Normal(const KThreeVector& aPoint) const
{
    double tDistance;
    KThreeVector tPoint;
    vector<KGSurface*>::const_iterator tSide;

    double tNearestDistance = std::numeric_limits<double>::max();
    const KGSurface* tNearestSide = nullptr;

    for (tSide = fContents.begin(); tSide != fContents.end(); tSide++) {
        tPoint = (*tSide)->Point(aPoint);
        tDistance = (tPoint - aPoint).Magnitude();
        if (tDistance < tNearestDistance) {
            tNearestDistance = tDistance;
            tNearestSide = (*tSide);
        }
    }

    if (tNearestSide != nullptr)
        return tNearestSide->Normal(aPoint);

    geomsg(eWarning) << "geo side <" << GetName() << "> could not find a nearest space to position " << aPoint << eom;
    return KThreeVector::sInvalid;
}

void KSGeoSide::AddContent(KGSurface* aSurface)
{
    vector<KGSurface*>::iterator tSurface;
    for (tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++) {
        if ((*tSurface) == aSurface) {
            geomsg(eWarning) << "surface <" << aSurface->GetName() << "> was already added to geo side <"
                             << this->GetName() << ">" << eom;
            return;
        }
    }
    fContents.push_back(aSurface);
    return;
}
void KSGeoSide::RemoveContent(KGSurface* aSurface)
{
    vector<KGSurface*>::iterator tSurface;
    for (tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++) {
        if ((*tSurface) == aSurface) {
            fContents.erase(tSurface);
            return;
        }
    }
    geomsg(eWarning) << "can not remove surface <" << aSurface->GetName() << ">, as it was not found in geo side <"
                     << this->GetName() << ">" << eom;
    return;
}

vector<KGSurface*> KSGeoSide::GetContent()
{
    return fContents;
}

void KSGeoSide::AddCommand(KSCommand* anCommand)
{
    vector<KSCommand*>::iterator tCommand;
    for (tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++) {
        if ((*tCommand) == anCommand) {
            geomsg(eWarning) << "command <" << anCommand->GetName() << "> was already added to geo side <"
                             << this->GetName() << ">" << eom;
            return;
        }
    }
    fCommands.push_back(anCommand);
    return;
}
void KSGeoSide::RemoveCommand(KSCommand* anCommand)
{
    vector<KSCommand*>::iterator tCommand;
    for (tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++) {
        if ((*tCommand) == anCommand) {
            fCommands.erase(tCommand);
            return;
        }
    }
    geomsg(eWarning) << "can not remove command <" << anCommand->GetName() << ">, as it was not found in geo side <"
                     << this->GetName() << ">" << eom;
    return;
}

void KSGeoSide::InitializeComponent()
{
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->GetParent()->Initialize();
        (*tCommandIt)->GetChild()->Initialize();
    }

    return;
}
void KSGeoSide::DeinitializeComponent()
{
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->GetParent()->Deinitialize();
        (*tCommandIt)->GetChild()->Deinitialize();
    }

    return;
}

}  // namespace Kassiopeia
