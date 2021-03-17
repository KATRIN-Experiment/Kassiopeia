#include "KSGeoSurface.h"

#include "KSGeoSpace.h"
#include "KSGeometryMessage.h"

#include <limits>

using namespace std;
using KGeoBag::KGSurface;
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSGeoSurface::KSGeoSurface() : fParent(nullptr), fContents() {}
KSGeoSurface::KSGeoSurface(const KSGeoSurface& aCopy) : KSComponent(aCopy), fParent(nullptr), fContents(aCopy.fContents)
{}
KSGeoSurface* KSGeoSurface::Clone() const
{
    return new KSGeoSurface(*this);
}
KSGeoSurface::~KSGeoSurface() = default;

void KSGeoSurface::On() const
{
    geomsg_debug("on geo surface <" << this->GetName() << ">" << eom) vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->Activate();
    }
    return;
}
void KSGeoSurface::Off() const
{
    geomsg_debug("off geo surface <" << this->GetName() << ">" << eom) vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->Deactivate();
    }
    return;
}

KThreeVector KSGeoSurface::Point(const KThreeVector& aPoint) const
{
    double tDistance;
    KThreeVector tPoint;
    std::vector<KGSurface*>::const_iterator tSurface;

    double tNearestDistance = std::numeric_limits<double>::max();
    KThreeVector tNearestPoint;

    for (tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++) {
        tPoint = (*tSurface)->Point(aPoint);
        tDistance = (tPoint - aPoint).Magnitude();
        if (tDistance < tNearestDistance) {
            tNearestDistance = tDistance;
            tNearestPoint = tPoint;
        }
    }

    return tNearestPoint;
}
KThreeVector KSGeoSurface::Normal(const KThreeVector& aPoint) const
{
    double tDistance;
    KThreeVector tPoint;
    vector<KGSurface*>::const_iterator tSurface;

    double tNearestDistance = std::numeric_limits<double>::max();
    const KGSurface* tNearestSurface = nullptr;

    for (tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++) {
        tPoint = (*tSurface)->Point(aPoint);
        tDistance = (tPoint - aPoint).Magnitude();
        if (tDistance < tNearestDistance) {
            tNearestDistance = tDistance;
            tNearestSurface = (*tSurface);
        }
    }

    if (tNearestSurface != nullptr)
        return tNearestSurface->Normal(aPoint);

    geomsg(eWarning) << "geo surface <" << GetName() << "> could not find a nearest space to position " << aPoint
                     << eom;
    return KThreeVector::sInvalid;
}

void KSGeoSurface::AddContent(KGSurface* aSurface)
{
    vector<KGSurface*>::iterator tSurface;
    for (tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++) {
        if ((*tSurface) == aSurface) {
            geomsg(eWarning) << "surface <" << aSurface->GetName() << "> was already added to geo surface <"
                             << this->GetName() << ">" << eom;
            return;
        }
    }
    fContents.push_back(aSurface);
    return;
}
void KSGeoSurface::RemoveContent(KGSurface* aSurface)
{
    vector<KGSurface*>::iterator tSurface;
    for (tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++) {
        if ((*tSurface) == aSurface) {
            fContents.erase(tSurface);
            return;
        }
    }
    geomsg(eWarning) << "can not remove surface <" << aSurface->GetName() << ">, as it was not found in geo surface <"
                     << this->GetName() << ">" << eom;
    return;
}

vector<KGSurface*> KSGeoSurface::GetContent()
{
    return fContents;
}

void KSGeoSurface::AddCommand(KSCommand* anCommand)
{
    vector<KSCommand*>::iterator tCommand;
    for (tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++) {
        if ((*tCommand) == anCommand) {
            geomsg(eWarning) << "command <" << anCommand->GetName() << "> was already added to geo surface <"
                             << this->GetName() << ">" << eom;
            return;
        }
    }
    fCommands.push_back(anCommand);
    return;
}
void KSGeoSurface::RemoveCommand(KSCommand* anCommand)
{
    vector<KSCommand*>::iterator tCommand;
    for (tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++) {
        if ((*tCommand) == anCommand) {
            fCommands.erase(tCommand);
            return;
        }
    }
    geomsg(eWarning) << "can not remove command <" << anCommand->GetName() << ">, as it was not found in geo surface <"
                     << this->GetName() << ">" << eom;
    return;
}

void KSGeoSurface::InitializeComponent()
{
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->GetParent()->Initialize();
        (*tCommandIt)->GetChild()->Initialize();
    }

    return;
}
void KSGeoSurface::DeinitializeComponent()
{
    vector<KSCommand*>::iterator tCommandIt;
    for (tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++) {
        (*tCommandIt)->GetParent()->Deinitialize();
        (*tCommandIt)->GetChild()->Deinitialize();
    }

    return;
}

}  // namespace Kassiopeia
