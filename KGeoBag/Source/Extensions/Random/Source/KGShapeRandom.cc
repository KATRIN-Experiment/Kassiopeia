#include "KGShapeRandom.hh"

#include "KGRandomMessage.hh"
#include "KRandom.h"

#include <limits>

namespace KGeoBag
{
KThreeVector KGShapeRandom::Random(KGSurface* surface)
{
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();
    surface->AcceptNode(this);

    if (!fRandom.IsValid()) {
        randommsg(eError) << "no random generator implemented for surface <" << surface->GetName() << ">" << eom;
    }

    //fRandom = surface->GetOrigin() + fRandom;
    fRandom = surface->GetOrigin() + fRandom[0] * surface->GetXAxis() + fRandom[1] * surface->GetYAxis() +
              fRandom[2] * surface->GetZAxis();

    return fRandom;
}

KThreeVector KGShapeRandom::Random(KGSpace* space)
{
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();
    space->AcceptNode(this);

    if (!fRandom.IsValid()) {
        randommsg(eError) << "no random generator implemented for space <" << space->GetName() << ">" << eom;
    }

    //fRandom = space->GetOrigin() + fRandom;
    fRandom = space->GetOrigin() + fRandom[0] * space->GetXAxis() + fRandom[1] * space->GetYAxis() +
              fRandom[2] * space->GetZAxis();

    return fRandom;
}

KThreeVector KGShapeRandom::Random(std::vector<KGSurface*>& surfaces)
{
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();

    if (surfaces.empty()) {
        return fRandom;
    }

    double totalArea = 0;
    for (auto* surface : surfaces) {
        if (!surface->HasExtension<KGMetrics>()) {
            surface->MakeExtension<KGMetrics>();
        }

        totalArea += surface->AsExtension<KGMetrics>()->GetArea();
    }

    if (fabs(totalArea) < std::numeric_limits<double>::min()) {
        randommsg(eError) << "total surface area for random generator is zero" << eom;
    }

    KGSurface* selectedSurface = nullptr;
    double decision = Uniform(0, totalArea);
    for (auto* surface : surfaces) {
        selectedSurface = surface;

        totalArea -= selectedSurface->AsExtension<KGMetrics>()->GetArea();

        if (decision > totalArea) {
            break;
        }
    }

    return Random(selectedSurface);
}

KThreeVector KGShapeRandom::Random(std::vector<KGSpace*>& spaces)
{
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();

    if (spaces.empty()) {
        return fRandom;
    }

    double totalVolume = 0;
    for (auto* space : spaces) {
        if (!space->HasExtension<KGMetrics>()) {
            space->MakeExtension<KGMetrics>();
        }

        totalVolume += space->AsExtension<KGMetrics>()->GetVolume();
    }

    if (fabs(totalVolume) < std::numeric_limits<double>::min()) {
        randommsg(eError) << "total space volume for random generator is zero" << eom;
    }

    KGSpace* selectedSpace = nullptr;
    double decision = Uniform(0, totalVolume);
    for (auto* space : spaces) {
        selectedSpace = space;

        totalVolume -= selectedSpace->AsExtension<KGMetrics>()->GetVolume();

        if (decision > totalVolume) {
            break;
        }
    }

    return Random(selectedSpace);
}

double KGShapeRandom::Uniform(double min, double max)
{
    return katrin::KRandom::GetInstance().Uniform(min, max);
}
}  // namespace KGeoBag
