#include "KSortedSurfaceContainer.hh"

#include <algorithm>

namespace KEMField
{
KSortedSurfaceContainer::KSortedSurfaceContainer(const KSurfaceContainer& container) : fSurfaceContainer(container)
{

    //temporary vectors to store look up table
    std::vector<std::vector<unsigned int>*> sorted_surface_list;

    // We sort the surfaces by unique boundary description
    unsigned int index = 0;
    for (KSurfaceContainer::iterator it = fSurfaceContainer.begin(); it != fSurfaceContainer.end(); it++) {
        bool isAssigned = false;

        if (!fSortedSurfaces.empty()) {
            int indexLevel = 0;
            for (auto dataIt = fSortedSurfaces.begin(); dataIt != fSortedSurfaces.end(); ++dataIt) {
                KSurfacePrimitive* sP = (*dataIt)->at(0);
                if (*((*it)->GetBoundary()) == *(sP->GetBoundary())) {
                    fSortedSurfaces.at(indexLevel)->push_back(*it);
                    sorted_surface_list[indexLevel]->push_back(index);
                    isAssigned = true;
                    break;
                }
                indexLevel++;
            }
        }

        if (!isAssigned) {
            fSortedSurfaces.push_back(new KSurfaceContainer::KSurfaceArray(1, *it));
            sorted_surface_list.push_back(new std::vector<unsigned int>(1, index));
        }

        index++;
    }


    //now we construct the index look-up maps
    fNormalToSortedIndexMap.clear();
    fSortedToNormalIndexMap.clear();

    for (auto& i : sorted_surface_list) {
        for (unsigned int& j : *i) {
            fSortedToNormalIndexMap.push_back(j);
        }
    }

    unsigned int map_size = fSortedToNormalIndexMap.size();
    fNormalToSortedIndexMap.resize(map_size);

    for (unsigned int i = 0; i < map_size; i++) {
        unsigned int normal_index = fSortedToNormalIndexMap[i];
        fNormalToSortedIndexMap[normal_index] = i;
    }

    //clean up
    for (auto& surface : sorted_surface_list) {
        delete surface;
    }
}

KSortedSurfaceContainer::~KSortedSurfaceContainer()
{
    for (auto& surface : fSortedSurfaces) {
        delete surface;
    }
}

bool KSortedSurfaceContainer::BoundaryType(unsigned int i) const
{
    return fSortedSurfaces.at(i)->at(0)->GetID().BoundaryID;
}

unsigned int KSortedSurfaceContainer::IndexOfFirstSurface(unsigned int i) const
{
    unsigned int j = 0;
    for (unsigned int ii = 0; ii < i; ii++)
        j += fSortedSurfaces.at(ii)->size();
    return j;
}
}  // namespace KEMField
