#ifndef KSNavMeshedSpace_HH__
#define KSNavMeshedSpace_HH__

#include "KGMesh.hh"
#include "KGMeshElementCollector.hh"
#include "KGMesher.hh"
#include "KGNavigableMeshFirstIntersectionFinder.hh"
#include "KGNavigableMeshProximityCheck.hh"
#include "KGNavigableMeshTree.hh"
#include "KGNavigableMeshTreeBuilder.hh"
#include "KSGeoSide.h"
#include "KSGeoSpace.h"
#include "KSGeoSurface.h"
#include "KSNavigatorsMessage.h"
#include "KSSpaceNavigator.h"

#include <utility>
#include <vector>


using namespace KGeoBag;
using namespace katrin;

#define KSNAVMESHEDSPACE_SPACE   0
#define KSNAVMESHEDSPACE_SIDE    1
#define KSNAVMESHEDSPACE_SURFACE 2

namespace Kassiopeia
{

/*
    *
    *@file KSNavMeshedSpace.hh
    *@class KSNavMeshedSpace
    *@brief
    * alternate navigator based off of Dan's original class, which navigates on
    * on the mesh elements constructed from the geometry elements using an octree
    *@details
    *
    *<b>Revision History:<b>
    *Date Name Brief Description
    *Thu, 23 Jul 2015 EST 12:17:17  J. Barrett (barrettj@mit.edu) First Version
    *
    */

class KSNavMeshedSpace : public KSComponentTemplate<KSNavMeshedSpace, KSSpaceNavigator>
{
  public:
    KSNavMeshedSpace();
    KSNavMeshedSpace(const KSNavMeshedSpace& aCopy);
    KSNavMeshedSpace* Clone() const override;
    ~KSNavMeshedSpace() override;

    void InitializeComponent() override
    {
        CollectMeshElements();
        ConstructTree();
        fFirstIntersectionFinder.SetMeshElementContainer(&fElementContainer);
        fProximityChecker.SetMeshElementContainer(&fElementContainer);
    }

    void SetEnterSplit(const bool& anEnterSplit)
    {
        fEnterSplit = anEnterSplit;
    };
    bool GetEnterSplit() const
    {
        return fEnterSplit;
    };

    void SetExitSplit(const bool& anExitSplit)
    {
        fExitSplit = anExitSplit;
    };
    bool GetExitSplit() const
    {
        return fExitSplit;
    };

    void SetFailCheck(const bool& aValue)
    {
        fFailCheck = aValue;
    };
    bool GetFailCheck() const
    {
        return fFailCheck;
    };

    void SetRootSpace(KSSpace* root_space)
    {
        fRootSpace = root_space;
    };
    KSSpace* GetRootSpace() const
    {
        return fRootSpace;
    };

    void SetMaximumOctreeDepth(unsigned int d)
    {
        fMaxDepth = d;
        fSpecifyMaxDepth = true;
    };
    unsigned int GetMaximumOctreeDepth() const
    {
        return fMaxDepth;
    };

    void SetSpatialResolution(double r)
    {
        fSpatialResolution = r;
        fSpecifyResolution = true;
    };
    double GetSpatialResolution() const
    {
        return fSpatialResolution;
    };

    void SetNumberOfAllowedElements(unsigned int n)
    {
        fNAllowedElements = n;
        fSpecifyAllowedElements = true;
    };
    unsigned int GetNumberOfAllowedElements() const
    {
        return fNAllowedElements;
    };

    void SetAbsoluteTolerance(double abs_tol)
    {
        fAbsoluteTolerance = abs_tol;
        fUserSpecifiedAbsoluteTolerance = true;
    };
    double GetAbsoluteTolerance() const
    {
        return fAbsoluteTolerance;
    };

    void SetRelativeTolerance(double rel_tol)
    {
        fRelativeTolerance = rel_tol;
    };
    double GetRelativeTolerance() const
    {
        return fRelativeTolerance;
    };

    void SetFileName(std::string filename)
    {
        fFileName = filename;
    };
    std::string GetFileName() const
    {
        return fFileName;
    };

    void SetPath(std::string path)
    {
        fPath = path;
    };
    std::string GetPath() const
    {
        return fPath;
    };

  public:
    void CalculateNavigation(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                             const KSParticle& aTrajectoryFinalParticle, const KThreeVector& aTrajectoryCenter,
                             const double& aTrajectoryRadius, const double& aTrajectoryStep,
                             KSParticle& aNavigationParticle, double& aNavigationStep, bool& aNavigationFlag) override;
    void ExecuteNavigation(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                           KSParticleQueue& aSecondaries) const override;
    void FinalizeNavigation(KSParticle& aFinalParticle) const override;
    void StartNavigation(KSParticle& aParticle, KSSpace* aRoot) override;
    void StopNavigation(KSParticle& aParticle, KSSpace* aRoot) override;

  private:
    void CollectMeshElements();
    void ConstructTree();
    bool RetrieveTree();
    void SaveTree();

    std::string fFileName;
    std::string fPath;
    std::vector<std::string> fLabels;

    //solve quadratic for intersection time
    double SolveForTime(double distance, double t, double v1, double v2) const;

    //navigation parameters
    bool fExitSplit;
    bool fEnterSplit;
    bool fFailCheck;
    double fRelativeTolerance;
    double fAbsoluteTolerance;
    bool fUserSpecifiedAbsoluteTolerance;

    //pointer to the root space so we can traverse the geometry tree
    //in order to build the mesh, octree, and element look up map
    KSSpace* fRootSpace;

    //octree parameters
    unsigned int fMaxDepth;
    bool fSpecifyMaxDepth;

    double fSpatialResolution;
    bool fSpecifyResolution;

    unsigned int fNAllowedElements;
    bool fSpecifyAllowedElements;

    //container to hold all of the mesh elements (with global coordinates)
    KGNavigableMeshElementContainer fElementContainer;

    //data stuct so we can associate each mesh element with its parent space/side/surface
    //for navigation actions
    struct MeshElementAssociation
    {
        unsigned int fElementID;
        unsigned int fType;  //parent entity type (space, side, surface)
        KSGeoSpace* fSpace;
        KSGeoSide* fSide;
        KSGeoSurface* fSurface;
    };

    //vector to store the mesh to parent entity map
    std::vector<MeshElementAssociation> fElementMap;

    //sorting struct so the elements are inserted in their proper place
    struct MeshElementAssociationSortingPredicate
    {
        bool operator()(const MeshElementAssociation& a, const MeshElementAssociation& b)
        {
            return a.fElementID < b.fElementID;
        }
    };
    //            MeshElementAssociationSortingPredicate fSortingPred;

    //private class to do the collection with a specific post collection action
    //this helps us construct the element -> parent map
    class KSMeshElementCollector : public KGMeshElementCollector
    {
      public:
        KSMeshElementCollector() : fType(0), fSpace(nullptr), fSide(nullptr), fSurface(nullptr), fMap(nullptr) {}
        ~KSMeshElementCollector() override {}

        void SetSpace(KSGeoSpace* space)
        {
            fType = KSNAVMESHEDSPACE_SPACE;
            fSpace = space;
            fSide = nullptr;
            fSurface = nullptr;
        };

        void SetSide(KSGeoSide* side)
        {
            fType = KSNAVMESHEDSPACE_SIDE;
            fSpace = nullptr;
            fSide = side;
            fSurface = nullptr;
        };

        void SetSurface(KSGeoSurface* surface)
        {
            fType = KSNAVMESHEDSPACE_SURFACE;
            fSpace = nullptr;
            fSide = nullptr;
            fSurface = surface;
        };

        void SetElementMap(std::vector<MeshElementAssociation>* element_map)
        {
            fMap = element_map;
        };

        void PreCollectionActionExecute(KGMeshData* aData) override
        {
            unsigned int n_elem = aData->Elements()->size();
            double sum_area = 0.0;
            for (auto tElementIt = aData->Elements()->begin(); tElementIt != aData->Elements()->end(); tElementIt++) {
                sum_area += (*tElementIt)->Area();
            }

            if (n_elem == 1) {
                if (fType == KSNAVMESHEDSPACE_SPACE) {
                    navmsg(eWarning)
                        << " space <" << fSpace->GetName()
                        << "> has only 1 mesh element, please increase the discretization parameters for this space if you wish to use the meshed space navigator."
                        << eom;
                    if (sum_area == 0.0) {
                        navmsg(eWarning) << " navigator will ignore the mesh elements of space <" << fSpace->GetName()
                                         << "> which have a cumulative area of zero!" << eom;
                    }
                }

                if (fType == KSNAVMESHEDSPACE_SIDE) {
                    navmsg(eWarning)
                        << " side <" << fSide->GetName()
                        << "> has only 1 mesh element, please increase the discretization parameters for this side if you wish to use the meshed space navigator."
                        << eom;
                    if (sum_area == 0.0) {
                        navmsg(eWarning) << " navigator will ignore the mesh elements of side <" << fSide->GetName()
                                         << "> which have a cumulative area of zero!" << eom;
                    }
                }

                if (fType == KSNAVMESHEDSPACE_SURFACE) {
                    navmsg(eWarning)
                        << " surface <" << fSurface->GetName()
                        << "> has only 1 mesh element, please increase the discretization parameters for this surface if you wish to use the meshed space navigator."
                        << eom;
                    if (sum_area == 0.0) {
                        navmsg(eWarning) << " navigator will ignore the mesh elements of surface <"
                                         << fSurface->GetName() << "> which have a cumulative area of zero!" << eom;
                    }
                }
            }
        }

        void PostCollectionActionExecute(KGNavigableMeshElement* /*element */) override
        {
            MeshElementAssociation temp;
            temp.fElementID = fMap->size();
            temp.fType = fType;
            temp.fSpace = fSpace;
            temp.fSide = fSide;
            temp.fSurface = fSurface;

            //insert the element into the map
            fMap->push_back(temp);
        }

      private:
        unsigned int fType;
        KSGeoSpace* fSpace;
        KSGeoSide* fSide;
        KSGeoSurface* fSurface;
        std::vector<MeshElementAssociation>* fMap;
    };

    //collector for the mesh elements
    KSMeshElementCollector fCollector;

    //vectors of geometry entities
    std::vector<KSGeoSpace*> fSpaces;
    std::vector<KSGeoSide*> fSides;
    std::vector<KSGeoSurface*> fSurfaces;

    //the octree and its builder and parameters
    KGNavigableMeshTree fTree;
    KGNavigableMeshTreeBuilder fTreeBuilder;
    KGNavigableMeshFirstIntersectionFinder fFirstIntersectionFinder;
    KGNavigableMeshProximityCheck fProximityChecker;
    KGCube<KGMESH_DIM>* fWorldCube;

    //temp entity the mesh element belongs to
    mutable unsigned int fMeshElementID;
    mutable KSSpace* fSpaceEntity;
    mutable KSSide* fSideEntity;
    mutable KSSurface* fSurfaceEntity;

    //used to determine the time of intersection and exit/entry of spaces
    mutable std::vector<KSParticle> fIntermediateParticleStates;
    mutable bool fIsEntry;

    //long count of navigation actions
    mutable long fNavigationCount;

    //persistent state to avoid repeated intersection in the same
    //place (due to numerical error) on subsequent navigation calls
    mutable unsigned int fLastEntityType;
    mutable unsigned int fLastMeshElementID;
    mutable KSSpace* fLastSpaceEntity;
    mutable KSSide* fLastSideEntity;
    mutable KSSurface* fLastSurfaceEntity;
    mutable KThreeVector fLastIntersection;
    mutable KThreeVector fLastDirection;
    mutable long fLastSpaceCount;
    mutable long fLastSideCount;
    mutable long fLastSurfaceCount;
    mutable bool fNavigationFail;
};

}  // namespace Kassiopeia


#endif /* end of include guard: KSNavMeshedSpace_H__ */
