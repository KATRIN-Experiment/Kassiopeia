#include "KSNavMeshedSpace.h"

#include "KEMFileInterface.hh"
#include "KFile.h"
#include "KGObjectCollector.hh"
#include "KGTreeStructureExtractor.hh"
#include "KMD5HashGenerator.hh"
#include "KSNavOctreeData.h"

#include <cstdlib>
#include <ctime>
#include <limits>
#include <queue>
#include <stack>

using std::numeric_limits;
using namespace KEMField;
using namespace KGeoBag;

namespace Kassiopeia
{

KSNavMeshedSpace::KSNavMeshedSpace() :
    fFileName(""),
    fExitSplit(false),
    fEnterSplit(false),
    fFailCheck(false),
    fRelativeTolerance(1e-2),
    fAbsoluteTolerance(1e-3),
    fRootSpace(nullptr),
    fMaxDepth(8),
    fSpecifyMaxDepth(false),
    fSpatialResolution(1e-6),
    fSpecifyResolution(false),
    fNAllowedElements(1),
    fSpecifyAllowedElements(false)
{}

KSNavMeshedSpace::KSNavMeshedSpace(const KSNavMeshedSpace& aCopy) :
    KSComponent(aCopy),
    fFileName(aCopy.fFileName),
    fExitSplit(aCopy.fExitSplit),
    fEnterSplit(aCopy.fEnterSplit),
    fFailCheck(aCopy.fFailCheck),
    fRelativeTolerance(aCopy.fRelativeTolerance),
    fAbsoluteTolerance(aCopy.fAbsoluteTolerance),
    fRootSpace(aCopy.fRootSpace),
    fMaxDepth(aCopy.fMaxDepth),
    fSpecifyMaxDepth(aCopy.fSpecifyMaxDepth),
    fSpatialResolution(aCopy.fSpatialResolution),
    fSpecifyResolution(aCopy.fSpecifyResolution),
    fNAllowedElements(aCopy.fNAllowedElements),
    fSpecifyAllowedElements(aCopy.fSpecifyAllowedElements)
{
    //this would be is slow if we copy the navigator a lot
    InitializeComponent();
}

KSNavMeshedSpace* KSNavMeshedSpace::Clone() const
{
    return new KSNavMeshedSpace(*this);
}

KSNavMeshedSpace::~KSNavMeshedSpace() = default;

void KSNavMeshedSpace::CollectMeshElements()
{
    //temp space
    KSSpace* tSpace;
    fSpaces.clear();
    fSides.clear();
    fSurfaces.clear();

    //starting with the root space, we need to retrieve all of the surfaces, spaces and sides it contains
    //recursively iterate over the objects in the root space
    auto* spaceQueue = new std::queue<KSSpace*>();
    spaceQueue->push(fRootSpace);

    if (fRootSpace == nullptr) {
        navmsg(eError) << "please check your configuration, the root space has not been specified!" << eom;
    }

    while (!(spaceQueue->empty())) {
        tSpace = spaceQueue->front();

        //collect this space if it is a geo space
        auto* geo_space = dynamic_cast<KSGeoSpace*>(tSpace);
        if (geo_space != nullptr) {
            navmsg_debug(" navigator <" << GetName() << "> is collecting the space <" << geo_space->GetName()
                                        << "> for meshing " << eom);
            fSpaces.push_back(geo_space);
        };

        //collect the surfaces that are children of this space and are geo surfaces
        for (int i = 0; i < tSpace->GetSurfaceCount(); i++) {
            auto* geo_surface = dynamic_cast<KSGeoSurface*>(tSpace->GetSurface(i));
            if (geo_surface != nullptr) {
                navmsg_debug(" navigator <" << GetName() << "> is collecting the surface <" << geo_surface->GetName()
                                            << "> for meshing " << eom);
                fSurfaces.push_back(geo_surface);
            }
        }

        //collect the sides of this space that are geo sides
        for (int i = 0; i < tSpace->GetSideCount(); i++) {
            auto* geo_side = dynamic_cast<KSGeoSide*>(tSpace->GetSide(i));
            if (geo_side != nullptr) {
                navmsg_debug(" navigator <" << GetName() << "> is collecting the side <" << geo_side->GetName()
                                            << "> for meshing " << eom);
                fSides.push_back(geo_side);
            }
        }

        //add this spaces children to the queue
        if (tSpace->GetSpaceCount() != 0) {
            for (int i = 0; i < tSpace->GetSpaceCount(); i++) {
                spaceQueue->push(tSpace->GetSpace(i));
            }
        }
        //pop this space off the queue
        spaceQueue->pop();
    };
    delete spaceQueue;

    //now we discretize the KGeoBag objects
    //construct the mesh element to parent entity map
    //and collect the mesh elements for the octree container
    fElementMap.clear();
    KGMesher tMesher;
    fCollector.SetElementMap(&fElementMap);
    fCollector.SetMeshElementContainer(&fElementContainer);
    //do spaces
    for (auto& outerSpace : fSpaces) {
        fCollector.SetSpace(outerSpace);
        std::vector<KGSpace*> spaces = outerSpace->GetContent();
        for (auto& space : spaces) {
            if (!(space->HasExtension<KGMesh>())) {
                //no mesh extension, better make it first
                space->MakeExtension<KGMesh>();
                space->AcceptNode(&tMesher);
            }

            //now collect the mesh and map the elements to the KSSpace
            space->AcceptNode(&fCollector);
        }
    }

    //do sides
    for (auto& outerSide : fSides) {
        fCollector.SetSide(outerSide);
        std::vector<KGSurface*> sides = outerSide->GetContent();
        for (auto& side : sides) {
            if (!(side->HasExtension<KGMesh>())) {
                //no mesh extension, better make it first
                side->MakeExtension<KGMesh>();
                side->AcceptNode(&tMesher);
            }
            //now collect the mesh and map the elements to the KSSide
            side->AcceptNode(&fCollector);
        }
    }

    for (auto& outerSurface : fSurfaces) {
        fCollector.SetSurface(outerSurface);
        std::vector<KGSurface*> surfaces = outerSurface->GetContent();
        for (auto& surface : surfaces) {
            if (!(surface->HasExtension<KGMesh>())) {
                //no mesh extension, better make it first
                surface->MakeExtension<KGMesh>();
                surface->AcceptNode(&tMesher);
            }
            //now collect the mesh and map the elements to the KSSurface
            surface->AcceptNode(&fCollector);
        }
    }
}

void KSNavMeshedSpace::ConstructTree()
{
    bool haveTree = RetrieveTree();

    if (!haveTree) {
        navmsg(eNormal) << "navigation space <" << this->GetName() << "> is constructing the region octree..." << eom;

        if (fSpecifyMaxDepth) {
            fTreeBuilder.SetMaxTreeDepth(fMaxDepth);
        }

        if (fSpecifyResolution) {
            fTreeBuilder.SetSpatialResolution(fSpatialResolution);
        }

        if (fSpecifyAllowedElements) {
            fTreeBuilder.SetNAllowedElements(fNAllowedElements);
        }

        fTreeBuilder.SetNavigableMeshElementContainer(&fElementContainer);
        fTreeBuilder.SetTree(&fTree);
        fTreeBuilder.ConstructTree();

        std::string octree_info = fTreeBuilder.GetInformation();
        navmsg(eNormal) << "navigation space <" << this->GetName() << "> octree structure is...\n"
                        << octree_info << eom;

        SaveTree();
    }
    else {
        navmsg(eNormal) << "navigation space <" << this->GetName() << "> reconstructed an octree from <" << fFileName
                        << ">" << eom;
    }

    fWorldCube = KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::GetNodeObject(fTree.GetRootNode());

    //if no user specified tolerance, automatically try to estimate one
    if (!fUserSpecifiedAbsoluteTolerance) {
        double world_length = fWorldCube->GetLength();
        fAbsoluteTolerance = fRelativeTolerance * (world_length / std::pow(2.0, fMaxDepth));
    }
};

void KSNavMeshedSpace::SaveTree()
{
    KSNavOctreeData data;

    //compute a unique id tag for this tree
    //based on entity names, tree parameters and number of mesh elements

    std::stringstream se;
    for (auto& space : fSpaces) {
        se << space->GetName();
    }
    for (auto& side : fSides) {
        se << side->GetName();
    }
    for (auto& surface : fSurfaces) {
        se << surface->GetName();
    }
    se << fElementMap.size();
    std::string temp = se.str();

    //compute a hash for entity data
    KMD5HashGenerator entity_hasher;
    std::string entity_hash = entity_hasher.GenerateHashFromString(temp);

    //concatenate the parameters and hash them too
    std::stringstream sp;
    sp << fSpecifyMaxDepth;
    sp << fSpecifyResolution;
    sp << fSpecifyAllowedElements;

    if (fSpecifyMaxDepth) {
        sp << fMaxDepth;
    };
    if (fSpecifyResolution) {
        sp << fSpecifyResolution;
    };
    if (fSpecifyAllowedElements) {
        sp << fNAllowedElements;
    };
    temp = sp.str();

    KMD5HashGenerator param_hasher;
    std::string param_hash = param_hasher.GenerateHashFromString(temp);

    //make and set the unique_id
    std::stringstream us;
    us << entity_hash.substr(0, 6);
    us << param_hash.substr(0, 6);
    std::string unique_id = us.str();
    data.SetTreeID(unique_id);

    //set the tree parameters
    data.SetMaximumOctreeDepth(fMaxDepth);
    data.SetSpecifyMaximumOctreeDepth(0);
    if (fSpecifyMaxDepth) {
        data.SetSpecifyMaximumOctreeDepth(1);
    };

    data.SetSpatialResolution(fSpatialResolution);
    data.SetSpecifySpatialResolution(0);
    if (fSpecifyResolution) {
        data.SetSpecifySpatialResolution(1);
    };

    data.SetNumberOfAllowedElements(fNAllowedElements);
    data.SetSpecifyNumberOfAllowedElements(0);
    if (fSpecifyAllowedElements) {
        data.SetSpecifyNumberOfAllowedElements(1);
    };

    //flatten the tree
    KGTreeStructureExtractor<KGMeshNavigationNodeObjects> flattener;
    fTree.ApplyRecursiveAction(&flattener);

    //set number of tree nodes
    data.SetNumberOfTreeNodes(flattener.GetNumberOfNodes());

    //set the flattend tree data
    data.SetFlattenedTree(flattener.GetFlattenedTree());

    //collect the identity sets and the associated node data
    KGObjectCollector<KGMeshNavigationNodeObjects, KGIdentitySet> id_set_collector;
    fTree.ApplyCorecursiveAction(&id_set_collector);
    data.SetIdentitySetNodeIDs(id_set_collector.GetCollectedObjectAssociatedNodeIDs());
    data.SetIdentitySets(id_set_collector.GetCollectedObjects());

    //collect the cubes and associated node data
    KGObjectCollector<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>> cube_collector;
    fTree.ApplyCorecursiveAction(&cube_collector);
    data.SetCubeNodeIDs(cube_collector.GetCollectedObjectAssociatedNodeIDs());
    data.SetCubes(cube_collector.GetCollectedObjects());

    //stream the data out to file
    std::vector<std::string> labels;
    labels.push_back(entity_hash);
    labels.push_back(param_hash);
    labels.push_back(unique_id);
    labels.push_back(GetName());

    if (fFileName == "") {
        time_t t = time(nullptr);
        struct tm* now = localtime(&t);
        std::stringstream s;
        s << "KSOctree_" << (now->tm_year + 1900) << '-' << std::setfill('0') << std::setw(2) << (now->tm_mon + 1)
          << '-' << std::setfill('0') << std::setw(2) << now->tm_mday << "_" << std::setfill('0') << std::setw(2)
          << now->tm_hour << "-" << std::setfill('0') << std::setw(2) << now->tm_min << "-" << std::setfill('0')
          << std::setw(2) << now->tm_sec << ".kbd";
        fFileName = s.str();
    }

    //we use the kemfield file interface
    if (fPath == "") {
        fPath = SCRATCH_DEFAULT_DIR;
    }
    std::string file = fPath + "/" + fFileName;

    KEMFileInterface::GetInstance()->Write(file, data, "octree", labels);
}

bool KSNavMeshedSpace::RetrieveTree()
{
    //construct the labels
    fLabels.clear();

    std::stringstream se;
    for (auto& space : fSpaces) {
        se << space->GetName();
    }
    for (auto& side : fSides) {
        se << side->GetName();
    }
    for (auto& surface : fSurfaces) {
        se << surface->GetName();
    }
    se << fElementMap.size();
    std::string temp = se.str();

    //compute a hash for entity data
    KMD5HashGenerator entity_hasher;
    std::string entity_hash = entity_hasher.GenerateHashFromString(temp);

    //concatenate the parameters and hash them too
    std::stringstream sp;
    sp << fSpecifyMaxDepth;
    sp << fSpecifyResolution;
    sp << fSpecifyAllowedElements;

    if (fSpecifyMaxDepth) {
        sp << fMaxDepth;
    };
    if (fSpecifyResolution) {
        sp << fSpecifyResolution;
    };
    if (fSpecifyAllowedElements) {
        sp << fNAllowedElements;
    };
    temp = sp.str();

    KMD5HashGenerator param_hasher;
    std::string param_hash = param_hasher.GenerateHashFromString(temp);

    //make and set the unique_id
    std::stringstream us;
    us << entity_hash.substr(0, 6);
    us << param_hash.substr(0, 6);
    std::string unique_id = us.str();

    fLabels.push_back(entity_hash);
    fLabels.push_back(param_hash);
    fLabels.push_back(unique_id);
    fLabels.push_back(GetName());

    //switch to the scratch directory
    std::string previousDirectory = KEMFileInterface::GetInstance()->ActiveDirectory();
    KEMFileInterface::GetInstance()->ActiveDirectory(std::string(SCRATCH_DEFAULT_DIR));

    //find the file by labels
    std::set<std::string> file_set = KEMFileInterface::GetInstance()->FileNamesWithLabels(fLabels);

    if (file_set.size() != 1) {
        return false;
    }
    else {
        //we have tree data, so reconstruct the tree
        KSNavOctreeData data;
        fFileName = *(file_set.begin());
        KEMFileInterface::GetInstance()->ReadLabeled(fFileName, data, fLabels);

        //reset the active directory to its original state
        KEMFileInterface::GetInstance()->ActiveDirectory(previousDirectory);

        //reconstruct tree from the data;
        unsigned int n_nodes = data.GetNumberOfTreeNodes();

        //create the tree properties
        KGSpaceTreeProperties<KGMESH_DIM>* tree_prop = fTree.GetTreeProperties();
        tree_prop->SetMaxTreeDepth(fMaxDepth);
        unsigned int dim[KGMESH_DIM];
        for (unsigned int& i : dim) {
            i = 2;
        };  //we use an octtree in 3d
        tree_prop->SetDimensions(dim);
        tree_prop->SetNeighborOrder(0);

        //now we need to create a vector of N empty nodes, and enumerate them
        std::vector<KGMeshNavigationNode*> tree_nodes;
        tree_nodes.resize(n_nodes, nullptr);
        for (unsigned int i = 0; i < n_nodes; i++) {
            tree_nodes[i] = new KGMeshNavigationNode();
            tree_nodes[i]->SetID(i);

            //attach the tree properties to these nodes
            KGObjectRetriever<KGMeshNavigationNodeObjects, KGSpaceTreeProperties<KGMESH_DIM>>::SetNodeObject(
                tree_prop,
                tree_nodes[i]);

            //attach the mesh element container
            KGObjectRetriever<KGMeshNavigationNodeObjects, KGNavigableMeshElementContainer>::SetNodeObject(
                &fElementContainer,
                tree_nodes[i]);
        }

        //now we need to re-link the tree, by connecting child nodes to their parents
        std::vector<KGNodeData> tree_structure_data;
        data.GetFlattenedTree(&tree_structure_data);

        for (auto& i : tree_structure_data) {
            KGMeshNavigationNode* current_node = tree_nodes[i.GetID()];

            //link the children of this current node
            std::vector<unsigned int> child_ids;
            i.GetChildIDs(&child_ids);
            for (unsigned int child_id : child_ids) {
                current_node->AddChild(tree_nodes[child_id]);
            }
        }

        //now we need to re-attach objects to the node which owns them
        std::vector<int> id_set_node_ids;
        std::vector<KGIdentitySet*> id_sets;
        data.GetIdentitySetNodeIDs(&id_set_node_ids);
        data.GetIdentitySets(&id_sets);

        for (unsigned int i = 0; i < id_set_node_ids.size(); i++) {
            KGMeshNavigationNode* node = tree_nodes[id_set_node_ids[i]];
            KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet>::SetNodeObject(id_sets[i], node);
        }

        std::vector<int> cube_node_ids;
        std::vector<KGCube<KGMESH_DIM>*> cubes;
        data.GetCubeNodeIDs(&cube_node_ids);
        data.GetCubes(&cubes);

        for (unsigned int i = 0; i < cube_node_ids.size(); i++) {
            KGMeshNavigationNode* node = tree_nodes[cube_node_ids[i]];
            KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::SetNodeObject(cubes[i], node);
        }

        //now replace the tree's root node with the new one
        fTree.ReplaceRootNode(tree_nodes[0]);
        return true;
    }
};

void KSNavMeshedSpace::CalculateNavigation(const KSTrajectory& aTrajectory,
                                           const KSParticle& aTrajectoryInitialParticle,
                                           const KSParticle& aTrajectoryFinalParticle,
                                           const KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius,
                                           const double& aTrajectoryStep, KSParticle& aNavigationParticle,
                                           double& aNavigationStep, bool& aNavigationFlag)
{
    navmsg_debug("navigation space <" << this->GetName() << "> calculating navigation:" << eom);
    KSSpace* tCurrentSpace = aTrajectoryInitialParticle.GetCurrentSpace();

    KThreeVector tInitialPoint = aTrajectoryInitialParticle.GetPosition();
    KThreeVector tFinalPoint = aTrajectoryFinalParticle.GetPosition();
    double tTime;

    fSpaceEntity = nullptr;
    fSideEntity = nullptr;
    fSurfaceEntity = nullptr;
    fIsEntry = false;

    fNavigationFail = false;
    if (fFailCheck) {
        //check if the particle is within the space specifed by the octree
        if (!(fWorldCube->PointIsInside(KGPoint<KGMESH_DIM>(tInitialPoint)))) {
            navmsg(eWarning) << "inital position <" << tInitialPoint.X() << ", " << tInitialPoint.Y() << ", "
                             << tInitialPoint.Z() << "> is outside of octree search volume" << eom;
            fNavigationFail = true;
            aNavigationStep = 0.0;
            aNavigationFlag = true;
            return;
        }
    }

    //first we check if the bounding ball of the trajectory comes close enough
    //to the mesh that an intersection might occur
    fProximityChecker.SetPointAndRadius(aTrajectoryCenter, aTrajectoryRadius);
    fProximityChecker.ApplyAction(fTree.GetRootNode());  //TODO: implement some caching on search root node?

    if (fProximityChecker.SphereIntersectsMesh()) {
        navmsg_debug("navigation space <" << this->GetName() << "> detected trajectory is near mesh elements" << eom);

        //now we retrieve the piecewise linear approximation to the trajectory
        aTrajectory.GetPiecewiseLinearApproximation(aTrajectoryInitialParticle,
                                                    aTrajectoryFinalParticle,
                                                    &fIntermediateParticleStates);

        //TODO: may want to add a few stages of recursively subdividing and bounding sub-sections
        //of the intermediate states to possibley reduce the number of false
        //positive triggers for an intersection search
        //(since a bounding ball, while simple, is not the best culling volume for a nearly linear trajectory)

        auto startIt = fIntermediateParticleStates.begin();
        auto stopIt = startIt;
        stopIt++;

        unsigned int repeatCount = 0;
        unsigned int seg = 0;
        while (stopIt != fIntermediateParticleStates.end()) {
            tInitialPoint = startIt->GetPosition();
            tFinalPoint = stopIt->GetPosition();
            double length = (tFinalPoint - tInitialPoint).Magnitude();

            fFirstIntersectionFinder.SetLineSegment(tInitialPoint, tFinalPoint);

            //TODO: implement some caching on search root node?
            fFirstIntersectionFinder.ApplyAction(fTree.GetRootNode());

            if (fFirstIntersectionFinder.HasIntersectionWithMesh()) {

                //this segment of the trajectory has an intersection
                //compute the the time to intersection
                KThreeVector tIntersection = fFirstIntersectionFinder.GetIntersection();

                navmsg_debug("navigation space <" << this->GetName() << "> has detected a possible intersection"
                                                  << eom);

                KThreeVector tDirection = (tFinalPoint - tInitialPoint).Unit();
                double distance = (tIntersection - tInitialPoint).Magnitude();
                double t = (stopIt->GetTime() - startIt->GetTime());
                double v1 = (startIt->GetVelocity()).Dot(tDirection);
                double v2 = (stopIt->GetVelocity()).Dot(tDirection);
                double tDelta = SolveForTime(distance, t, v1, v2);
                tTime = startIt->GetTime() + tDelta;

                navmsg_debug("navigation space <" << this->GetName() << "> estimates time step of " << tDelta << eom);

                //sanity check
                if (tTime > stopIt->GetTime()) {
                    tTime = stopIt->GetTime();
                };
                if (tTime < startIt->GetTime()) {
                    tTime = startIt->GetTime();
                };

                //now deterimine what object it has encountered
                //get the mesh element and its id
                const KGNavigableMeshElement* mesh_elem = fFirstIntersectionFinder.GetIntersectedMeshElement();
                fMeshElementID = mesh_elem->GetID();

                //TODO: we need to explicitly deal with the case where we are both intersecting
                //a side and a space at the same time (this is more complicated than it sounds)

                //look up the mesh elements parent entity to determine if we
                //hit a parent side, child side, or surface, we also check to
                //see if this intersection extremely close to the last surface/space
                //crossing (if it is, its probably because of numerical error) and we should ignore it
                bool validIntersection = true;
                bool skipSegment = false;

                double dist_to_add = 0.0;
                double dist_from_last = std::numeric_limits<double>::max();

                switch (fElementMap[fMeshElementID].fType) {
                    case KSNAVMESHEDSPACE_SPACE:
                        //intersection with space element
                        fSpaceEntity = fElementMap[fMeshElementID].fSpace;
                        fSideEntity = nullptr;
                        fSurfaceEntity = nullptr;

                        if (tCurrentSpace == fSpaceEntity) {
                            //we are exiting the current space
                            fIsEntry = false;
                        }
                        else {
                            //we have an entry
                            fIsEntry = true;
                        }

                        dist_from_last = (tIntersection - fLastIntersection).Magnitude();
                        if (dist_from_last < fRelativeTolerance * length || dist_from_last < fAbsoluteTolerance) {
                            if (dist_from_last > fRelativeTolerance * length) {
                                dist_to_add = fAbsoluteTolerance;
                            }
                            else {
                                dist_to_add = fRelativeTolerance * length;
                            }

                            //just started tracking this particle
                            if (fLastSpaceEntity == nullptr) {
                                navmsg_debug("  invalid intersection detected near starting position on <"
                                             << fSpaceEntity->GetName() << ">" << eom);
                                validIntersection = false;
                            }
                            else if (fLastSpaceEntity == fSpaceEntity) {
                                //same entity or very close to last intersection
                                navmsg_debug("  invalid intersection detected near previous intersection on "
                                             << fSpaceEntity->GetName() << ">" << eom);
                                validIntersection = false;
                            }
                        }
                        else if (fLastSpaceEntity == fSpaceEntity || fLastSpaceEntity == nullptr) {
                            //check if navigation count is within 1 step
                            //and distance from last intersection is within the length of the segment
                            //and n segments is more than 1
                            if ((fNavigationCount - fLastSpaceCount <= 1) && (dist_from_last < length)) {
                                //very likely have an invalid intersection due to numerical round off
                                //but because the absolute/relative tolerances are too tight
                                //it has not been detected, so we issue a debug warning and skip this segment
                                navmsg(eWarning)
                                    << " skipping segment because invalid intersection detected near previous intersection at on <"
                                    << fSpaceEntity->GetName() << ">, your navigator tolerances are probably too small!"
                                    << eom;

                                validIntersection = false;
                                skipSegment = true;
                            }
                        }

                        break;
                    case KSNAVMESHEDSPACE_SIDE:
                        //intersection with side element
                        fSpaceEntity = nullptr;
                        fSideEntity = fElementMap[fMeshElementID].fSide;
                        fSurfaceEntity = nullptr;

                        dist_from_last = (tIntersection - fLastIntersection).Magnitude();
                        if (dist_from_last < fRelativeTolerance * aTrajectoryRadius ||
                            dist_from_last < fAbsoluteTolerance) {
                            if (dist_from_last > fRelativeTolerance * aTrajectoryRadius) {
                                dist_to_add = fAbsoluteTolerance;
                            }
                            else {
                                dist_to_add = fRelativeTolerance * aTrajectoryRadius;
                            }

                            //just started tracking this particle
                            if (fLastSideEntity == nullptr) {
                                navmsg_debug("  invalid intersection detected near starting position at on <"
                                             << fSideEntity->GetName() << ">" << eom);
                                validIntersection = false;
                            }
                            else if (fLastSideEntity == fSideEntity) {
                                //same entity or very close to last intersection
                                navmsg_debug("  invalid intersection detected near previous intersection at on <"
                                             << fSideEntity->GetName() << ">" << eom);
                                validIntersection = false;
                            }
                        }
                        else if (fLastSideEntity == fSideEntity || fLastSideEntity == nullptr) {
                            //check if navigation count is within 1 step
                            //and distance from last intersection is within the length of the segment
                            //and n segments is more than 1
                            if ((fNavigationCount - fLastSideCount <= 1) && (dist_from_last < length)) {
                                //very likely have an invalid intersection due to numerical round off
                                //but because the absolute/relative tolerances are too tight
                                //it has not been detected, so we issue a debug warning and skip this segment
                                navmsg(eWarning)
                                    << " skipping segment because invalid intersection detected near previous intersection at on <"
                                    << fSideEntity->GetName() << ">, your navigator tolerances are probably too small!"
                                    << eom;
                                validIntersection = false;
                                skipSegment = true;
                            }
                        }

                        break;
                    case KSNAVMESHEDSPACE_SURFACE:
                        //intersection with surface element
                        fSpaceEntity = nullptr;
                        fSideEntity = nullptr;
                        fSurfaceEntity = fElementMap[fMeshElementID].fSurface;

                        dist_from_last = (tIntersection - fLastIntersection).Magnitude();
                        if (dist_from_last < fRelativeTolerance * aTrajectoryRadius ||
                            dist_from_last < fAbsoluteTolerance) {
                            if (dist_from_last > fRelativeTolerance * aTrajectoryRadius) {
                                dist_to_add = fAbsoluteTolerance;
                            }
                            else {
                                dist_to_add = fRelativeTolerance * aTrajectoryRadius;
                            }

                            //just started tracking this particle
                            if (fLastSurfaceEntity == nullptr) {
                                navmsg_debug("  invalid intersection detected near starting position at on <"
                                             << fSurfaceEntity->GetName() << ">" << eom);
                                validIntersection = false;
                            }
                            else if (fLastSurfaceEntity == fSurfaceEntity) {
                                //same entity or very close to last intersection
                                navmsg_debug("  invalid intersection detected near previous intersection at on <"
                                             << fSurfaceEntity->GetName() << ">" << eom);
                                validIntersection = false;
                            }
                        }
                        else if (fLastSurfaceEntity == fSurfaceEntity || fLastSurfaceEntity == nullptr) {
                            //check if navigation count is within 1 step
                            //and distance from last intersection is within the length of the segment
                            //and n segments is more than 1
                            if ((fNavigationCount - fLastSurfaceCount <= 1) && (dist_from_last < length)) {
                                //very likely have an invalid intersection due to numerical round off
                                //but because the absolute/relative tolerances are too tight
                                //it has not been detected, so we issue a debug warning and skip this segment
                                navmsg(eWarning)
                                    << " skipping segment because invalid intersection detected near previous intersection at on <"
                                    << fSurfaceEntity->GetName()
                                    << ">, your navigator tolerances are probably too small!" << eom;
                                validIntersection = false;
                                skipSegment = true;
                            }
                        }

                        break;
                };

                seg++;

                if (validIntersection)  //intersection is not a repeat of the last
                {
                    //compute the particle state at the intersection
                    aNavigationParticle = aTrajectoryInitialParticle;
                    aNavigationParticle.SetCurrentSpace(tCurrentSpace);
                    aNavigationStep = tTime - aTrajectoryInitialParticle.GetTime();
                    aTrajectory.ExecuteTrajectory(aNavigationStep, aNavigationParticle);
                    aNavigationFlag = true;

                    navmsg_debug("navigation space <"
                                 << this->GetName()
                                 << "> detected valid intersection after time step: " << aNavigationStep << eom);

                    return;
                }
                else {
                    //we have a repeated intersection due to numerical error
                    //however we need to check the remainder of this current segment,
                    //this is needed in case of a very thin volume
                    //because if we discard the rest of this segment without checking for additional
                    //intersections we might miss the next surface just past the current one
                    //so compute a point just beyond the repeated intersection

                    if (skipSegment) {
                        //this entire segment is too close to last intersection
                        //go to next one
                        startIt++;
                        stopIt++;
                        repeatCount = 0;
                        continue;
                    }

                    if (repeatCount < 4)  //avoid infinite loop in case of severe problems
                    {
                        distance += dist_to_add;
                        if (distance < length) {
                            navmsg_debug(" incrementing intersection search by <" << dist_to_add << "> and continuing "
                                                                                  << eom);
                            tTime = startIt->GetTime() + SolveForTime(distance, t, v1, v2);
                            //sanity check
                            if (tTime > stopIt->GetTime()) {
                                tTime = stopIt->GetTime();
                            };
                            if (tTime < startIt->GetTime()) {
                                tTime = startIt->GetTime();
                            };
                            double step = tTime - aTrajectoryInitialParticle.GetTime();
                            aTrajectory.ExecuteTrajectory(step, *startIt);
                            repeatCount++;
                            continue;
                        }
                        else {
                            navmsg_debug(
                                "  skipping trajectory segment because it is entirely within tolerance of the last intersection "
                                << eom);
                            //this entire segment is too close to last intersection
                            //go to next one
                            startIt++;
                            stopIt++;
                            repeatCount = 0;
                            continue;
                        }
                    }
                    else {
                        navmsg_debug("skipping trajectory segment after reaching incremental search limit " << eom);
                    }
                }
            }

            //update iterator for next segment
            startIt++;
            stopIt++;
            repeatCount = 0;
        }
    }


    //no intersections
    aNavigationParticle.SetCurrentSpace(tCurrentSpace);
    aNavigationParticle = aTrajectoryFinalParticle;
    aNavigationStep = aTrajectoryStep;
    aNavigationFlag = false;

    return;
}

void KSNavMeshedSpace::ExecuteNavigation(const KSParticle& aNavigationParticle, KSParticle& aFinalParticle,
                                         KSParticleQueue& aParticleQueue) const
{
    navmsg_debug("navigation space <" << this->GetName() << "> executing navigation:" << eom);

    //kill the particle if the navigation was wrong
    if (fNavigationFail) {
        fNavigationFail = false;
        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel("navigator_fail");
        aFinalParticle.SetActive(false);
        return;
    }

    //we had a space intersection
    if (fSpaceEntity != nullptr) {
        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fSpaceEntity->GetName());

        if (fIsEntry) {
            navmsg(eNormal) << "  space <" << fSpaceEntity->GetName() << "> was entered " << eom;
            aFinalParticle.AddLabel("enter");

            if (fEnterSplit == true) {
                auto* tEnterSplitParticle = new KSParticle(aFinalParticle);
                tEnterSplitParticle->SetCurrentSpace(fSpaceEntity);
                tEnterSplitParticle->SetCurrentSurface(nullptr);
                tEnterSplitParticle->SetCurrentSide(nullptr);
                tEnterSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tEnterSplitParticle);
                aFinalParticle.SetActive(false);
            }
        }
        else {
            navmsg(eNormal) << "  space <" << fSpaceEntity->GetName() << "> was exited " << eom;
            aFinalParticle.AddLabel("exit");

            bool world_exit = false;
            if (fSpaceEntity->GetParent() == nullptr) {
                world_exit = true;
            }

            if (fExitSplit == true && !world_exit) {
                auto* tExitSplitParticle = new KSParticle(aFinalParticle);
                tExitSplitParticle->SetCurrentSpace(fSpaceEntity->GetParent());
                tExitSplitParticle->SetCurrentSurface(nullptr);
                tExitSplitParticle->SetCurrentSide(nullptr);
                tExitSplitParticle->ResetFieldCaching();
                aParticleQueue.push_back(tExitSplitParticle);
                aFinalParticle.SetActive(false);
            }

            //particle has exited the world volume, kill it
            if (world_exit) {
                aFinalParticle.SetLabel(GetName());
                aFinalParticle.AddLabel("world_exit");
                aFinalParticle.SetActive(false);
            }
        }

        return;
    }

    if (fSideEntity != nullptr) {
        navmsg(eNormal) << "  side <" << fSideEntity->GetName() << "> was crossed" << eom;
        aFinalParticle = aNavigationParticle;
        ;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fSideEntity->GetName());
        aFinalParticle.AddLabel("crossed");
        return;
    }

    if (fSurfaceEntity != nullptr) {
        navmsg(eNormal) << "   surface <" << fSurfaceEntity->GetName() << "> was crossed" << eom;
        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fSurfaceEntity->GetName());
        aFinalParticle.AddLabel("crossed");
        return;
    }

    navmsg(eError) << "could not determine space navigation" << eom;

    return;
}


void KSNavMeshedSpace::FinalizeNavigation(KSParticle& aFinalParticle) const
{
    navmsg_debug("navigation space <" << this->GetName() << "> finalizing navigation:" << eom);

    //we had a space intersection
    if (fSpaceEntity != nullptr) {
        if (fIsEntry) {
            navmsg_debug("  finalizing navigation for exiting of space <" << fSpaceEntity->GetName() << "> " << eom);
            aFinalParticle.SetCurrentSpace(fSpaceEntity);
            aFinalParticle.SetCurrentSide(nullptr);
            aFinalParticle.SetCurrentSurface(nullptr);
            aFinalParticle.ResetFieldCaching();

            fLastEntityType = KSNAVMESHEDSPACE_SPACE;
            fLastMeshElementID = fMeshElementID;
            fLastSpaceEntity = fSpaceEntity;
            fLastSideEntity = nullptr;
            fLastSurfaceEntity = nullptr;
            fLastIntersection = aFinalParticle.GetPosition();
            fLastDirection = aFinalParticle.GetMomentum().Unit();

            fSpaceEntity->Enter();
            fSpaceEntity = nullptr;

            fLastSpaceCount = fNavigationCount;
        }
        else {
            navmsg_debug("  finalizing navigation for entering of space <" << fSpaceEntity->GetName() << "> " << eom);
            aFinalParticle.SetCurrentSpace(fSpaceEntity->GetParent());
            aFinalParticle.SetCurrentSide(nullptr);
            aFinalParticle.SetCurrentSurface(nullptr);
            aFinalParticle.ResetFieldCaching();

            fLastEntityType = KSNAVMESHEDSPACE_SPACE;
            fLastMeshElementID = fMeshElementID;
            fLastSpaceEntity = fSpaceEntity;
            fLastSideEntity = nullptr;
            fLastSurfaceEntity = nullptr;
            fLastIntersection = aFinalParticle.GetPosition();
            fLastDirection = aFinalParticle.GetMomentum().Unit();

            fSpaceEntity->Exit();
            fSpaceEntity = nullptr;

            fLastSpaceCount = fNavigationCount;
        }

        fNavigationCount++;
        return;
    }

    if (fSideEntity != nullptr) {
        navmsg_debug("  finalizing navigation for crossing of side <" << fSideEntity->GetName() << "> " << eom);

        aFinalParticle.SetCurrentSide(fSideEntity);
        aFinalParticle.SetCurrentSurface(nullptr);
        aFinalParticle.ResetFieldCaching();

        fLastEntityType = KSNAVMESHEDSPACE_SIDE;
        fLastMeshElementID = fMeshElementID;
        fLastSpaceEntity = aFinalParticle.GetCurrentSpace();
        fLastSideEntity = fSideEntity;
        fLastSurfaceEntity = nullptr;
        fLastIntersection = aFinalParticle.GetPosition();
        fLastDirection = aFinalParticle.GetMomentum().Unit();

        fSideEntity->On();
        fSideEntity = nullptr;

        fLastSideCount = fNavigationCount;
        fNavigationCount++;
        return;
    }

    if (fSurfaceEntity != nullptr) {
        navmsg_debug("  finalizing navigation for crossing of surface <" << fSurfaceEntity->GetName() << "> " << eom);

        aFinalParticle.SetCurrentSide(nullptr);
        aFinalParticle.SetCurrentSurface(fSurfaceEntity);
        aFinalParticle.ResetFieldCaching();

        fLastEntityType = KSNAVMESHEDSPACE_SURFACE;
        fLastMeshElementID = fMeshElementID;
        fLastSpaceEntity = aFinalParticle.GetCurrentSpace();
        fLastSideEntity = nullptr;
        fLastSurfaceEntity = fSurfaceEntity;
        fLastIntersection = aFinalParticle.GetPosition();
        fLastDirection = aFinalParticle.GetMomentum().Unit();

        fSurfaceEntity->On();
        fSurfaceEntity = nullptr;

        fLastSurfaceCount = fNavigationCount;
        fNavigationCount++;
        return;
    }

    navmsg(eError) << "could not finalize space navigation" << eom;

    return;
}


void KSNavMeshedSpace::StartNavigation(KSParticle& aParticle, KSSpace* aRoot)
{
    // reset navigation
    fSpaceEntity = nullptr;
    fSideEntity = nullptr;
    fSurfaceEntity = nullptr;
    fNavigationCount = 0;
    fLastSideCount = 0;
    fLastSpaceCount = 0;
    fLastSurfaceCount = 0;

    fLastSpaceEntity = aParticle.GetCurrentSpace();
    fLastSurfaceEntity = aParticle.GetCurrentSurface();
    fLastSideEntity = aParticle.GetCurrentSide();

    fLastIntersection = aParticle.GetPosition();

    navmsg_debug("navigation space <" << this->GetName() << "> starting navigation:" << eom);

    if (aParticle.GetCurrentSpace() == nullptr) {
        navmsg_debug("  computing fresh initial state" << eom);
        int tIndex = 0;
        KSSpace* tParentSpace = aRoot;
        KSSpace* tSpace = nullptr;
        while (tIndex < tParentSpace->GetSpaceCount()) {
            tSpace = tParentSpace->GetSpace(tIndex);
            if (tSpace->Outside(aParticle.GetPosition()) == false) {
                navmsg_debug("  activating space <" << tSpace->GetName() << ">" << eom);
                tSpace->Enter();
                tParentSpace = tSpace;
                tIndex = 0;
            }
            else {
                navmsg_debug("  skipping space <" << tSpace->GetName() << ">" << eom);
                tIndex++;
            }
        }

        aParticle.SetCurrentSpace(tParentSpace);
    }
    else {
        navmsg_debug("  entering given initial state" << eom);

        KSSpace* tSpace = aParticle.GetCurrentSpace();
        KSSurface* tSurface = aParticle.GetCurrentSurface();
        KSSide* tSide = aParticle.GetCurrentSide();
        std::deque<KSSpace*> tSequence;

        // get into the correct space state
        while (tSpace != aRoot && tSpace != nullptr) {
            tSequence.push_front(tSpace);
            tSpace = tSpace->GetParent();
        }

        for (auto& tIt : tSequence) {
            tSpace = tIt;
            navmsg_debug("  entering space <" << tSpace->GetName() << ">" << eom);
            tSpace->Enter();
        }

        if (tSurface != nullptr) {
            navmsg_debug("  activating surface <" << tSurface->GetName() << ">" << eom);
            tSurface->On();
        }

        if (tSide != nullptr) {
            navmsg_debug("  activating side <" << tSide->GetName() << ">" << eom);
            tSide->On();
        }
    }
    return;
}

void KSNavMeshedSpace::StopNavigation(KSParticle& aParticle, KSSpace* aRoot)
{
    // reset navigation
    fSpaceEntity = nullptr;
    fSideEntity = nullptr;
    fSurfaceEntity = nullptr;
    fNavigationCount = 0;
    fLastSideCount = 0;
    fLastSpaceCount = 0;
    fLastSurfaceCount = 0;

    fLastSideEntity = nullptr;
    fLastSpaceEntity = nullptr;
    fLastSurfaceEntity = nullptr;

    std::deque<KSSpace*> tSpaces;

    KSSpace* tSpace = aParticle.GetCurrentSpace();
    KSSurface* tSurface = aParticle.GetCurrentSurface();
    KSSide* tSide = aParticle.GetCurrentSide();

    // undo side state
    if (tSide != nullptr) {
        navmsg_debug("  deactivating side <" << tSide->GetName() << ">" << eom);
        tSide->Off();
    }

    // undo surface state
    if (tSurface != nullptr) {
        navmsg_debug("  deactivating surface <" << tSurface->GetName() << ">" << eom);
        tSurface->Off();
    }

    // undo space state
    while (tSpace != aRoot && tSpace != nullptr) {
        tSpaces.push_back(tSpace);
        tSpace = tSpace->GetParent();
    }
    for (auto& tIt : tSpaces) {
        tSpace = tIt;
        navmsg_debug("  deactivating space <" << tSpace->GetName() << ">" << eom);
        tSpace->Exit();
    }

    return;
}

double KSNavMeshedSpace::SolveForTime(double distance, double t, double v1, double v2) const
{
    //solving for the time to travel the specified distance
    //along a straight line path with initial velocity
    //v1 and final velocity v2, assuming constant acceleration
    double acc = (v2 - v1) / t;

    //solve the quadratic
    double a = 0.5 * acc;
    double b = v1;
    double c = -1.0 * distance;
    double disc = b * b - 4.0 * a * c;

    if (disc < 0.0) {
        if (std::fabs(disc) < (1e-13) * b * b) {
            //disc became negative because of rounding error
            //clamp to zero return single root
            return -b / (2.0 * a);
        }
        else {
            //something went wrong with the coefficients, compute the linear estimate
            return distance / ((v1 + v2) / 2.0);
        }
    }

    bool x1_is_valid = true;
    bool x2_is_valid = true;

    disc = std::sqrt(disc);
    double x1 = (-b + disc) / (2.0 * a);
    double x2 = (-b - disc) / (2.0 * a);

    if (x1 < 0.0 || x1 > t) {
        x1_is_valid = false;
    };
    if (x2 < 0.0 || x1 > t) {
        x2_is_valid = false;
    };

    if (!x1_is_valid && !x2_is_valid) {
        //something went wrong with the coefficients, compute the linear estimate
        return distance / ((v1 + v2) / 2.0);
    }
    else {
        if (x1_is_valid && x2_is_valid) {
            if (x1 < x2) {
                return x1;
            }
            else {
                return x2;
            }
        }
        if (x1_is_valid) {
            return x1;
        };
        if (x2_is_valid) {
            return x2;
        };
    }

    //should never reach here, but if we do return end-point
    return t;
}


}  // namespace Kassiopeia
