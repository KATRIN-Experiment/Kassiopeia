#include "KGNavigableMeshTreeBuilder.hh"

#include "KGBoundaryCalculator.hh"
#include "KGNavigableMeshElementSorter.hh"
#include "KGNavigableMeshTreeInformationExtractor.hh"
#include "KGSubdivisionCondition.hh"

#define KGBBALL_EPS 1e-14
//#define KGNavigableMeshTreeBuilder_DEBUG__

namespace KGeoBag
{

KGNavigableMeshTreeBuilder::KGNavigableMeshTreeBuilder() :
    fUseAuto(true),
    fUseSpatialResolution(false),
    fMaximumTreeDepth(0),
    fNAllowedElements(1),
    fTree(nullptr),
    fContainer(nullptr)
{
    fInfoString = "";
};

KGNavigableMeshTreeBuilder::~KGNavigableMeshTreeBuilder() = default;
;

void KGNavigableMeshTreeBuilder::SetNavigableMeshElementContainer(KGNavigableMeshElementContainer* container)
{
    fContainer = container;
}

KGNavigableMeshElementContainer* KGNavigableMeshTreeBuilder::GetNavigableMeshElementContainer()
{
    return fContainer;
}

void KGNavigableMeshTreeBuilder::SetTree(KGNavigableMeshTree* tree)
{
    fTree = tree;
}

KGNavigableMeshTree* KGNavigableMeshTreeBuilder::GetTree()
{
    return fTree;
}

void KGNavigableMeshTreeBuilder::ConstructTree()
{
    ConstructRootNode();
    PerformSpatialSubdivision();
}

void KGNavigableMeshTreeBuilder::ConstructRootNode()
{
    //estimate the bounding cube containing all of the mesh elements
    //and determine the minimum element size
    KGBoundaryCalculator<KGMESH_DIM> boundary_calc;
    boundary_calc.Reset();
    unsigned int n_elements = fContainer->GetNElements();


    double bball_r_min = (fContainer->GetElementBoundingBall(0)).GetRadius();
    if (bball_r_min == 0.0) {
        bball_r_min = KGBBALL_EPS;
    };
    double bball_r_max = bball_r_min;
    for (unsigned int i = 0; i < n_elements; i++) {
        KGBall<KGMESH_DIM> ball = fContainer->GetElementBoundingBall(i);
        boundary_calc.AddBall(&ball);
        if (ball.GetRadius() < bball_r_min) {
            //prevent the minimum bounding ball radius from becoming zero
            //in case the mesh has bad elements
            if (ball.GetRadius() > KGBBALL_EPS * bball_r_min) {
                bball_r_min = ball.GetRadius();
            }
        }

        if (ball.GetRadius() > bball_r_max) {
            bball_r_max = ball.GetRadius();
        }
    }

    auto* world_volume = new KGCube<KGMESH_DIM>();
    *world_volume = boundary_calc.GetMinimalBoundingCube();

    KGPoint<KGMESH_DIM> fWorldCenter = world_volume->GetCenter();
    double fWorldLength = 1.01 * (world_volume->GetLength());  //make volume length 1% bigger than strictly necessary

    //world_volume->SetCenter(p);
    world_volume->SetLength(fWorldLength);

    fRegionSideLength = fWorldLength;

#ifdef KGNavigableMeshTreeBuilder_DEBUG__
    std::cout << "mesh container has: " << n_elements << " elements" << std::endl;
    std::cout << "estimated world cube center is (" << fWorldCenter[0] << ", " << fWorldCenter[1] << ", "
              << fWorldCenter[2] << ")" << std::endl;
    std::cout << "estimated world cube size length is " << fWorldLength << std::endl;
    std::cout << "min bounding ball radius = " << bball_r_min << std::endl;
    std::cout << "max bounding ball radius = " << bball_r_max << std::endl;
    std::cout << "geometric mean = " << std::sqrt(bball_r_min * bball_r_max) << std::endl;
#endif

    if (fUseAuto)  //automatically determine appropriate max tree depth
    {
        //determime how many tree levels we need in order to get a appropriate size
        //for the smallest leaf node (we use the geometric mean of smallest/largest elements)
        unsigned int n_levels = 0;
        double len = fWorldLength;
        double geom_mean = std::sqrt(bball_r_min * bball_r_max);
        do {
            len *= 0.5;
            n_levels++;
        } while (len > 2 * geom_mean);
        fMaximumTreeDepth = n_levels;
    }
    else  //either use user set max-tree depth, or spatial resolution to determine tree depth
    {
        if (fUseSpatialResolution) {
            //determime how many tree levels we need in order to get to the required spatial resolution
            unsigned int n_levels = 0;
            double len = fWorldLength;
            do {
                len *= 0.5;
                n_levels++;
            } while (len > fSpatialResolution);
            fMaximumTreeDepth = n_levels;
        }
    }


    KGSpaceTreeProperties<KGMESH_DIM>* tree_prop = fTree->GetTreeProperties();
    tree_prop->SetMaxTreeDepth(fMaximumTreeDepth);
    unsigned int dim[KGMESH_DIM];
    for (unsigned int& i : dim) {
        i = 2;
    };  //we use an oct-tree in 3d
    tree_prop->SetDimensions(dim);
    tree_prop->SetNeighborOrder(0);

    KGMeshNavigationNode* root = fTree->GetRootNode();
    //set the world volume
    KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::SetNodeObject(world_volume, root);
    //set the tree properties
    KGObjectRetriever<KGMeshNavigationNodeObjects, KGSpaceTreeProperties<KGMESH_DIM>>::SetNodeObject(tree_prop, root);
    //set the element container
    KGObjectRetriever<KGMeshNavigationNodeObjects, KGNavigableMeshElementContainer>::SetNodeObject(fContainer, root);

    //add the complete id set of all elements to the root node
    auto* root_list = new KGIdentitySet();
    for (unsigned int i = 0; i < n_elements; i++) {
        root_list->AddID(i);
    }
    //set the id set
    KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet>::SetNodeObject(root_list, root);

    //set basic root node properties
    root->SetID(tree_prop->RegisterNode());
    root->SetIndex(0);
    root->SetParent(nullptr);
}

void KGNavigableMeshTreeBuilder::PerformSpatialSubdivision()
{
    //conditions for subdivision of a node
    KGSubdivisionCondition sub_cond;
    sub_cond.SetMeshElementContainer(fContainer);
    sub_cond.SetNAllowedElements(fNAllowedElements);
    fTree->SetSubdivisionCondition(&sub_cond);

    //things to do on a node after it has been visited by the progenitor
    KGNavigableMeshElementSorter element_sorter;
    element_sorter.SetMeshElementContainer(fContainer);
    fTree->AddPostSubdivisionAction(&element_sorter);

    fTree->ConstructTree();

    KGNavigableMeshTreeInformationExtractor extractor;
    extractor.SetNElements(fContainer->GetNElements());
    fTree->ApplyCorecursiveAction(&extractor);
    fInfoString = extractor.GetStatistics();
#ifdef KGNavigableMeshTreeBuilder_DEBUG__
    extractor.PrintStatistics();
#endif
}


}  // namespace KGeoBag
