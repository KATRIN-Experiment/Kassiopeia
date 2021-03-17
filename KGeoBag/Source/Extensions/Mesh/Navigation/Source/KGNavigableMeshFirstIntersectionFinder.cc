#include "KGNavigableMeshFirstIntersectionFinder.hh"

namespace KGeoBag
{

KGNavigableMeshFirstIntersectionFinder::KGNavigableMeshFirstIntersectionFinder() :
    fContainer(nullptr),
    fDefaultStackSize(512),
    fStackReallocateLimit(384)
{
    fPreallocatedStack.resize(fDefaultStackSize, nullptr);
    fVerbose = false;
};

KGNavigableMeshFirstIntersectionFinder::~KGNavigableMeshFirstIntersectionFinder() = default;
;


void KGNavigableMeshFirstIntersectionFinder::SortOctreeNodes(unsigned int n_nodes,
                                                             std::pair<KGMeshNavigationNode*, double>* nodes)
{
    //using a sorting network for arrays of up to 8 values
    //sorting rules were generated from: http://pages.ripco.net/~jgamble/nw.html
    std::pair<KGMeshNavigationNode*, double> tmp;
#define SWAP(x, y)                                                                                                     \
    if (nodes[y].second > nodes[x].second) {                                                                           \
        tmp = nodes[x];                                                                                                \
        nodes[x] = nodes[y];                                                                                           \
        nodes[y] = tmp;                                                                                                \
    }

    switch (n_nodes) {
        case 2:
            SWAP(0, 1);
            return;
        case 3:
            SWAP(1, 2);
            SWAP(0, 2);
            SWAP(0, 1);
            return;
        case 4:
            SWAP(0, 1);
            SWAP(2, 3);
            SWAP(0, 2);
            SWAP(1, 3);
            SWAP(1, 2);
            return;
        case 5:
            SWAP(0, 1);
            SWAP(3, 4);
            SWAP(2, 4);
            SWAP(2, 3);
            SWAP(1, 4);
            SWAP(0, 3);
            SWAP(0, 2);
            SWAP(1, 3);
            SWAP(1, 2);
            return;
        case 6:
            SWAP(1, 2);
            SWAP(4, 5);
            SWAP(0, 2);
            SWAP(3, 5);
            SWAP(0, 1);
            SWAP(3, 4);
            SWAP(2, 5);
            SWAP(0, 3);
            SWAP(1, 4);
            SWAP(2, 4);
            SWAP(1, 3);
            SWAP(2, 3);
            return;
        case 7:
            SWAP(1, 2);
            SWAP(3, 4);
            SWAP(5, 6);
            SWAP(0, 2);
            SWAP(3, 5);
            SWAP(4, 6);
            SWAP(0, 1);
            SWAP(4, 5);
            SWAP(2, 6);
            SWAP(0, 4);
            SWAP(1, 5);
            SWAP(0, 3);
            SWAP(2, 5);
            SWAP(1, 3);
            SWAP(2, 4);
            SWAP(2, 3);
            return;
        case 8:
            SWAP(0, 1);
            SWAP(2, 3);
            SWAP(4, 5);
            SWAP(6, 7);
            SWAP(0, 2);
            SWAP(1, 3);
            SWAP(4, 6);
            SWAP(5, 7);
            SWAP(1, 2);
            SWAP(5, 6);
            SWAP(0, 4);
            SWAP(3, 7);
            SWAP(1, 5);
            SWAP(2, 6);
            SWAP(1, 4);
            SWAP(3, 6);
            SWAP(2, 4);
            SWAP(3, 5);
            SWAP(3, 4);
            return;
        default:
            return;
    }
#undef SWAP
}


void KGNavigableMeshFirstIntersectionFinder::SetLineSegment(const KThreeVector& start, const KThreeVector& end)
{
    fStartPoint = start;
    fEndPoint = end;
    fDirection = fEndPoint - fStartPoint;
    fLength = fDirection.Magnitude();
    fDirection = fDirection.Unit();
    fHaveIntersection = false;
    fFirstIntersection = KThreeVector::sZero;
}

bool KGNavigableMeshFirstIntersectionFinder::HasIntersectionWithMesh() const
{
    return fHaveIntersection;
};

KThreeVector KGNavigableMeshFirstIntersectionFinder::GetIntersection() const
{
    return fFirstIntersection;
};


void KGNavigableMeshFirstIntersectionFinder::NearestPointOnLineSegment(const KThreeVector& aPoint,
                                                                       KThreeVector& aNearest, double& t) const
{
    t = ((aPoint - fStartPoint) * fDirection);
    if (t < 0.) {
        aNearest = fStartPoint;
        t = 0;
        return;
    }
    if (t > fLength) {
        aNearest = fEndPoint;
        t = fLength;
        return;
    }
    aNearest = fStartPoint + t * fDirection;
}

double KGNavigableMeshFirstIntersectionFinder::LineSegmentDistanceToPoint(const KThreeVector& aPoint) const
{
    KThreeVector nearest;
    double t;
    NearestPointOnLineSegment(aPoint, nearest, t);
    return (aPoint - nearest).Magnitude();
    // KThreeVector del = fStartPoint -  aPoint;
    // double t = -1.0*(del*fDirection);
    // if(t < 0.)
    // {
    //     return del.Magnitude();
    // }
    // if(t > fLength)
    // {
    //     return (fEndPoint - aPoint).Magnitude();
    // }
    // return ( del + t*fDirection ).Magnitude();
}

bool KGNavigableMeshFirstIntersectionFinder::LineSegmentIntersectsCube(const KGCube<KGMESH_DIM>& cube,
                                                                       double& distance) const
{
    //uses 'slab clipping' to check if line segment intersects the cube
    //as specified in chapter 11.2 of Geometric Tools for Computer Graphics
    //by Schneider & Eberly

    double len_over_two = cube[3] / 2.;
    double t_min = 0;
    double t_max = fLength;
    double tmp, disp;

    //check x planes
    disp = cube[0] - fStartPoint[0];
    double t0x = (disp - len_over_two) / fDirection[0];
    double t1x = (disp + len_over_two) / fDirection[0];
    //swap if out of order
    if (t0x > t1x) {
        tmp = t0x;
        t0x = t1x;
        t1x = tmp;
    };

    //update valid interval
    if (t0x > t_min) {
        t_min = t0x;
    };
    if (t1x < t_max) {
        t_max = t1x;
    };

    if (t_min > t_max || t_max < 0 || t_min > fLength) {
        return false;
    }

    //check y planes
    disp = cube[1] - fStartPoint[1];
    double t0y = (disp - len_over_two) / fDirection[1];
    double t1y = (disp + len_over_two) / fDirection[1];
    //swap if out of order
    if (t0y > t1y) {
        tmp = t0y;
        t0y = t1y;
        t1y = tmp;
    };

    //update valid interval
    if (t0y > t_min) {
        t_min = t0y;
    };
    if (t1y < t_max) {
        t_max = t1y;
    };

    if (t_min > t_max || t_max < 0 || t_min > fLength) {
        return false;
    }

    //check z planes
    disp = cube[2] - fStartPoint[2];
    double t0z = (disp - len_over_two) / fDirection[2];
    double t1z = (disp + len_over_two) / fDirection[2];
    //swap if out of order
    if (t0z > t1z) {
        tmp = t0z;
        t0z = t1z;
        t1z = tmp;
    };

    //update valid interval
    if (t0z > t_min) {
        t_min = t0z;
    };
    if (t1z < t_max) {
        t_max = t1z;
    };

    //interval is invalid, no intersection
    if (t_min > t_max || t_max < 0 || t_min > fLength) {
        return false;
    }

    distance = t_min;
    return true;
}

void KGNavigableMeshFirstIntersectionFinder::ApplyAction(KGMeshNavigationNode* node)
{
    fHaveIntersection = false;
    fIntersectedElement = nullptr;


    if (node != nullptr) {
        //init stack
        {
            fPreallocatedStack.clear();
            fPreallocatedStackTopPtr = &(fPreallocatedStack[0]);
            fStackSize = 0;
        }

        //push on the first node
        {
            ++fPreallocatedStackTopPtr;          //increment top pointer
            *(fPreallocatedStackTopPtr) = node;  //set pointer
            ++fStackSize;                        //increment size
        }

        do {
            //pop node
            fTempNode = *fPreallocatedStackTopPtr;
            {
                --fPreallocatedStackTopPtr;  //decrement top pointer;
                --fStackSize;
            }

            //check if we are a leaf node or not
            if (fTempNode->HasChildren()) {
                unsigned int n_intersected_children = 0;
                for (unsigned int i = 0; i < 8; i++)  //always 8 chilren in octree
                {
                    KGMeshNavigationNode* child = fTempNode->GetChild(i);
                    //get the child's cube data
                    KGCube<KGMESH_DIM>* child_cube =
                        KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::GetNodeObject(child);

                    //check if the line segment intersects the cube
                    double dist;
                    if (LineSegmentIntersectsCube(*child_cube, dist)) {
                        fOrderedChildren[n_intersected_children] =
                            std::pair<KGMeshNavigationNode*, double>(child, dist);
                        n_intersected_children++;
                    }
                }

                //sort the child nodes by their distance to the begining of the line segment
                if (n_intersected_children > 1) {
                    SortOctreeNodes(n_intersected_children, fOrderedChildren);
                }

                for (unsigned int i = 0; i < n_intersected_children; i++) {
                    //push node
                    {
                        ++fPreallocatedStackTopPtr;                               //increment top pointer
                        *(fPreallocatedStackTopPtr) = fOrderedChildren[i].first;  //set pointer
                        ++fStackSize;                                             //increment size
                    }
                }
            }
            else {
                //get the list of mesh element id's from the node
                KGIdentitySet* element_list =
                    KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet>::GetNodeObject(fTempNode);

                if (element_list != nullptr && element_list->GetSize() != 0) {
                    fHaveIntersection = false;
                    KThreeVector anIntersection;
                    double min_distance = std::numeric_limits<double>::max();
                    double dist;

                    unsigned int n_elem = element_list->GetSize();
                    for (unsigned int i = 0; i < n_elem; i++) {
                        //first check if the line intersects the bounding ball of the element
                        KGBall<KGMESH_DIM> bball = fContainer->GetElementBoundingBall(element_list->GetID(i));
                        double lin_seg_to_bball = LineSegmentDistanceToPoint(KThreeVector(bball.GetCenter()));
                        if (lin_seg_to_bball <= bball.GetRadius()) {
                            bool inter = fContainer->GetElement(element_list->GetID(i))
                                             ->GetMeshElement()
                                             ->NearestIntersection(fStartPoint, fEndPoint, anIntersection);
                            if (inter) {
                                if (fHaveIntersection) {
                                    dist = (anIntersection - fStartPoint).Magnitude();
                                    if (dist < min_distance) {
                                        min_distance = dist;
                                        fFirstIntersection = anIntersection;
                                        fIntersectedElement = fContainer->GetElement(element_list->GetID(i));
                                        fHaveIntersection = true;
                                    }
                                }
                                else {
                                    min_distance = (anIntersection - fStartPoint).Magnitude();
                                    fFirstIntersection = anIntersection;
                                    fIntersectedElement = fContainer->GetElement(element_list->GetID(i));
                                    fHaveIntersection = true;
                                }
                            }
                        }
                    }
                    //short cut if we obtained an itersection
                    if (fHaveIntersection) {
                        return;
                    };
                }
            }
            CheckStackSize();
        } while (fStackSize != 0);
    }
}


}  // namespace KGeoBag
