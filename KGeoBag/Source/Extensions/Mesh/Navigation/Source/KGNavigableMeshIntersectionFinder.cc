#include "KGNavigableMeshIntersectionFinder.hh"

using katrin::KThreeVector;

namespace KGeoBag
{

KGNavigableMeshIntersectionFinder::KGNavigableMeshIntersectionFinder() :
    fContainer(nullptr){

    };

KGNavigableMeshIntersectionFinder::~KGNavigableMeshIntersectionFinder() = default;
;

void KGNavigableMeshIntersectionFinder::SetLineSegment(const KThreeVector& start, const KThreeVector& end)
{
    fStartPoint = start;
    fEndPoint = end;
    fDirection = fEndPoint - fStartPoint;
    fLength = fDirection.Magnitude();
    fDirection = fDirection.Unit();
    fHaveIntersection = false;
}


bool KGNavigableMeshIntersectionFinder::HasIntersectionWithMesh() const
{
    return fHaveIntersection;
};


void KGNavigableMeshIntersectionFinder::GetIntersectedMeshElements(
    std::vector<const KGNavigableMeshElement*>* intersected_mesh_elements) const
{
    *intersected_mesh_elements = fIntersectedElements;
}

void KGNavigableMeshIntersectionFinder::GetIntersections(std::vector<KThreeVector>* intersections) const
{
    *intersections = fIntersections;
};


void KGNavigableMeshIntersectionFinder::NearestPointOnLineSegment(const KThreeVector& aPoint, KThreeVector& aNearest,
                                                                  double& t) const
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

double KGNavigableMeshIntersectionFinder::LineSegmentDistanceToPoint(const KThreeVector& aPoint) const
{
    KThreeVector del = fStartPoint - aPoint;
    double t = -1.0 * (del * fDirection);
    if (t < 0.) {
        return del.Magnitude();
    }
    if (t > fLength) {
        return (fEndPoint - aPoint).Magnitude();
    }
    return (del + t * fDirection).Magnitude();
}

bool KGNavigableMeshIntersectionFinder::LineSegmentIntersectsCube(const KGCube<KGMESH_DIM>* cube,
                                                                  double& distance) const
{
    distance = 1e30;
    //uses 'slab clipping' to check if line segment intersects the cube
    //as specified in chapter 11.2 of Geometric Tools for Computer Graphics
    //by Schneider & Eberly

    KThreeVector lower_corner(cube->GetCorner(0));
    KThreeVector upper_corner(cube->GetCorner(7));

    //compute displacement from upper and lower corners
    KThreeVector lower_diplacement = lower_corner - fStartPoint;
    KThreeVector upper_diplacement = upper_corner - fStartPoint;

    double t_min = 0;
    double t_max = fLength;
    double tmp;

    //check x planes
    double t0x = lower_diplacement.X() / fDirection.X();
    double t1x = upper_diplacement.X() / fDirection.X();
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

    //check y planes
    double t0y = lower_diplacement.Y() / fDirection.Y();
    double t1y = upper_diplacement.Y() / fDirection.Y();
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

    //check z planes
    double t0z = lower_diplacement.Z() / fDirection.Z();
    double t1z = upper_diplacement.Z() / fDirection.Z();
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
    if (t_min > t_max) {
        return false;
    }

    distance = t_min;
    return true;
}


void KGNavigableMeshIntersectionFinder::ApplyAction(KGMeshNavigationNode* node)
{
    fHaveIntersection = false;
    fIntersections.clear();
    fIntersectedElements.clear();
    fCheckedElements.clear();

    if (node != nullptr) {
        //make sure the stacks are empty
        fNodeStack = std::stack<KGMeshNavigationNode*>();

        //push on the first node
        fNodeStack.push(node);

        do {
            //check if we are a leaf node or not
            if (fNodeStack.top()->HasChildren()) {
                unsigned int n_children = fNodeStack.top()->GetNChildren();
                fTempNode = fNodeStack.top();
                fNodeStack.pop();  //non-leaf node, pop it off the stack

                fOrderedChildren.clear();
                for (unsigned int i = 0; i < n_children; i++) {
                    KGMeshNavigationNode* child = fTempNode->GetChild(i);

                    //get the child's cube data
                    KGCube<KGMESH_DIM>* child_cube =
                        KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::GetNodeObject(child);
                    KThreeVector cube_center((*child_cube)[0], (*child_cube)[1], (*child_cube)[2]);

                    //check if the child's cube intersects the line segment
                    KThreeVector nearest_point;
                    double dist;
                    NearestPointOnLineSegment(cube_center, nearest_point, dist);

                    //the ratio of the cube's diagonal to its side length is sqrt(3/4)
                    //since we use the square distance to compare, this gives us the factor of 0.75
                    if ((nearest_point - cube_center).MagnitudeSquared() < 0.75 * child_cube->GetLength()) {
                        //check if the line segment intersects the cube
                        double dist;
                        if (LineSegmentIntersectsCube(child_cube, dist)) {
                            fOrderedChildren.emplace_back(child, dist);
                        }
                    }
                }

                //sort the child nodes by their distance to the begining of the line segment
                std::sort(fOrderedChildren.begin(), fOrderedChildren.end(), fOrderingPredicate);

                for (auto& i : fOrderedChildren) {
                    fNodeStack.push(i.first);
                }
            }
            else {
                //get the list of mesh element id's from the node
                KGIdentitySet* element_list =
                    KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet>::GetNodeObject(fNodeStack.top());

                if (element_list != nullptr && element_list->GetSize() != 0) {
                    KThreeVector anIntersection;
                    unsigned int n_elem = element_list->GetSize();
                    for (unsigned int i = 0; i < n_elem; i++) {
                        //have to check if we have seen this element before since, the set lists
                        //for each leaf node can overlap
                        KGNavigableMeshElement* current_elem = fContainer->GetElement(element_list->GetID(i));
                        std::set<const KGNavigableMeshElement*>::iterator check_it;
                        check_it = fCheckedElements.find(current_elem);

                        //have not checked this element before
                        if (check_it == fCheckedElements.end()) {
                            fCheckedElements.insert(current_elem);
                            //first check if the line intersects the bounding ball of the element
                            KGBall<KGMESH_DIM> bball = fContainer->GetElementBoundingBall(element_list->GetID(i));
                            double lin_seg_to_bball = LineSegmentDistanceToPoint(KThreeVector(bball.GetCenter()));
                            if (lin_seg_to_bball < bball.GetRadius()) {
                                //now compute possible intersection
                                bool inter = current_elem->GetMeshElement()->NearestIntersection(fStartPoint,
                                                                                                 fEndPoint,
                                                                                                 anIntersection);
                                if (inter) {
                                    fHaveIntersection = true;
                                    fIntersections.push_back(anIntersection);
                                    fIntersectedElements.push_back(fContainer->GetElement(element_list->GetID(i)));
                                }
                            }
                        }
                    }
                }
                fNodeStack.pop();
            }
        } while (!fNodeStack.empty());
    }
}

}  // namespace KGeoBag
