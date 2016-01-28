#include "KGNavigableMeshProximityCheck.hh"



namespace KGeoBag
{

const double KGNavigableMeshProximityCheck::fCubeLengthToRadius = 0.866025403784438597;

KGNavigableMeshProximityCheck::KGNavigableMeshProximityCheck():
    fContainer(NULL),
    fDefaultStackSize(512),
    fStackReallocateLimit(384)
{
    fPreallocatedStack.resize(fDefaultStackSize, NULL);
};

KGNavigableMeshProximityCheck::~KGNavigableMeshProximityCheck(){};

void
KGNavigableMeshProximityCheck::SetPointAndRadius(const KThreeVector& point, double radius)
{
    fPoint = point;
    fRadius = radius;
    fBall.SetRadius(fRadius);
    fBall.SetCenter( KGPoint<KGMESH_DIM>(fPoint) );
    fSphereIntersectsMesh = false;
}

bool
KGNavigableMeshProximityCheck::BallIntersectsCube(const KGBall<KGMESH_DIM>& ball, const KGCube<KGMESH_DIM>& cube) const
{
    //uses 'Avro's algorithm to check if axis aligned bounding box intersects a sphere
    //as specified in chapter 11.12 of Geometric Tools for Computer Graphics
    //by Schneider & Eberly

    KGPoint<KGMESH_DIM> ball_center = ball.GetCenter();
    KGPoint<KGMESH_DIM> cube_center = cube.GetCenter();
    double len_over_two = cube.GetLength()/2.0;
    KGPoint<KGMESH_DIM> delta; delta[0] = len_over_two; delta[1] = len_over_two; delta[2] = len_over_two;
    KGPoint<KGMESH_DIM> lower_corner = cube_center - delta;
    KGPoint<KGMESH_DIM> upper_corner = cube_center + delta;

    double dist2 = 0;
    for(unsigned int i=0; i<KGMESH_DIM; i++)
    {
        if(ball_center[i] < lower_corner[i])
        {
            double del = ball_center[i] - lower_corner[i];
            dist2 += del*del;
        }
        else if(ball_center[i] > upper_corner[i])
        {
            double del = ball_center[i] - upper_corner[i];
            dist2 += del*del;
        }
    }

    double ball_radius = ball.GetRadius();

    if(dist2 <= ball_radius*ball_radius)
    {
        return true;
    }

    return false;
}

bool
KGNavigableMeshProximityCheck::CubeEnclosedByBall(const KGBall<KGMESH_DIM>& ball, const KGCube<KGMESH_DIM>& cube) const
{
    //this is not a strict check (we only compare the two bounding balls)
    //if bounding ball of the cube is inside the test ball, then it is definitely contained
    //by the test ball, but it can also be contained even if its bounding ball isn't entirely
    //contained (we don't check for this case)

    KGPoint<KGMESH_DIM> ball_center = ball.GetCenter();
    KGPoint<KGMESH_DIM> cube_center = cube.GetCenter();
    double cube_radius = (cube.GetLength())*fCubeLengthToRadius;

    double dist2 = (ball_center - cube_center).Magnitude();

    if(dist2 < (ball.GetRadius() - cube_radius) )
    {
        return true;
    }

    return false;
}


void
KGNavigableMeshProximityCheck::ApplyAction(KGMeshNavigationNode* node)
{
    fSphereIntersectsMesh = false;
    fLeafNodes.clear();
    KGMeshNavigationNode* tempNode;

    if(node != NULL)
    {
        //init stack
        {
            fPreallocatedStack.clear();
            fPreallocatedStackTopPtr = &(fPreallocatedStack[0]);
            fStackSize = 0;
        }

        //push on the first node
        {
            ++fPreallocatedStackTopPtr; //increment top pointer
            *(fPreallocatedStackTopPtr) = node; //set pointer
            ++fStackSize;//increment size
        }

        do
        {
            //pop node
            tempNode = *fPreallocatedStackTopPtr;
            {
                --fPreallocatedStackTopPtr; //decrement top pointer;
                --fStackSize;
            }

            //check if we are a leaf node or not
            if( tempNode->HasChildren() )
            {
                //determine which child intersect the ball
                for(unsigned int i=0; i<8; i++) //always 8 children in octree
                {
                    KGMeshNavigationNode* child = tempNode->GetChild(i);

                    //get child node cube
                    KGCube<KGMESH_DIM>* cube =
                    KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM> >::GetNodeObject(child);

                    //check if the bounding ball intersects the child's cube
                    if( BallIntersectsCube(fBall, *cube) )
                    {
                        //push child node
                        {
                            ++fPreallocatedStackTopPtr; //increment top pointer
                            *(fPreallocatedStackTopPtr) = child; //set pointer
                            ++fStackSize;//increment size
                        }
                    }
                }
            }
            else
            {
                //now we check if this leaf node has any elements
                KGIdentitySet* element_list =
                KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet >::GetNodeObject(tempNode);

                if(element_list != NULL)
                {
                    if( element_list->GetSize() != 0)
                    {
                        fLeafNodes.push_back(tempNode);
                    }
                }
            }
            CheckStackSize();
        }
        while( fStackSize != 0 );


        //no leaf nodes which contain elements intersect the bounding ball
        //so there is no possible intersection with the mesh
        if(fLeafNodes.size() == 0)
        {
            fSphereIntersectsMesh = false;
            return;
        }

        //now we have a list of all the leaf nodes which intersect the bounding ball
        //and also contain mesh elements, now we need to check if any of these elements are inside the ball
        for(unsigned int i=0; i<fLeafNodes.size(); i++)
        {
            //get the cube
            KGCube<KGMESH_DIM>* cube =
            KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM> >::GetNodeObject(fLeafNodes[i]);

            //first we can perform a quick culling if there is a cube that
            //contains elements and lies entirely within the bounding ball
            if( CubeEnclosedByBall(fBall, *cube) )
            {
                fSphereIntersectsMesh = true;
                return;
            }
        }

        //culling didn't find a quick intersection, so we need to loop over the leaf nodes
        //and inspect their elements direcly to see if they intersect the ball
        for(unsigned int i=0; i<fLeafNodes.size(); i++)
        {
            //get the element id set of this node
            KGIdentitySet* element_list =
            KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet >::GetNodeObject(fLeafNodes[i]);

            unsigned int n_elem = element_list->GetSize();
            for(unsigned int j=0; j<n_elem; j++)
            {
                double edist = fContainer->GetElement(element_list->GetID(j))->GetMeshElement()->NearestDistance(fPoint);
                if(edist < fRadius)
                {
                    fSphereIntersectsMesh = true;
                    return;
                }
            }
        }
    }
}


}
