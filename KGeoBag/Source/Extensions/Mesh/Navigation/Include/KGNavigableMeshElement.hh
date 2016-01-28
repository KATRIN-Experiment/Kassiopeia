#ifndef KGNavigableMeshElement_HH__
#define KGNavigableMeshElement_HH__

#include "KGCore.hh"
#include "KGMeshElement.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshWire.hh"

//only three element types supported
#define KGMESH_TRIANGLE_ID 0
#define KGMESH_RECTANGLE_ID 1
#define KGMESH_WIRE_ID 2

namespace KGeoBag
{

class KGNavigableMeshElement
{
    public:

        KGNavigableMeshElement();
        virtual ~KGNavigableMeshElement();

        void SetMeshElement(KGMeshTriangle* triangle);
        void SetMeshElement(KGMeshRectangle* rectange);
        void SetMeshElement(KGMeshWire* wire);

        short GetMeshElementType();
        const KGMeshElement* GetMeshElement() const;

        void SetID(unsigned int id){fID = id;};
        unsigned int GetID() const {return fID;};

    protected:

        short fType; //mesh element type
        KGMeshElement* fMeshElement; //owned pointer to mesh element data
        unsigned int fID; //unique id used for associating the mesh element with its parent space/surface

};

}//end of KGeoBag


#endif /* end of include guard: KGNavigableMeshElement_HH__ */
