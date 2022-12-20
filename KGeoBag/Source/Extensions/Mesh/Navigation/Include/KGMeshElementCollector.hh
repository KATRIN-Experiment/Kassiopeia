#ifndef KGMeshElementCollector_HH__
#define KGMeshElementCollector_HH__


#include "KGCore.hh"
#include "KGMesh.hh"
#include "KGMeshElement.hh"
#include "KGNavigableMeshElement.hh"
#include "KGNavigableMeshElementContainer.hh"

#include "KThreeVector.hh"

namespace KGeoBag
{

/*
*
*@file KGMeshElementCollector.hh
*@class KGMeshElementCollector
*@brief collects mesh elements from various surfaces into global
* coordinate system and inserts them into a container
* somewhat based on KGBEMConverter, but it is only interested in spatial/shape data
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jul  9 20:40:56 EDT 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KGMeshElementCollector : public KGVisitor, public KGSurface::Visitor, public KGSpace::Visitor
{
  public:
    KGMeshElementCollector();
    ~KGMeshElementCollector() override;

    void SetMeshElementContainer(KGNavigableMeshElementContainer* aContainer)
    {
        fMeshContainer = aContainer;
    }

    void SetSystem(const katrin::KThreeVector& anOrigin, const katrin::KThreeVector& aXAxis,
                   const katrin::KThreeVector& aYAxis, const katrin::KThreeVector& aZAxis);
    const katrin::KThreeVector& GetOrigin() const;
    const katrin::KThreeVector& GetXAxis() const;
    const katrin::KThreeVector& GetYAxis() const;
    const katrin::KThreeVector& GetZAxis() const;

    void VisitSurface(KGSurface* aSurface) override;
    void VisitSpace(KGSpace* aSpace) override;

  protected:
    katrin::KThreeVector LocalToInternal(const katrin::KThreeVector& aVector);
    void Add(KGMeshData* aData);

    void PreCollectionAction(KGMeshData* aData)
    {
        this->PreCollectionActionExecute(aData);
    };
    void PostCollectionAction(KGNavigableMeshElement* element)
    {
        this->PostCollectionActionExecute(element);
    };

  protected:
    virtual void PreCollectionActionExecute(KGMeshData* /*aData */);
    virtual void PostCollectionActionExecute(KGNavigableMeshElement* /*element */);

    enum ElementTypes
    {
        eTriangle = 0,
        eRectangle = 1,
        eWire = 2
    };

    KGNavigableMeshElementContainer* fMeshContainer;

    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;

    katrin::KThreeVector fCurrentOrigin;
    katrin::KThreeVector fCurrentXAxis;
    katrin::KThreeVector fCurrentYAxis;
    katrin::KThreeVector fCurrentZAxis;

    KGSurface* fCurrentSurface;
    KGSpace* fCurrentSpace;

    int fCurrentElementType;
};

}  // namespace KGeoBag

#endif /* end of include guard: KGMeshElementCollector_HH__ */
