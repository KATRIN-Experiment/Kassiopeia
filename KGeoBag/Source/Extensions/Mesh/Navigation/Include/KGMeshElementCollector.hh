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

    void SetSystem(const KGeoBag::KThreeVector& anOrigin, const KGeoBag::KThreeVector& aXAxis,
                   const KGeoBag::KThreeVector& aYAxis, const KGeoBag::KThreeVector& aZAxis);
    const KGeoBag::KThreeVector& GetOrigin() const;
    const KGeoBag::KThreeVector& GetXAxis() const;
    const KGeoBag::KThreeVector& GetYAxis() const;
    const KGeoBag::KThreeVector& GetZAxis() const;

    void VisitSurface(KGSurface* aSurface) override;
    void VisitSpace(KGSpace* aSpace) override;

  protected:
    KGeoBag::KThreeVector LocalToInternal(const KGeoBag::KThreeVector& aVector);
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

    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

    KGeoBag::KThreeVector fCurrentOrigin;
    KGeoBag::KThreeVector fCurrentXAxis;
    KGeoBag::KThreeVector fCurrentYAxis;
    KGeoBag::KThreeVector fCurrentZAxis;

    KGSurface* fCurrentSurface;
    KGSpace* fCurrentSpace;

    int fCurrentElementType;
};

}  // namespace KGeoBag

#endif /* end of include guard: KGMeshElementCollector_HH__ */
