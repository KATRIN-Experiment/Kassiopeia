#ifndef KGELECTROMAGNETCONVERTER_DEF
#define KGELECTROMAGNETCONVERTER_DEF

#include "KGCylinderSurface.hh"
#include "KGCylinderTubeSpace.hh"
#include "KGElectromagnet.hh"
#include "KGRodSpace.hh"
#include "KTagged.h"
#include "KThreeMatrix_KEMField.hh"

#include <KTextFile.h>

namespace KGeoBag
{
class KGElectromagnetConverter :
    virtual public KGVisitor,
    virtual public KGSpace::Visitor,
    virtual public KGSurface::Visitor,
    virtual public KGExtendedSpace<KGElectromagnet>::Visitor,
    virtual public KGExtendedSurface<KGElectromagnet>::Visitor,
    virtual public KGRodSpace::Visitor,
    virtual public KGCylinderTubeSpace::Visitor,
    virtual public KGCylinderSurface::Visitor
{
  public:
    KGElectromagnetConverter();
    ~KGElectromagnetConverter() override;

  public:
    void SetElectromagnetContainer(KEMField::KElectromagnetContainer* aContainer)
    {
        fElectromagnetContainer = aContainer;
        return;
    }

    void SetDumpMagfield3ToFile(const std::string& aDirectory, const std::string& aFileName);

  protected:
    KEMField::KElectromagnetContainer* fElectromagnetContainer;

  public:
    void SetSystem(const katrin::KThreeVector& anOrigin, const katrin::KThreeVector& anXAxis, const katrin::KThreeVector& aYAxis,
                   const katrin::KThreeVector& aZAxis);
    const katrin::KThreeVector& GetOrigin() const;
    const katrin::KThreeVector& GetXAxis() const;
    const katrin::KThreeVector& GetYAxis() const;
    const katrin::KThreeVector& GetZAxis() const;

    katrin::KThreeVector GlobalToInternalPosition(const katrin::KThreeVector& aPosition);
    katrin::KThreeVector GlobalToInternalVector(const katrin::KThreeVector& aVector);
    katrin::KThreeVector InternalToGlobalPosition(const katrin::KThreeVector& aVector);
    katrin::KThreeVector InternalToGlobalVector(const katrin::KThreeVector& aVector);
    katrin::KThreeMatrix InternalTensorToGlobal(const KEMField::KGradient& aGradient);

    void VisitSpace(KGSpace* aSpace) override;
    void VisitSurface(KGSurface* aSurface) override;

    void VisitExtendedSpace(KGExtendedSpace<KGElectromagnet>* electromagnetSpace) override;
    void VisitExtendedSurface(KGExtendedSurface<KGElectromagnet>* electromagnetSurface) override;

    void VisitWrappedSpace(KGRodSpace* rod) override;
    void VisitCylinderSurface(KGCylinderSurface* cylinder) override;
    void VisitCylinderTubeSpace(KGCylinderTubeSpace* cylinderTube) override;

  private:
    void Clear();

    katrin::KTextFile* fMagfield3File;

    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;

    katrin::KThreeVector fCurrentOrigin;
    katrin::KThreeVector fCurrentXAxis;
    katrin::KThreeVector fCurrentYAxis;
    katrin::KThreeVector fCurrentZAxis;

    KGExtendedSpace<KGElectromagnet>* fCurrentElectromagnetSpace;
    KGExtendedSurface<KGElectromagnet>* fCurrentElectromagnetSurface;
};

}  // namespace KGeoBag

#endif /* KGELECTROMAGNETCONVERTER_DEF */
