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
    void SetSystem(const KThreeVector& anOrigin, const KThreeVector& anXAxis, const KThreeVector& aYAxis,
                   const KThreeVector& aZAxis);
    const KThreeVector& GetOrigin() const;
    const KThreeVector& GetXAxis() const;
    const KThreeVector& GetYAxis() const;
    const KThreeVector& GetZAxis() const;

    KThreeVector GlobalToInternalPosition(const KThreeVector& aPosition);
    KThreeVector GlobalToInternalVector(const KThreeVector& aVector);
    KThreeVector InternalToGlobalPosition(const KThreeVector& aVector);
    KThreeVector InternalToGlobalVector(const KThreeVector& aVector);
    KThreeMatrix InternalTensorToGlobal(const KEMField::KGradient& aGradient);

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

    KThreeVector fOrigin;
    KThreeVector fXAxis;
    KThreeVector fYAxis;
    KThreeVector fZAxis;

    KThreeVector fCurrentOrigin;
    KThreeVector fCurrentXAxis;
    KThreeVector fCurrentYAxis;
    KThreeVector fCurrentZAxis;

    KGExtendedSpace<KGElectromagnet>* fCurrentElectromagnetSpace;
    KGExtendedSurface<KGElectromagnet>* fCurrentElectromagnetSurface;
};

}  // namespace KGeoBag

#endif /* KGELECTROMAGNETCONVERTER_DEF */
