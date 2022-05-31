#ifndef KGGEOMETRYPRINTER_HH_
#define KGGEOMETRYPRINTER_HH_

#include "KGAnnulusSurface.hh"
#include "KGConeSpace.hh"
#include "KGConeSurface.hh"
#include "KGConicalWireArraySurface.hh"
#include "KGCore.hh"
#include "KGCutConeSpace.hh"
#include "KGCutConeSurface.hh"
#include "KGCutConeTubeSpace.hh"
#include "KGCylinderSpace.hh"
#include "KGCylinderSurface.hh"
#include "KGCylinderTubeSpace.hh"
#include "KGDiskSurface.hh"
#include "KGExtrudedArcSegmentSurface.hh"
#include "KGExtrudedCircleSpace.hh"
#include "KGExtrudedCircleSurface.hh"
#include "KGExtrudedLineSegmentSurface.hh"
#include "KGExtrudedPolyLineSurface.hh"
#include "KGExtrudedPolyLoopSpace.hh"
#include "KGExtrudedPolyLoopSurface.hh"
#include "KGFlattenedCircleSurface.hh"
#include "KGFlattenedPolyLoopSurface.hh"
#include "KGRodSpace.hh"
#include "KGRodSurface.hh"
#include "KGRotatedArcSegmentSpace.hh"
#include "KGRotatedArcSegmentSurface.hh"
#include "KGRotatedCircleSpace.hh"
#include "KGRotatedCircleSurface.hh"
#include "KGRotatedLineSegmentSpace.hh"
#include "KGRotatedLineSegmentSurface.hh"
#include "KGRotatedPolyLineSpace.hh"
#include "KGRotatedPolyLineSurface.hh"
#include "KGRotatedPolyLoopSpace.hh"
#include "KGRotatedPolyLoopSurface.hh"
#include "KGShellArcSegmentSurface.hh"
#include "KGShellCircleSurface.hh"
#include "KGShellLineSegmentSurface.hh"
#include "KGShellPolyLineSurface.hh"
#include "KGShellPolyLoopSurface.hh"
#include "KPainter.h"
#include "KWindow.h"

#include <deque>
#include <vector>

namespace KGeoBag
{

class KGGeometryPrinter :
    public katrin::KPainter,

    public KGSurface::Visitor,
    public KGAnnulusSurface::Visitor,
    public KGConeSurface::Visitor,
    public KGCutConeSurface::Visitor,
    public KGCylinderSurface::Visitor,
    public KGDiskSurface::Visitor,
    public KGRodSurface::Visitor,

    public KGSpace::Visitor,
    public KGConeSpace::Visitor,
    public KGCutConeSpace::Visitor,
    public KGCutConeTubeSpace::Visitor,
    public KGCylinderSpace::Visitor,
    public KGCylinderTubeSpace::Visitor,
    public KGRodSpace::Visitor,
#if 0
    public KGFlattenedCircleSurface::Visitor,
    public KGFlattenedPolyLoopSurface::Visitor,
    public KGRotatedLineSegmentSurface::Visitor,
    public KGRotatedArcSegmentSurface::Visitor,
    public KGRotatedPolyLineSurface::Visitor,
    public KGRotatedCircleSurface::Visitor,
    public KGRotatedPolyLoopSurface::Visitor,
    public KGShellLineSegmentSurface::Visitor,
    public KGShellArcSegmentSurface::Visitor,
    public KGShellPolyLineSurface::Visitor,
    public KGShellPolyLoopSurface::Visitor,
    public KGShellCircleSurface::Visitor,
    public KGExtrudedLineSegmentSurface::Visitor,
    public KGExtrudedArcSegmentSurface::Visitor,
    public KGExtrudedPolyLineSurface::Visitor,
    public KGExtrudedCircleSurface::Visitor,
    public KGExtrudedPolyLoopSurface::Visitor,
    public KGConicalWireArraySurface::Visitor,
    public KGRotatedLineSegmentSpace::Visitor,
    public KGRotatedArcSegmentSpace::Visitor,
    public KGRotatedPolyLineSpace::Visitor,
    public KGRotatedCircleSpace::Visitor,
    public KGRotatedPolyLoopSpace::Visitor,
    public KGExtrudedCircleSpace::Visitor,
    public KGExtrudedPolyLoopSpace::Visitor,
#endif
    public KGVisitor
{
    class Private;

  public:
    KGGeometryPrinter();
    ~KGGeometryPrinter() override;

  public:
    void Render() override;
    void Display() override;
    void Write() override;

  protected:
    void WriteCSV();
    /*
    template<typename KeyT>
    std::string ColorHash(const KeyT& aValue);
    template<>
    std::string ColorHash(const double& aValue);
    template<>
    std::string ColorHash(const int& aValue);

    template <template <typename, typename...> class Container, typename T>
    std::string ColorizeAndJoin(const Container<T>& aSequence, std::string aSeparator = " ");
    std::string Colorize(const std::string& aValue);
    std::string Colorize(const double& aValue);
    std::string Colorize(const KTwoVector& aValue);
    std::string Colorize(const katrin::KThreeVector& aValue);

    template<typename T>
    void Dump(T* aTagged);

    std::string Indent(int aOffset = 0);
*/
  public:
    void SetFile(const std::string& aName);
    const std::string& GetFile() const;
    void SetPath(const std::string& aPath);

    void SetStream(std::ostream& aStream);
    void SetUseColors(bool aFlag = true);
    void SetWriteJSON(bool aFlag = true);
    void SetWriteXML(bool aFlag = true);
    void SetWriteDOT(bool aFlag = true);

    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

    void SetWindow(katrin::KWindow*) override{};
    void ClearWindow(katrin::KWindow*) override{};

  protected:
    void WriteGraphViz(std::ostream& aStream, bool with_tags = false) const;

  private:
    std::string fFile;
    std::string fPath;
    bool fWriteJSON;
    bool fWriteXML;
    bool fWriteDOT;
    bool fUseColors;

    std::ostream* fStream;

    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;

    std::vector<KGSurface*> fVisitedSurfaces;
    std::vector<KGSpace*> fVisitedSpaces;

    Private* fPrivate;

    //****************
    //surface visitors
    //****************

  protected:
    void VisitSurface(KGSurface* aSurface) override;
    void VisitAnnulusSurface(KGAnnulusSurface* aAnnulusSurface) override;
    void VisitConeSurface(KGConeSurface* aConeSurface) override;
    void VisitCutConeSurface(KGCutConeSurface* aCutConeSurface) override;
    void VisitCylinderSurface(KGCylinderSurface* aCylinderSurface) override;
    void VisitDiskSurface(KGDiskSurface* aDiskSurface) override;
    void VisitWrappedSurface(KGRodSurface* aRodSurface) override;

    //**************
    //space visitors
    //**************

  protected:
    void VisitSpace(KGSpace* aSpace) override;
    void VisitConeSpace(KGConeSpace* aConeSpace) override;
    void VisitCutConeSpace(KGCutConeSpace* aCutConeSpace) override;
    void VisitCutConeTubeSpace(KGCutConeTubeSpace* aCutConeTubeSpace) override;
    void VisitCylinderSpace(KGCylinderSpace* aCylinderSpace) override;
    void VisitCylinderTubeSpace(KGCylinderTubeSpace* aGCylinderTubeSpace) override;
    void VisitWrappedSpace(KGRodSpace* aRodSpace) override;

  private:
    void LocalToGlobal(const katrin::KThreeVector& aLocal, katrin::KThreeVector& aGlobal);

  private:
    KGSpace* fCurrentSpace;
    KGSurface* fCurrentSurface;

    katrin::KThreeVector fCurrentOrigin;
    katrin::KThreeVector fCurrentXAxis;
    katrin::KThreeVector fCurrentYAxis;
    katrin::KThreeVector fCurrentZAxis;

    bool fIgnore;
};

}  // namespace KGeoBag

#endif
