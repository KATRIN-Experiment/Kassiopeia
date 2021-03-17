#ifndef _Kassiopeia_KSVTKGeneratorPainter_h_
#define _Kassiopeia_KSVTKGeneratorPainter_h_

#include "KField.h"
#include "KSVisualizationMessage.h"
#include "KThreeMatrix.hh"
#include "KThreeVector.hh"
#include "KVTKPainter.h"
#include "KVTKWindow.h"
#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkGlyph3D.h"
#include "vtkLookupTable.h"
#include "vtkNamedColors.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkSmartPointer.h"

#include <vector>

namespace Kassiopeia
{
class KSElectricField;
class KSMagneticField;
class KSRootElectricField;
class KSRootMagneticField;
class KSParticle;

class KSVTKGeneratorPainter : public katrin::KVTKPainter
{
  public:
    KSVTKGeneratorPainter();
    ~KSVTKGeneratorPainter() override;

    void Render() override;
    void Display() override;
    void Write() override;

    ;
    K_SET(std::string, Path);
    K_SET(std::string, File);
    K_SET(int, NumSamples);
    K_SET(double, ScaleFactor);
    K_SET(std::string, ColorVariable)

  public:
    void AddGenerator(const std::string& aGenerator);
    void AddColor(const KGeoBag::KThreeVector& aColor);

    void AddElectricField(KSElectricField* aField);
    void AddMagneticField(KSMagneticField* aField);

  protected:
    static double GetScalarValue(KSParticle& aParticle, std::string aName);
    static double GetScalarValue(const KGeoBag::KThreeVector& aVector, std::string aName);
    static double GetScalarValue(const KGeoBag::KThreeMatrix& aTensor, std::string aName);

  private:
    KSRootElectricField* fElectricField;
    KSRootMagneticField* fMagneticField;

    vtkSmartPointer<vtkGlyph3D> fGlyph;
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkUnsignedCharArray> fColors;
    vtkSmartPointer<vtkDoubleArray> fScalars;
    vtkSmartPointer<vtkDoubleArray> fVectors;
    vtkSmartPointer<vtkPolyData> fData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;
    vtkSmartPointer<vtkLookupTable> fColorTable;
    vtkSmartPointer<vtkNamedColors> fNamedColors;
    std::vector<std::string> fGenerators;
};

inline void KSVTKGeneratorPainter::AddGenerator(const std::string& aGenerator)
{
    fGenerators.push_back(aGenerator);
}

inline void KSVTKGeneratorPainter::AddColor(const KGeoBag::KThreeVector& aColor)
{
    // sets color for last generator that was added
    if (!fGenerators.empty()) {
        std::string tGenerator = fGenerators.back();
        vismsg_debug("generator painter <" << GetName() << "> uses color <" << aColor << "> for generator <"
                                           << tGenerator << ">" << eom);
        KGeoBag::KThreeVector tColorRGB =
            aColor /
            255.;  // VTK assumes RGB floats to be in range [0,1] but it is more convenient to use [0,255] in XML input
        fNamedColors->SetColor(tGenerator, tColorRGB.X(), tColorRGB.Y(), tColorRGB.Z());
    }
}

}  // namespace Kassiopeia

#endif
