#ifndef _Kassiopeia_KSVTKTrackTerminatorPainter_h_
#define _Kassiopeia_KSVTKTrackTerminatorPainter_h_

#include "KSVisualizationMessage.h"
#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KField.h"
#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkLookupTable.h"
#include "vtkNamedColors.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkSmartPointer.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

class KSVTKTrackTerminatorPainter : public KVTKPainter
{
  public:
    KSVTKTrackTerminatorPainter();
    ~KSVTKTrackTerminatorPainter();

    void Render();
    void Display();
    void Write();

    ;
    K_SET(std::string, File);
    K_SET(std::string, Path);
    K_SET(std::string, OutFile);
    K_SET(std::string, PointObject);
    K_SET(std::string, PointVariable);
    K_SET(std::string, TerminatorObject);
    K_SET(std::string, TerminatorVariable);
    K_SET(int, PointSize)

  public:
    void AddTerminator(const std::string& aTerminator);
    void AddColor(const KThreeVector& aColor);

  private:
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkCellArray> fVertices;
    vtkSmartPointer<vtkUnsignedCharArray> fColors;
    vtkSmartPointer<vtkPolyData> fData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;
    vtkSmartPointer<vtkNamedColors> fNamedColors;
    std::vector<std::string> fTerminators;
};

void KSVTKTrackTerminatorPainter::AddTerminator(const std::string& aTerminator)
{
    fTerminators.push_back(aTerminator);
}

void KSVTKTrackTerminatorPainter::AddColor(const KThreeVector& aColor)
{
    // sets color for last terminator that was added
    if (!fTerminators.empty()) {
        std::string tTerminator = fTerminators.back();
        vismsg_debug("track terminator painter <" << GetName() << "> uses color <" << aColor << "> for terminator <"
                                                  << tTerminator << ">" << eom);
        KThreeVector tColorRGB =
            aColor /
            255.;  // VTK assumes RGB floats to be in range [0,1] but it is more convenient to use [0,255] in XML input
        fNamedColors->SetColor(tTerminator, tColorRGB.X(), tColorRGB.Y(), tColorRGB.Z());
    }
}

}  // namespace Kassiopeia

#endif
