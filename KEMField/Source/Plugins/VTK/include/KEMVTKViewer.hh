#ifndef KEMVTKVIEWER_DEF
#define KEMVTKVIEWER_DEF

#include "KSurfaceContainer.hh"
#include "KThreeVector_KEMField.hh"
#include "KTypeManipulation.hh"

#include <limits>
#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCleanPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkDiskSource.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkLine.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkProperty.h>
#include <vtkQuad.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkShortArray.h>
#include <vtkSmartPointer.h>
#include <vtkStripper.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>


namespace KEMField
{

/**
   * @class KEMVTKViewer
   *
   * @brief A class for rendering EM geometries with VTK.
   *
   * KEMVTKViewer is a class for rendering EM geometries with VTK.  The result
   * can be saved to a file, or displayed on the screen.
   *
   * @author T.J. Corona
   */

class KEMVTKViewer : public KSurfaceAction<KEMVTKViewer>
{
  public:
    KEMVTKViewer(KSurfaceContainer& aSurfaceContainer);
    ~KEMVTKViewer() = default;

    void GenerateGeometryFile(const std::string& fileName = "EMSurfaces.vtp");

    void ViewGeometry();

    template<typename Surface> void PerformAction(Type2Type<Surface>);

  private:
    void SetSurfacePrimitive(KSurfacePrimitive* sP)
    {
        fSurfacePrimitive = sP;
    }

    template<typename Surface> void AddIfTriangleSymmetryGroup(Int2Type<false>) {}

    template<typename Surface> void AddIfTriangleSymmetryGroup(Int2Type<true>);

    template<typename Surface> void AddIfRectangleSymmetryGroup(Int2Type<false>) {}

    template<typename Surface> void AddIfRectangleSymmetryGroup(Int2Type<true>);

    template<typename Surface> void AddIfLineSegmentSymmetryGroup(Int2Type<false>) {}

    template<typename Surface> void AddIfLineSegmentSymmetryGroup(Int2Type<true>);

    template<typename Surface> void AddIfConicSectionSymmetryGroup(Int2Type<false>) {}

    template<typename Surface> void AddIfConicSectionSymmetryGroup(Int2Type<true>);

    template<typename Surface> void AddIfRingSymmetryGroup(Int2Type<false>) {}

    template<typename Surface> void AddIfRingSymmetryGroup(Int2Type<true>);

    template<typename Surface> void AddIfTriangle(Int2Type<false>) {}

    template<typename Surface> void AddIfTriangle(Int2Type<true>);

    template<typename Surface> void AddIfRectangle(Int2Type<false>) {}

    template<typename Surface> void AddIfRectangle(Int2Type<true>);

    template<typename Surface> void AddIfLineSegment(Int2Type<false>) {}

    template<typename Surface> void AddIfLineSegment(Int2Type<true>);

    template<typename Surface> void AddIfConicSection(Int2Type<false>) {}

    template<typename Surface> void AddIfConicSection(Int2Type<true>);

    template<typename Surface> void AddIfRing(Int2Type<false>) {}

    template<typename Surface> void AddIfRing(Int2Type<true>);

    template<typename Surface> void AddIfElectrostaticBasis(Int2Type<false>) {}

    template<typename Surface> void AddIfElectrostaticBasis(Int2Type<true>);

    template<typename Surface> void AddIfDirichletBoundary(Int2Type<false>) {}

    template<typename Surface> void AddIfDirichletBoundary(Int2Type<true>);

    template<typename Surface> void AddIfNeumannBoundary(Int2Type<false>) {}

    template<typename Surface> void AddIfNeumannBoundary(Int2Type<true>);


    double TriangleAspectRatio(KFieldVector P0, KFieldVector P1, KFieldVector P2);
    double RectangleAspectRatio(const KFieldVector& P0, const KFieldVector& P1, const KFieldVector& P2,
                                const KFieldVector& P3);

    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkCellArray> fCells;
    vtkSmartPointer<vtkDoubleArray> fArea;
    vtkSmartPointer<vtkDoubleArray> fLogArea;
    vtkSmartPointer<vtkDoubleArray> fAspectRatio;
    vtkSmartPointer<vtkShortArray> fModulo;
    vtkSmartPointer<vtkDoubleArray> fChargeDensity;
    vtkSmartPointer<vtkDoubleArray> fLogChargeDensity;
    vtkSmartPointer<vtkDoubleArray> fPotential;
    vtkSmartPointer<vtkDoubleArray> fPermittivity;

    vtkSmartPointer<vtkTriangle> fTriangle;
    vtkSmartPointer<vtkQuad> fQuad;

    unsigned int fPointCounter;

    double fLineSegmentRadiusMin;

    // polygon cross-section for line segments (otherwise, they have no surface)
    unsigned int fLineSegmentPolyApprox;
    unsigned int fArcPolyApprox;

    KSurfacePrimitive* fSurfacePrimitive;

    mutable unsigned int fNElementsInRepresentation;
};

template<typename Surface> void KEMVTKViewer::PerformAction(Type2Type<Surface>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    AddIfTriangleSymmetryGroup<Surface>(Int2Type<IsDerivedFrom<Surface, KSymmetryGroup<KTriangle>>::Is>());
    AddIfRectangleSymmetryGroup<Surface>(Int2Type<IsDerivedFrom<Surface, KSymmetryGroup<KRectangle>>::Is>());
    AddIfLineSegmentSymmetryGroup<Surface>(Int2Type<IsDerivedFrom<Surface, KSymmetryGroup<KLineSegment>>::Is>());
    AddIfConicSectionSymmetryGroup<Surface>(Int2Type<IsDerivedFrom<Surface, KSymmetryGroup<KConicSection>>::Is>());
    AddIfRingSymmetryGroup<Surface>(Int2Type<IsDerivedFrom<Surface, KSymmetryGroup<KRing>>::Is>());
    AddIfTriangle<Surface>(Int2Type<IsDerivedFrom<Surface, KTriangle>::Is>());
    AddIfRectangle<Surface>(Int2Type<IsDerivedFrom<Surface, KRectangle>::Is>());
    AddIfLineSegment<Surface>(Int2Type<IsDerivedFrom<Surface, KLineSegment>::Is>());
    AddIfConicSection<Surface>(Int2Type<IsDerivedFrom<Surface, KConicSection>::Is>());
    AddIfRing<Surface>(Int2Type<IsDerivedFrom<Surface, KRing>::Is>());

    for (unsigned int i = 0; i < fNElementsInRepresentation; i++) {
        double area = s->Area();
        fArea->InsertNextValue(area);
        fLogArea->InsertNextValue(log10(area));
        static int modulo = 0;
        fModulo->InsertNextValue((modulo++) % 13);

        AddIfElectrostaticBasis<Surface>(Int2Type<IsDerivedFrom<Surface, KElectrostaticBasis>::Is>());
    }
}

template<typename Surface> void KEMVTKViewer::AddIfTriangleSymmetryGroup(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);
    fNElementsInRepresentation = s->size();

    for (unsigned int i = 0; i < s->size(); i++) {
        fPoints->InsertNextPoint(s->at(i)->GetP0()[0], s->at(i)->GetP0()[1], s->at(i)->GetP0()[2]);
        fPoints->InsertNextPoint(s->at(i)->GetP1()[0], s->at(i)->GetP1()[1], s->at(i)->GetP1()[2]);
        fPoints->InsertNextPoint(s->at(i)->GetP2()[0], s->at(i)->GetP2()[1], s->at(i)->GetP2()[2]);

        fTriangle->GetPointIds()->SetId(0, fPointCounter++);
        fTriangle->GetPointIds()->SetId(1, fPointCounter++);
        fTriangle->GetPointIds()->SetId(2, fPointCounter++);

        fCells->InsertNextCell(fTriangle);

        double aspect_ratio = TriangleAspectRatio(s->at(i)->GetP0(), s->at(i)->GetP1(), s->at(i)->GetP2());
        fAspectRatio->InsertNextValue(aspect_ratio);
    }
}

template<typename Surface> void KEMVTKViewer::AddIfRectangleSymmetryGroup(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);
    fNElementsInRepresentation = s->size();

    for (unsigned int i = 0; i < s->size(); i++) {
        fPoints->InsertNextPoint(s->at(i)->GetP0()[0], s->at(i)->GetP0()[1], s->at(i)->GetP0()[2]);
        fPoints->InsertNextPoint(s->at(i)->GetP1()[0], s->at(i)->GetP1()[1], s->at(i)->GetP1()[2]);
        fPoints->InsertNextPoint(s->at(i)->GetP2()[0], s->at(i)->GetP2()[1], s->at(i)->GetP2()[2]);
        fPoints->InsertNextPoint(s->at(i)->GetP3()[0], s->at(i)->GetP3()[1], s->at(i)->GetP3()[2]);

        fQuad->GetPointIds()->SetId(0, fPointCounter++);
        fQuad->GetPointIds()->SetId(1, fPointCounter++);
        fQuad->GetPointIds()->SetId(2, fPointCounter++);
        fQuad->GetPointIds()->SetId(3, fPointCounter++);

        fCells->InsertNextCell(fQuad);

        double aspect_ratio =
            RectangleAspectRatio(s->at(i)->GetP0(), s->at(i)->GetP1(), s->at(i)->GetP2(), s->at(i)->GetP3());
        fAspectRatio->InsertNextValue(aspect_ratio);
    }
}

template<typename Surface> void KEMVTKViewer::AddIfLineSegmentSymmetryGroup(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);
    fNElementsInRepresentation = fLineSegmentPolyApprox * s->size();

    for (unsigned int i = 0; i < s->size(); i++) {
        double radius = s->at(i)->GetDiameter() / 2.;
        if (radius < fLineSegmentRadiusMin)
            radius = fLineSegmentRadiusMin;

        double length = (s->at(i)->GetP0() - s->at(i)->GetP1()).Magnitude();

        KFieldVector tmp1, tmp2;
        for (unsigned int m = 0; m < 3; m++) {
            tmp1[m] = (s->at(i)->GetP1()[m] - s->at(i)->GetP0()[m]) / length;
            tmp2[m] = (s->at(i)->GetP1()[(m + 1) % 3] - s->at(i)->GetP0()[(m + 1) % 3]) / length;
        }

        KFieldVector norm1, norm2;
        for (unsigned int m = 0; m < 3; m++)
            norm1[m] = tmp1[(m + 1) % 3] * tmp2[(m + 2) % 3] - tmp1[(m + 2) % 3] * tmp2[(m + 1) % 3];
        for (unsigned int m = 0; m < 3; m++)
            norm2[m] = (tmp1[(m + 1) % 3] * norm1[(m + 2) % 3] - tmp1[(m + 2) % 3] * norm1[(m + 1) % 3]);

        KFieldVector p0, p1, p2, p3;

        for (unsigned int k = 0; k < fLineSegmentPolyApprox; k++) {
            double theta1 = 2. * M_PI * ((double) k) / fLineSegmentPolyApprox;
            double theta2 = 2. * M_PI * ((double) ((k + 1) % fLineSegmentPolyApprox)) / fLineSegmentPolyApprox;

            for (unsigned int m = 0; m < 3; m++) {
                p0[m] = (s->at(i)->GetP0()[m] + radius * norm1[m] * cos(theta1) + radius * norm2[m] * sin(theta1));
                p1[m] = (s->at(i)->GetP0()[m] + radius * norm1[m] * cos(theta2) + radius * norm2[m] * sin(theta2));
                p2[m] = (s->at(i)->GetP1()[m] + radius * norm1[m] * cos(theta2) + radius * norm2[m] * sin(theta2));
                p3[m] = (s->at(i)->GetP1()[m] + radius * norm1[m] * cos(theta1) + radius * norm2[m] * sin(theta1));
            }

            fPoints->InsertNextPoint(p0[0], p0[1], p0[2]);
            fPoints->InsertNextPoint(p1[0], p1[1], p1[2]);
            fPoints->InsertNextPoint(p2[0], p2[1], p2[2]);
            fPoints->InsertNextPoint(p3[0], p3[1], p3[2]);

            fQuad->GetPointIds()->SetId(0, fPointCounter++);
            fQuad->GetPointIds()->SetId(1, fPointCounter++);
            fQuad->GetPointIds()->SetId(2, fPointCounter++);
            fQuad->GetPointIds()->SetId(3, fPointCounter++);

            fCells->InsertNextCell(fQuad);

            double aspect_ratio = length / (2 * radius);
            fAspectRatio->InsertNextValue(aspect_ratio);
        }
    }
}

template<typename Surface> void KEMVTKViewer::AddIfConicSectionSymmetryGroup(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);
    fNElementsInRepresentation = fArcPolyApprox * s->size();

    for (unsigned int i = 0; i < s->size(); i++) {
        double z[2];
        double r[2];
        double theta[2];

        z[0] = s->at(i)->GetP0()[2];
        r[0] = sqrt(s->at(i)->GetP0()[0] * s->at(i)->GetP0()[0] + s->at(i)->GetP0()[1] * s->at(i)->GetP0()[1]);

        z[1] = s->at(i)->GetP1()[2];
        r[1] = sqrt(s->at(i)->GetP1()[0] * s->at(i)->GetP1()[0] + s->at(i)->GetP1()[1] * s->at(i)->GetP1()[1]);

        KFieldVector p[2][2];

        for (unsigned int k = 0; k < fArcPolyApprox; k++) {
            theta[0] = 2. * M_PI * ((double) (k) / (double) (fArcPolyApprox));
            theta[1] = 2. * M_PI * ((double) (k + 1) / (double) (fArcPolyApprox));

            // m index is axial
            for (unsigned int m = 0; m < 2; m++) {
                // n index is angular
                for (unsigned int n = 0; n < 2; n++) {
                    p[m][n][0] = r[m] * cos(theta[n]);
                    p[m][n][1] = r[m] * sin(theta[n]);
                    p[m][n][2] = z[m];
                }
            }

            fPoints->InsertNextPoint(p[0][0][0], p[0][0][1], p[0][0][2]);
            fPoints->InsertNextPoint(p[1][0][0], p[1][0][1], p[1][0][2]);
            fPoints->InsertNextPoint(p[1][1][0], p[1][1][1], p[1][1][2]);
            fPoints->InsertNextPoint(p[0][1][0], p[0][1][1], p[0][1][2]);

            fQuad->GetPointIds()->SetId(0, fPointCounter++);
            fQuad->GetPointIds()->SetId(1, fPointCounter++);
            fQuad->GetPointIds()->SetId(2, fPointCounter++);
            fQuad->GetPointIds()->SetId(3, fPointCounter++);

            fCells->InsertNextCell(fQuad);
            fAspectRatio->InsertNextValue(1.0);
        }
    }
    return;
}

template<typename Surface> void KEMVTKViewer::AddIfRingSymmetryGroup(Int2Type<true>)
{
    fNElementsInRepresentation = 0;
    return;
}

template<typename Surface> void KEMVTKViewer::AddIfTriangle(Int2Type<true>)
{
    fNElementsInRepresentation = 1;

    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    fPoints->InsertNextPoint(s->GetP0()[0], s->GetP0()[1], s->GetP0()[2]);
    fPoints->InsertNextPoint(s->GetP1()[0], s->GetP1()[1], s->GetP1()[2]);
    fPoints->InsertNextPoint(s->GetP2()[0], s->GetP2()[1], s->GetP2()[2]);

    fTriangle->GetPointIds()->SetId(0, fPointCounter++);
    fTriangle->GetPointIds()->SetId(1, fPointCounter++);
    fTriangle->GetPointIds()->SetId(2, fPointCounter++);

    fCells->InsertNextCell(fTriangle);

    double aspect_ratio = TriangleAspectRatio(s->GetP0(), s->GetP1(), s->GetP2());
    fAspectRatio->InsertNextValue(aspect_ratio);
}

template<typename Surface> void KEMVTKViewer::AddIfRectangle(Int2Type<true>)
{
    fNElementsInRepresentation = 1;

    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    fPoints->InsertNextPoint(s->GetP0()[0], s->GetP0()[1], s->GetP0()[2]);
    fPoints->InsertNextPoint(s->GetP1()[0], s->GetP1()[1], s->GetP1()[2]);
    fPoints->InsertNextPoint(s->GetP2()[0], s->GetP2()[1], s->GetP2()[2]);
    fPoints->InsertNextPoint(s->GetP3()[0], s->GetP3()[1], s->GetP3()[2]);

    fQuad->GetPointIds()->SetId(0, fPointCounter++);
    fQuad->GetPointIds()->SetId(1, fPointCounter++);
    fQuad->GetPointIds()->SetId(2, fPointCounter++);
    fQuad->GetPointIds()->SetId(3, fPointCounter++);

    fCells->InsertNextCell(fQuad);

    double aspect_ratio = RectangleAspectRatio(s->GetP0(), s->GetP1(), s->GetP2(), s->GetP3());
    fAspectRatio->InsertNextValue(aspect_ratio);
}

template<typename Surface> void KEMVTKViewer::AddIfLineSegment(Int2Type<true>)
{
    fNElementsInRepresentation = fLineSegmentPolyApprox;

    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    double radius = s->GetDiameter() / 2.;
    if (radius < fLineSegmentRadiusMin)
        radius = fLineSegmentRadiusMin;

    double length = (s->GetP0() - s->GetP1()).Magnitude();

    KFieldVector tmp1, tmp2;
    for (unsigned int m = 0; m < 3; m++) {
        tmp1[m] = (s->GetP1()[m] - s->GetP0()[m]) / length;
        tmp2[m] = (s->GetP1()[(m + 1) % 3] - s->GetP0()[(m + 1) % 3]) / length;
    }

    KFieldVector norm1, norm2;
    for (unsigned int m = 0; m < 3; m++)
        norm1[m] = tmp1[(m + 1) % 3] * tmp2[(m + 2) % 3] - tmp1[(m + 2) % 3] * tmp2[(m + 1) % 3];
    for (unsigned int m = 0; m < 3; m++)
        norm2[m] = (tmp1[(m + 1) % 3] * norm1[(m + 2) % 3] - tmp1[(m + 2) % 3] * norm1[(m + 1) % 3]);

    KFieldVector p0, p1, p2, p3;

    for (unsigned int k = 0; k < fLineSegmentPolyApprox; k++) {
        double theta1 = 2. * M_PI * ((double) k) / fLineSegmentPolyApprox;
        double theta2 = 2. * M_PI * ((double) ((k + 1) % fLineSegmentPolyApprox)) / fLineSegmentPolyApprox;

        for (unsigned int m = 0; m < 3; m++) {
            p0[m] = (s->GetP0()[m] + radius * norm1[m] * cos(theta1) + radius * norm2[m] * sin(theta1));
            p1[m] = (s->GetP0()[m] + radius * norm1[m] * cos(theta2) + radius * norm2[m] * sin(theta2));
            p2[m] = (s->GetP1()[m] + radius * norm1[m] * cos(theta2) + radius * norm2[m] * sin(theta2));
            p3[m] = (s->GetP1()[m] + radius * norm1[m] * cos(theta1) + radius * norm2[m] * sin(theta1));
        }

        fPoints->InsertNextPoint(p0[0], p0[1], p0[2]);
        fPoints->InsertNextPoint(p1[0], p1[1], p1[2]);
        fPoints->InsertNextPoint(p2[0], p2[1], p2[2]);
        fPoints->InsertNextPoint(p3[0], p3[1], p3[2]);

        fQuad->GetPointIds()->SetId(0, fPointCounter++);
        fQuad->GetPointIds()->SetId(1, fPointCounter++);
        fQuad->GetPointIds()->SetId(2, fPointCounter++);
        fQuad->GetPointIds()->SetId(3, fPointCounter++);

        fCells->InsertNextCell(fQuad);

        double aspect_ratio = length / (2 * radius);
        fAspectRatio->InsertNextValue(aspect_ratio);
    }
}

template<typename Surface> void KEMVTKViewer::AddIfConicSection(Int2Type<true>)
{
    fNElementsInRepresentation = fArcPolyApprox;

    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    double z[2];
    double r[2];
    double theta[2];

    z[0] = s->GetP0()[2];
    r[0] = sqrt(s->GetP0()[0] * s->GetP0()[0] + s->GetP0()[1] * s->GetP0()[1]);

    z[1] = s->GetP1()[2];
    r[1] = sqrt(s->GetP1()[0] * s->GetP1()[0] + s->GetP1()[1] * s->GetP1()[1]);

    KFieldVector p[2][2];

    for (unsigned int k = 0; k < fArcPolyApprox; k++) {
        theta[0] = 2. * M_PI * ((double) (k) / (double) (fArcPolyApprox));
        theta[1] = 2. * M_PI * ((double) (k + 1) / (double) (fArcPolyApprox));

        // m index is axial
        for (unsigned int m = 0; m < 2; m++) {
            // n index is angular
            for (unsigned int n = 0; n < 2; n++) {
                p[m][n][0] = r[m] * cos(theta[n]);
                p[m][n][1] = r[m] * sin(theta[n]);
                p[m][n][2] = z[m];
            }
        }

        fPoints->InsertNextPoint(p[0][0][0], p[0][0][1], p[0][0][2]);
        fPoints->InsertNextPoint(p[1][0][0], p[1][0][1], p[1][0][2]);
        fPoints->InsertNextPoint(p[1][1][0], p[1][1][1], p[1][1][2]);
        fPoints->InsertNextPoint(p[0][1][0], p[0][1][1], p[0][1][2]);

        fQuad->GetPointIds()->SetId(0, fPointCounter++);
        fQuad->GetPointIds()->SetId(1, fPointCounter++);
        fQuad->GetPointIds()->SetId(2, fPointCounter++);
        fQuad->GetPointIds()->SetId(3, fPointCounter++);

        fCells->InsertNextCell(fQuad);
        fAspectRatio->InsertNextValue(1.0);
    }

    return;
}

template<typename Surface> void KEMVTKViewer::AddIfRing(Int2Type<true>)
{
    fNElementsInRepresentation = 0;
    return;
}

template<typename Surface> void KEMVTKViewer::AddIfElectrostaticBasis(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    double cd = s->GetSolution();
    fChargeDensity->InsertNextValue(cd);
    if (cd > 0.)
        fLogChargeDensity->InsertNextValue(log(1. + 1.e16 * cd));
    else
        fLogChargeDensity->InsertNextValue(-log(1. + 1.e16 * fabs(cd)));

    AddIfDirichletBoundary<Surface>(Int2Type<IsDerivedFrom<Surface, KDirichletBoundary>::Is>());
    AddIfNeumannBoundary<Surface>(Int2Type<IsDerivedFrom<Surface, KNeumannBoundary>::Is>());
}

template<typename Surface> void KEMVTKViewer::AddIfDirichletBoundary(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    fPotential->InsertNextValue(s->GetBoundaryValue());
    fPermittivity->InsertNextValue(std::numeric_limits<double>::quiet_NaN());
}

template<typename Surface> void KEMVTKViewer::AddIfNeumannBoundary(Int2Type<true>)
{
    auto* s = static_cast<Surface*>(fSurfacePrimitive);

    fPotential->InsertNextValue(std::numeric_limits<double>::quiet_NaN());
    fPermittivity->InsertNextValue(s->GetNormalBoundaryFlux());
}


inline double KEMVTKViewer::TriangleAspectRatio(KFieldVector P0, KFieldVector P1, KFieldVector P2)
{
    double a, b, c, max;
    double delx, dely, delz;

    delx = P1[0] - P0[0];
    dely = P1[1] - P0[1];
    delz = P1[2] - P0[2];

    a = std::sqrt(delx * delx + dely * dely + delz * delz);

    delx = P2[0] - P0[0];
    dely = P2[1] - P0[1];
    delz = P2[2] - P0[2];

    b = std::sqrt(delx * delx + dely * dely + delz * delz);

    delx = P1[0] - P2[0];
    dely = P1[1] - P2[1];
    delz = P1[2] - P2[2];

    c = std::sqrt(delx * delx + dely * dely + delz * delz);

    KFieldVector PA;
    KFieldVector PB;
    KFieldVector PC;
    KFieldVector V;
    KFieldVector X;
    KFieldVector Y;
    KFieldVector Q;
    KFieldVector SUB;

    //find the longest side:
    if (a > b) {
        max = a;
        PA = KFieldVector(P2[0], P2[1], P2[2]);
        PB = KFieldVector(P0[0], P0[1], P0[2]);
        PC = KFieldVector(P1[0], P1[1], P1[2]);
    }
    else {
        max = b;
        PA = KFieldVector(P1[0], P1[1], P1[2]);
        PB = KFieldVector(P2[0], P2[1], P2[2]);
        PC = KFieldVector(P0[0], P0[1], P0[2]);
    }

    if (c > max) {
        max = c;
        PA = KFieldVector(P0[0], P0[1], P0[2]);
        PB = KFieldVector(P1[0], P1[1], P1[2]);
        PC = KFieldVector(P2[0], P2[1], P2[2]);
    }

    //the line pointing along v is the y-axis
    V = PC - PB;
    Y = V.Unit();

    //q is closest point to fP[0] on line connecting fP[1] to fP[2]
    double t = (PA.Dot(V) - PB.Dot(V)) / (V.Dot(V));
    Q = PB + t * V;

    //the line going from fP[0] to fQ is the x-axis
    X = Q - PA;
    //gram-schmidt out any y-axis component in the x-axis
    double proj = X.Dot(Y);
    SUB = proj * Y;
    X = X - SUB;
    double H = X.Magnitude();  //compute triangle height along x

    //compute the triangles aspect ratio
    double ratio = max / H;

    return ratio;
}


inline double KEMVTKViewer::RectangleAspectRatio(const KFieldVector& P0, const KFieldVector& P1, const KFieldVector& P2,
                                                 const KFieldVector& P3)
{
    //figure out which vertices make the sides
    KFieldVector p[4];
    p[0] = P0;
    p[1] = P1;
    p[2] = P2;
    p[3] = P3;

    double d01 = (p[0] - p[1]).Magnitude();
    double d02 = (p[0] - p[2]).Magnitude();
    double d03 = (p[0] - p[3]).Magnitude();

    int a_mid, b_mid;
    double max_dist = d01;
    a_mid = 2;
    b_mid = 3;
    if (d02 > max_dist) {
        max_dist = d02;
        a_mid = 3;
        b_mid = 1;
    }
    if (d03 > max_dist) {
        max_dist = d03;
        a_mid = 1;
        b_mid = 2;
    }

    double a = (p[a_mid] - p[0]).Magnitude();
    double b = (p[b_mid] - p[0]).Magnitude();

    double val = a / b;
    if (val < 1.0) {
        return 1.0 / val;
    };
    return val;
}


}  // namespace KEMField

#endif /* KEMVTKVIEWER_DEF */
