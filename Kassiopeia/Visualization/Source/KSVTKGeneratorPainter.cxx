#include "KSVTKGeneratorPainter.h"

#include "KFile.h"
#include "KSComponentTemplate.h"
#include "KSDictionary.h"
#include "KSGenerator.h"
#include "KSParticleFactory.h"
#include "KSRootElectricField.h"
#include "KSRootMagneticField.h"
#include "KSVisualizationMessage.h"
#include "KToolbox.h"
#include "vtkArrowSource.h"
#include "vtkCellData.h"
#include "vtkGlyph3D.h"
#include "vtkPointData.h"
#include "vtkPolyLine.h"
#include "vtkProperty.h"

#include <algorithm>
#include <limits>

using namespace KGeoBag;
using namespace katrin;
using namespace std;

namespace Kassiopeia
{

KSVTKGeneratorPainter::KSVTKGeneratorPainter() :
    fNumSamples(100),
    fScaleFactor(1.),
    fColorVariable(""),
    fElectricField(new KSRootElectricField()),
    fMagneticField(new KSRootMagneticField()),
    fGlyph(vtkSmartPointer<vtkGlyph3D>::New()),
    fPoints(vtkSmartPointer<vtkPoints>::New()),
    fColors(vtkSmartPointer<vtkUnsignedCharArray>::New()),
    fScalars(vtkSmartPointer<vtkDoubleArray>::New()),
    fVectors(vtkSmartPointer<vtkDoubleArray>::New()),
    fData(vtkSmartPointer<vtkPolyData>::New()),
    fMapper(vtkSmartPointer<vtkPolyDataMapper>::New()),
    fActor(vtkSmartPointer<vtkActor>::New()),
    fColorTable(vtkSmartPointer<vtkLookupTable>::New()),
    fNamedColors(vtkSmartPointer<vtkNamedColors>::New())
{
    vtkSmartPointer<vtkArrowSource> vArrowSource = vtkSmartPointer<vtkArrowSource>::New();
    //vArrowSource->SetShaftRadius( 0.03 );
    //vArrowSource->SetTipRadius( 0.1 );
    //vArrowSource->SetTipLength( 0.35 );
    //vArrowSource->SetShaftResolution( 6 );
    //vArrowSource->SetTipResolution( 6 );

    fGlyph->SetSourceConnection(vArrowSource->GetOutputPort());
    fGlyph->SetInputData(fData);
    fGlyph->SetVectorModeToUseVector();
    fGlyph->SetScaleModeToScaleByVector();
    fGlyph->SetColorModeToColorByScalar();

    fColorTable->SetNumberOfTableValues(256);
    fColorTable->SetHueRange(0.000, 0.667);
    fColorTable->SetRampToLinear();
    fColorTable->SetVectorModeToMagnitude();
    fColorTable->Build();

    fColors->SetNumberOfComponents(3);
    fScalars->SetNumberOfComponents(1);
    fVectors->SetNumberOfComponents(3);
    fData->SetPoints(fPoints);
    // point data is set in Update() since it depends on coloring scheme
#ifdef VTK6
    fMapper->SetInputConnection(fGlyph->GetOutputPort());
#else
    fMapper->SetInput(fGlyph->GetOutput());
#endif
    fMapper->SetScalarModeToUsePointData();
    fMapper->SetLookupTable(fColorTable);
    //fMapper->SetColorModeToMapScalars();
    fActor->SetMapper(fMapper);
}
KSVTKGeneratorPainter::~KSVTKGeneratorPainter() = default;

void KSVTKGeneratorPainter::AddElectricField(KSElectricField* aField)
{
    fElectricField->AddElectricField(aField);
}

void KSVTKGeneratorPainter::AddMagneticField(KSMagneticField* aField)
{
    fMagneticField->AddMagneticField(aField);
}

void KSVTKGeneratorPainter::Render()
{
    fElectricField->TryInitialize();
    fMagneticField->TryInitialize();

    KSParticleFactory::GetInstance().SetElectricField(fElectricField);
    KSParticleFactory::GetInstance().SetMagneticField(fMagneticField);

    double tNorm = 0.;
    for (auto tGeneratorName : fGenerators) {
        auto tGenerator = KToolbox::GetInstance().Get<KSGenerator>(tGeneratorName);
        tGenerator->TryInitialize();

        vismsg(eInfo) << "generator " << tGeneratorName << " creating " << fNumSamples << " particles for visualization"
                      << eom;

        for (auto tCount = 0; tCount < fNumSamples; tCount++) {
            KSParticleQueue tParticleQueue;
            tGenerator->ExecuteGeneration(tParticleQueue);

            for (auto* tParticle : tParticleQueue) {
                //tParticle->Print();

                KThreeVector tPosition = tParticle->GetPosition();
                KThreeVector tDirection = tParticle->GetMomentum().Unit();
                double tEnergy = tParticle->GetKineticEnergy_eV();

                // point and scale by direction and energy
                KThreeVector tVector = tDirection * tEnergy;
                tNorm = max(tNorm, tVector.Magnitude());

                fPoints->InsertNextPoint(tPosition.X(), tPosition.Y(), tPosition.Z());
                fVectors->InsertNextTuple3(tVector.X(), tVector.Y(), tVector.Z());

                if (fColorVariable.empty()) {
                    unsigned char red, green, blue, alpha;
                    fNamedColors->GetColor(tGeneratorName,
                                           red,
                                           green,
                                           blue,
                                           alpha);  // returns black if color is not found
                    fColors->InsertNextTuple3(red, green, blue);
                }
                else {
                    double color = GetScalarValue(*tParticle, fColorVariable);
                    fScalars->InsertNextValue(color);
                }
            }
        }
    }

    // rescale vectors so the arrows have a reasonable length
    vismsg(eDebug) << "vtk generator painter <" << GetName() << "> applying vector normalization of <" << 1. / tNorm
                   << ">" << eom;
    for (auto tId = 0; tId < fVectors->GetNumberOfTuples(); tId++) {
        double* tValue = fVectors->GetPointer(tId);
        *tValue /= tNorm;
    }

    fData->GetPointData()->SetVectors(fVectors);
    if (fColorVariable.empty()) {
        // color by set of fixed colors (user-defined)
        fData->GetPointData()->SetScalars(fColors);
        fGlyph->SetScaleModeToDataScalingOff();
    }
    else {
        // color by scalar variable (particle data)
        fData->GetPointData()->SetScalars(fScalars);
        fGlyph->SetScaleModeToScaleByVector();
        fMapper->SetColorModeToMapScalars();
        fMapper->SetScalarRange(fScalars->GetRange());
    }

    fGlyph->SetScaleFactor(fScaleFactor / 50);  // default arrows are a little too large
    fGlyph->Update();

    fMapper->Update();

    return;
}

void KSVTKGeneratorPainter::Display()
{
    if (fDisplayEnabled == true) {
        vtkSmartPointer<vtkRenderer> vRenderer = fWindow->GetRenderer();
        vRenderer->AddActor(fActor);
    }

    return;
}

void KSVTKGeneratorPainter::Write()
{
    if (fWriteEnabled == true) {
        string tFile;

        if (fFile.length() > 0) {
            if (!fPath.empty()) {
                tFile = string(fPath) + string("/") + fFile;
            }
            else {
                tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + fFile;
            }
        }
        else {
            if (!fPath.empty()) {
                tFile = string(fPath) + string("/") + GetName() + string(".vtp");
            }
            else {
                tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");
            }
        }

        vismsg(eNormal) << "vtk generator painter <" << GetName() << "> is writing <" << fData->GetNumberOfPoints()
                        << "> points to file <" << tFile << ">" << eom;

        vtkSmartPointer<vtkXMLPolyDataWriter> vWriter = fWindow->GetWriter();
        vWriter->SetFileName(tFile.c_str());
        vWriter->SetDataModeToBinary();
#ifdef VTK6
        vWriter->SetInputData(fData);
#else
        vWriter->SetInput(fData);
#endif
        vWriter->Write();
    }
    return;
}

double KSVTKGeneratorPainter::GetScalarValue(KSParticle& aParticle, std::string aName)
{
    /// NOTE: this repeats mappings from KSParticle - would be nice to use the KDictionary/KSComponent interface instead
    if (aName == "index_number")
        return aParticle.GetIndexNumber();
    else if (aName == "pid")
        return aParticle.GetPID();
    else if (aName == "mass")
        return aParticle.GetMass();
    else if (aName == "charge")
        return aParticle.GetCharge();
    else if (aName == "total_spin")
        return aParticle.GetSpinMagnitude();
    else if (aName == "gyromagnetic_ratio")
        return aParticle.GetGyromagneticRatio();
    else if (aName == "n")
        return aParticle.GetMainQuantumNumber();
    else if (aName == "l")
        return aParticle.GetSecondQuantumNumber();
    else if (aName == "time")
        return aParticle.GetTime();
    else if (aName == "clock_time")
        return aParticle.GetClockTime();
    else if (aName == "length")
        return aParticle.GetLength();
    else if (aName.substr(0, 8) == "position")
        return GetScalarValue(aParticle.GetPosition(), aName);
    else if (aName.substr(0, 8) == "momentum")
        return GetScalarValue(aParticle.GetMomentum(), aName);
    else if (aName.substr(0, 8) == "velocity")
        return GetScalarValue(aParticle.GetVelocity(), aName);
    else if (aName == "spin0")
        return aParticle.GetSpin0();
    else if (aName.substr(0, 4) == "spin")
        return GetScalarValue(aParticle.GetSpin(), aName);
    else if (aName == "aligned_spin")
        return aParticle.GetAlignedSpin();
    else if (aName == "spin_angle")
        return aParticle.GetSpinAngle();
    else if (aName == "speed")
        return aParticle.GetSpeed();
    else if (aName == "lorentz_factor")
        return aParticle.GetLorentzFactor();
    else if (aName == "kinetic_energy")
        return aParticle.GetKineticEnergy();
    else if (aName == "kinetic_energy_ev")
        return aParticle.GetKineticEnergy_eV();
    else if (aName == "polar_angle_to_z")
        return aParticle.GetPolarAngleToZ();
    else if (aName == "azimuthal_angle_to_x")
        return aParticle.GetAzimuthalAngleToX();
    else if (aName.substr(0, 14) == "magnetic_field")
        return GetScalarValue(aParticle.GetMagneticField(), aName);
    else if (aName.substr(0, 14) == "electric_field")
        return GetScalarValue(aParticle.GetElectricField(), aName);
    else if (aName.substr(0, 17) == "magnetic_gradient")
        return GetScalarValue(aParticle.GetMagneticGradient(), aName);
    else if (aName == "electric_potential")
        return aParticle.GetElectricPotential();
    else if (aName == "long_momentum")
        return aParticle.GetLongMomentum();
    else if (aName == "trans_momentum")
        return aParticle.GetTransMomentum();
    else if (aName == "long_velocity")
        return aParticle.GetLongVelocity();
    else if (aName == "trans_velocity")
        return aParticle.GetTransVelocity();
    else if (aName == "polar_angle_to_b")
        return aParticle.GetPolarAngleToB();
    else if (aName == "cyclotron_frequency")
        return aParticle.GetCyclotronFrequency();
    else if (aName == "orbital_magnetic_moment")
        return aParticle.GetOrbitalMagneticMoment();
    else if (aName.substr() == "guiding_center_position")
        return GetScalarValue(aParticle.GetGuidingCenterPosition(), aName);

    vismsg(eError) << "could not get scalar value from field <" << aName << ">" << eom;
    return std::numeric_limits<double>::quiet_NaN();
}

double KSVTKGeneratorPainter::GetScalarValue(const KThreeVector& aVector, std::string aName)
{
    size_t pos = aName.rfind("_");
    if (pos != string::npos)
        aName = aName.substr(pos + 1);

    if (aName.empty() || aName == "magnitude")
        return aVector.Magnitude();
    else if (aName == "x")
        return aVector.X();
    else if (aName == "y")
        return aVector.Y();
    else if (aName == "z")
        return aVector.Z();
    else if (aName == "r" || aName == "perp")
        return aVector.Perp();
    else if (aName == "p" || aName == "polar_angle")
        return aVector.PolarAngle();
    else if (aName == "a" || aName == "azimuthal_angle")
        return aVector.AzimuthalAngle();

    vismsg(eError) << "could not get scalar value <" << aName << "> from vector " << aVector << eom;
    return std::numeric_limits<double>::quiet_NaN();
}

double KSVTKGeneratorPainter::GetScalarValue(const KThreeMatrix& aTensor, std::string aName)
{
    size_t pos = aName.rfind("_");
    if (pos != string::npos)
        aName = aName.substr(pos + 1);

    if (aName.empty() || aName == "trace")
        return aTensor.Trace();
    else if (aName == "xx")
        return aTensor(0, 0);
    else if (aName == "xy")
        return aTensor(0, 1);
    else if (aName == "xz")
        return aTensor(0, 2);
    else if (aName == "yx")
        return aTensor(1, 0);
    else if (aName == "yy")
        return aTensor(1, 1);
    else if (aName == "yz")
        return aTensor(1, 2);
    else if (aName == "zx")
        return aTensor(2, 0);
    else if (aName == "zy")
        return aTensor(2, 1);
    else if (aName == "zz")
        return aTensor(2, 2);

    vismsg(eError) << "could not get scalar value <" << aName << "> from tensor " << aTensor << eom;
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace Kassiopeia
