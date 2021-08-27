#include "KSWriteVTK.h"

#include "KFile.h"
#include "KSComponentGroup.h"
#include "KSWritersMessage.h"

#ifdef Kassiopeia_USE_BOOST
#include "KPathUtils.h"
using katrin::KPathUtils;
#endif

using namespace std;
using KGeoBag::KTwoVector;
using KGeoBag::KThreeVector;
using KGeoBag::KTwoMatrix;
using KGeoBag::KThreeMatrix;

namespace Kassiopeia
{

KSWriteVTK::KSWriteVTK() :
    fBase(""),
    fPath(""),
    fTrackPointFlag(false),
    fTrackPointComponent(nullptr),
    fTrackPointAction(nullptr, nullptr),
    fTrackDataFlag(false),
    fTrackDataComponent(nullptr),
    fTrackDataActions(),
    fStepPointFlag(false),
    fStepPointComponent(nullptr),
    fStepPointAction(nullptr, nullptr),
    fStepDataFlag(false),
    fStepDataComponent(nullptr),
    fStepDataActions()
{}
KSWriteVTK::KSWriteVTK(const KSWriteVTK& aCopy) :
    KSComponent(aCopy),
    fBase(aCopy.fBase),
    fPath(aCopy.fPath),
    fTrackPointFlag(false),
    fTrackPointComponent(nullptr),
    fTrackPointAction(nullptr, nullptr),
    fTrackDataFlag(false),
    fTrackDataComponent(nullptr),
    fTrackDataActions(),
    fStepPointFlag(false),
    fStepPointComponent(nullptr),
    fStepPointAction(nullptr, nullptr),
    fStepDataFlag(false),
    fStepDataComponent(nullptr),
    fStepDataActions()
{}
KSWriteVTK* KSWriteVTK::Clone() const
{
    return new KSWriteVTK(*this);
}
KSWriteVTK::~KSWriteVTK() = default;

void KSWriteVTK::SetBase(const string& aBase)
{
    fBase = aBase;
    return;
}
void KSWriteVTK::SetPath(const string& aPath)
{
    fPath = aPath;
    return;
}

void KSWriteVTK::ExecuteRun()
{
    wtrmsg_debug("VTK writer <" << GetName() << "> is filling a run" << eom);
    return;
}
void KSWriteVTK::ExecuteEvent()
{
    wtrmsg_debug("VTK writer <" << GetName() << "> is filling an event" << eom);
    BreakTrack();
    return;
}
void KSWriteVTK::ExecuteTrack()
{
    wtrmsg_debug("VTK writer <" << GetName() << "> is filling a track" << eom);
    BreakStep();
    FillTrack();
    return;
}
void KSWriteVTK::ExecuteStep()
{
    wtrmsg_debug("VTK writer <" << GetName() << "> is filling a step" << eom);
    FillStep();
    return;
}

void KSWriteVTK::SetTrackPoint(KSComponent* anComponent)
{
    if (fTrackPointFlag == false) {
        if (fTrackPointComponent == nullptr) {
            wtrmsg_debug("VTK writer <" << GetName() << "> is adding a track point object" << eom);
            fTrackPointFlag = true;
            fTrackPointComponent = anComponent;
            AddTrackPoint(anComponent);
            return;
        }
        else {
            if (fTrackPointComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is enabling a track point object" << eom);
                fTrackPointFlag = true;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a different track point object" << eom;
                return;
            }
        }
    }
    else {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a second track point object" << eom;
        return;
    }
}
void KSWriteVTK::ClearTrackPoint(KSComponent* anComponent)
{
    if (fTrackPointFlag == false) {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a cleared track point object" << eom;
        return;
    }
    else {
        if (fTrackPointComponent == nullptr) {
            wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a null track point object" << eom;
            return;
        }
        else {
            if (fTrackPointComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is clearing a track point object" << eom);
                BreakTrack();
                fTrackPointFlag = false;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a different track point object"
                               << eom;
                return;
            }
        }
    }
}

void KSWriteVTK::SetTrackData(KSComponent* anComponent)
{
    if (fTrackDataFlag == false) {
        if (fTrackDataComponent == nullptr) {
            wtrmsg_debug("VTK writer <" << GetName() << "> is adding a track data object" << eom);
            fTrackDataFlag = true;
            fTrackDataComponent = anComponent;
            AddTrackData(anComponent);
            return;
        }
        else {
            if (fTrackDataComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is enabling a track data object" << eom);
                fTrackDataFlag = true;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a different track data object" << eom;
                return;
            }
        }
    }
    else {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a second track data object" << eom;
        return;
    }
}
void KSWriteVTK::ClearTrackData(KSComponent* anComponent)
{
    if (fTrackDataFlag == false) {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a cleared track data object" << eom;
        return;
    }
    else {
        if (fTrackDataComponent == nullptr) {
            wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a null track data object" << eom;
            return;
        }
        else {
            if (fTrackDataComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is clearing a track data object" << eom);
                BreakTrack();
                fTrackDataFlag = false;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a different track data object"
                               << eom;
                return;
            }
        }
    }
}

void KSWriteVTK::SetStepPoint(KSComponent* anComponent)
{
    if (fStepPointFlag == false) {
        if (fStepPointComponent == nullptr) {
            wtrmsg_debug("VTK writer <" << GetName() << "> is adding a step point object" << eom);
            fStepPointFlag = true;
            fStepPointComponent = anComponent;
            AddStepPoint(anComponent);
            return;
        }
        else {
            if (fStepPointComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is enabling a step point object" << eom);
                fStepPointFlag = true;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a different step point object" << eom;
                return;
            }
        }
    }
    else {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a second step point object" << eom;
        return;
    }
}
void KSWriteVTK::ClearStepPoint(KSComponent* anComponent)
{
    if (fStepPointFlag == false) {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a cleared step point object" << eom;
        return;
    }
    else {
        if (fStepPointComponent == nullptr) {
            wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a null step point object" << eom;
            return;
        }
        else {
            if (fStepPointComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is clearing a step point object" << eom);
                BreakStep();
                fStepPointFlag = false;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a different step point object"
                               << eom;
                return;
            }
        }
    }
}

void KSWriteVTK::SetStepData(KSComponent* anComponent)
{
    if (fStepDataFlag == false) {
        if (fStepDataComponent == nullptr) {
            wtrmsg_debug("VTK writer <" << GetName() << "> is adding a step data object" << eom);
            fStepDataFlag = true;
            fStepDataComponent = anComponent;
            AddStepData(anComponent);
            return;
        }
        else {
            if (fStepDataComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is enabling a step data object" << eom);
                fStepDataFlag = true;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a different step data object" << eom;
                return;
            }
        }
    }
    else {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to set a second step data object" << eom;
        return;
    }
}
void KSWriteVTK::ClearStepData(KSComponent* anComponent)
{
    if (fStepDataFlag == false) {
        wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a cleared step data object" << eom;
        return;
    }
    else {
        if (fStepDataComponent == nullptr) {
            wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a null step data object" << eom;
            return;
        }
        else {
            if (fStepDataComponent == anComponent) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is clearing a step data object" << eom);
                BreakStep();
                fStepDataFlag = false;
                return;
            }
            else {
                wtrmsg(eError) << "VTK writer <" << GetName() << "> tried to clear a different step data object" << eom;
                return;
            }
        }
    }
}

void KSWriteVTK::InitializeComponent()
{
    wtrmsg_debug("starting VTK writer" << eom);

    fStepPoints = vtkSmartPointer<vtkPoints>::New();
    fStepLines = vtkSmartPointer<vtkCellArray>::New();
    fStepData = vtkSmartPointer<vtkPolyData>::New();

    fStepData->SetPoints(fStepPoints);
    fStepData->SetLines(fStepLines);

    fTrackPoints = vtkSmartPointer<vtkPoints>::New();
    fTrackVertices = vtkSmartPointer<vtkCellArray>::New();
    fTrackData = vtkSmartPointer<vtkPolyData>::New();

    fTrackData->SetPoints(fTrackPoints);
    fTrackData->SetVerts(fTrackVertices);

    return;
}
void KSWriteVTK::DeinitializeComponent()
{
    wtrmsg_debug("stopping VTK writer" << eom);

    string tBase = fBase.empty() ? GetName() : fBase;
    string tPath = fPath.empty() ? OUTPUT_DEFAULT_DIR : fPath;

#ifdef Kassiopeia_USE_BOOST
    KPathUtils::MakeDirectory(tPath);
#endif

    vtkSmartPointer<vtkXMLPolyDataWriter> tStepWriter = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    tStepWriter->SetFileName((tPath + string("/") + tBase + string("Step.vtp")).c_str());
    tStepWriter->SetDataModeToBinary();
#ifdef VTK6
    tStepWriter->SetInputData(fStepData);
#else
    tStepWriter->SetInput(fStepData);
#endif

    if (tStepWriter->Write() == 1) {
        wtrmsg(eNormal) << "VTK step output was written to file <" << tStepWriter->GetFileName() << ">" << eom;
    }
    else {
        wtrmsg(eWarning) << "could not write VTK step output to file <" << tStepWriter->GetFileName() << ">" << eom;
    }

    vtkSmartPointer<vtkXMLPolyDataWriter> tTrackWriter = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    tTrackWriter->SetFileName((tPath + string("/") + tBase + string("Track.vtp")).c_str());
    tTrackWriter->SetDataModeToBinary();
#ifdef VTK6
    tTrackWriter->SetInputData(fTrackData);
#else
    tTrackWriter->SetInput(fTrackData);
#endif
    tTrackWriter->Write();

    if (tTrackWriter->Write() == 1) {
        wtrmsg(eNormal) << "VTK track output was written to file <" << tTrackWriter->GetFileName() << ">" << eom;
    }
    else {
        wtrmsg(eWarning) << "could not write VTK track output to file <" << tTrackWriter->GetFileName() << ">" << eom;
    }

    return;
}

void KSWriteVTK::AddTrackPoint(KSComponent* anComponent)
{
    wtrmsg_debug("VTK writer <" << GetName() << "> making track point action for object <" << anComponent->GetName()
                                << ">" << eom);

        auto* tThreeVector = anComponent->As<KThreeVector>();
    if (tThreeVector != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a three_vector" << eom);
        fTrackPointAction.first = anComponent;
        fTrackPointAction.second = new PointAction(tThreeVector, fTrackIds, fTrackPoints);
        return;
    }

    wtrmsg(eError) << "VTK writer <" << GetName() << "> cannot make point action for object <" << anComponent->GetName()
                   << ">" << eom;

    return;
}

void KSWriteVTK::AddTrackData(KSComponent* anComponent)
{
    wtrmsg_debug("VTK writer <" << GetName() << "> making track data action for object <" << anComponent->GetName()
                                << ">" << eom);

        auto* tComponentGroup = anComponent->As<KSComponentGroup>();
    if (tComponentGroup != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a group" << eom);
        for (unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++) {
            AddTrackData(tComponentGroup->ComponentAt(tIndex));
        }
        return;
    }

    auto* tString = anComponent->As<string>();
    if (tString != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a string" << eom);
        vtkSmartPointer<vtkStringArray> tArray = vtkSmartPointer<vtkStringArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new StringAction(tString, tArray)));
        return;
    }

    auto* tTwoVector = anComponent->As<KTwoVector>();
    if (tTwoVector != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a two_vector" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(2);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new TwoVectorAction(tTwoVector, tArray)));
        return;
    }
    auto* tThreeVector = anComponent->As<KThreeVector>();
    if (tThreeVector != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a three_vector" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(3);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new ThreeVectorAction(tThreeVector, tArray)));
        return;
    }

    auto* tTwoMatrix = anComponent->As<KTwoMatrix>();
    if (tTwoMatrix != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a two_matrix" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(4);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new TwoMatrixAction(tTwoMatrix, tArray)));
        return;
    }
    auto* tThreeMatrix = anComponent->As<KThreeMatrix>();
    if (tThreeMatrix != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a three_matrix" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(9);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new ThreeMatrixAction(tThreeMatrix, tArray)));
        return;
    }

    bool* tBool = anComponent->As<bool>();
    if (tBool != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a bool" << eom);
        vtkSmartPointer<vtkUnsignedCharArray> tArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new BoolAction(tBool, tArray)));
        return;
    }

    auto* tUChar = anComponent->As<unsigned char>();
    if (tUChar != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an unsigned_short" << eom);
        vtkSmartPointer<vtkUnsignedCharArray> tArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new UCharAction(tUChar, tArray)));
        return;
    }
    auto* tChar = anComponent->As<char>();
    if (tChar != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a short" << eom);
        vtkSmartPointer<vtkCharArray> tArray = vtkSmartPointer<vtkCharArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new CharAction(tChar, tArray)));
        return;
    }

    auto* tUShort = anComponent->As<unsigned short>();
    if (tUShort != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an unsigned_short" << eom);
        vtkSmartPointer<vtkUnsignedShortArray> tArray = vtkSmartPointer<vtkUnsignedShortArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new UShortAction(tUShort, tArray)));
        return;
    }
    auto* tShort = anComponent->As<short>();
    if (tShort != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a short" << eom);
        vtkSmartPointer<vtkShortArray> tArray = vtkSmartPointer<vtkShortArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new ShortAction(tShort, tArray)));
        return;
    }

    auto* tUInt = anComponent->As<unsigned int>();
    if (tUInt != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a unsigned_int" << eom);
            vtkSmartPointer<vtkUnsignedIntArray>
                tArray = vtkSmartPointer<vtkUnsignedIntArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new UIntAction(tUInt, tArray)));
        return;
    }
    int* tInt = anComponent->As<int>();
    if (tInt != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an int" << eom); vtkSmartPointer<vtkIntArray>
            tArray = vtkSmartPointer<vtkIntArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new IntAction(tInt, tArray)));
        return;
    }

    auto* tULong = anComponent->As<unsigned long>();
    if (tULong != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an unsigned_long" << eom);
        vtkSmartPointer<vtkUnsignedLongArray> tArray = vtkSmartPointer<vtkUnsignedLongArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new ULongAction(tULong, tArray)));
        return;
    }
    long* tLong = anComponent->As<long>();
    if (tLong != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a long" << eom);
        vtkSmartPointer<vtkLongArray> tArray = vtkSmartPointer<vtkLongArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new LongAction(tLong, tArray)));
        return;
    }
    long long* tLongLong = anComponent->As<long long>();
    if (tLongLong != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a long_long" << eom);
        vtkSmartPointer<vtkLongLongArray> tArray = vtkSmartPointer<vtkLongLongArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new LongLongAction(tLongLong, tArray)));
        return;
    }

    auto* tFloat = anComponent->As<float>();
    if (tFloat != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a float" << eom);
        vtkSmartPointer<vtkFloatArray> tArray = vtkSmartPointer<vtkFloatArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new FloatAction(tFloat, tArray)));
        return;
    }
    auto* tDouble = anComponent->As<double>();
    if (tDouble != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a double" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fTrackData->GetPointData()->AddArray(tArray);
        fTrackDataActions.insert(ActionEntry(anComponent, new DoubleAction(tDouble, tArray)));
        return;
    }

    wtrmsg(eError) << "VTK writer cannot make data action for object <" << anComponent->GetName() << ">" << eom;

    return;
}

void KSWriteVTK::FillTrack()
{
    if (fTrackPointFlag == true) {
        if (fTrackDataFlag == true) {
            wtrmsg_debug("VTK writer <" << GetName() << "> is filling a track" << eom);

            fTrackPointAction.first->PullUpdate();
            fTrackPointAction.second->Execute();
            fTrackPointAction.first->PullDeupdate();

            for (auto& trackDataAction : fTrackDataActions) {
                trackDataAction.first->PullUpdate();
                trackDataAction.second->Execute();
                trackDataAction.first->PullDeupdate();
            }
        }
    }

    return;
}

void KSWriteVTK::BreakTrack()
{
    if (fTrackPointFlag == true) {
        if (fTrackDataFlag == true) {
            if (fTrackIds.empty() == false) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is breaking a track set with <" << fTrackIds.size()
                                            << "> elements" << eom);

                vtkSmartPointer<vtkVertex> tVertex = vtkSmartPointer<vtkVertex>::New();
                for (unsigned int tIndex = 0; tIndex < fTrackIds.size(); tIndex++) {
                    tVertex->GetPointIds()->SetId(tIndex, fTrackIds.at(tIndex));
                    fTrackVertices->InsertNextCell(tVertex);
                }
                fTrackIds.clear();
            }
        }
    }

    return;
}

void KSWriteVTK::AddStepPoint(KSComponent* anComponent)
{
    wtrmsg_debug("VTK writer <" << GetName() << "> making step point action for object <" << anComponent->GetName()
                                << ">" << eom);

        auto* tThreeVector = anComponent->As<KThreeVector>();
    if (tThreeVector != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a three_vector" << eom);
        fStepPointAction.first = anComponent;
        fStepPointAction.second = new PointAction(tThreeVector, fStepIds, fStepPoints);
        return;
    }

    wtrmsg(eError) << "VTK writer <" << GetName() << "> cannot make point action for object <" << anComponent->GetName()
                   << ">" << eom;

    return;
}

void KSWriteVTK::AddStepData(KSComponent* anComponent)
{
    wtrmsg_debug("VTK writer <" << GetName() << "> making step data action for object <" << anComponent->GetName()
                                << ">" << eom);

        auto* tComponentGroup = anComponent->As<KSComponentGroup>();
    if (tComponentGroup != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a group" << eom);
        for (unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++) {
            AddStepData(tComponentGroup->ComponentAt(tIndex));
        }
        return;
    }

    auto* tString = anComponent->As<string>();
    if (tString != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a string" << eom);
        vtkSmartPointer<vtkStringArray> tArray = vtkSmartPointer<vtkStringArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new StringAction(tString, tArray)));
        return;
    }

    auto* tTwoVector = anComponent->As<KTwoVector>();
    if (tTwoVector != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a two_vector" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(2);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new TwoVectorAction(tTwoVector, tArray)));
        return;
    }
    auto* tThreeVector = anComponent->As<KThreeVector>();
    if (tThreeVector != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a three_vector" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(3);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new ThreeVectorAction(tThreeVector, tArray)));
        return;
    }

    auto* tTwoMatrix = anComponent->As<KTwoMatrix>();
    if (tTwoMatrix != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a two_matrix" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(4);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new TwoMatrixAction(tTwoMatrix, tArray)));
        return;
    }
    auto* tThreeMatrix = anComponent->As<KThreeMatrix>();
    if (tThreeMatrix != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a three_matrix" << eom);
            vtkSmartPointer<vtkDoubleArray>
                tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(9);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new ThreeMatrixAction(tThreeMatrix, tArray)));
        return;
    }

    bool* tBool = anComponent->As<bool>();
    if (tBool != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a bool" << eom);
        vtkSmartPointer<vtkUnsignedCharArray> tArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new BoolAction(tBool, tArray)));
        return;
    }

    auto* tUChar = anComponent->As<unsigned char>();
    if (tUChar != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an unsigned_short" << eom);
        vtkSmartPointer<vtkUnsignedCharArray> tArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new UCharAction(tUChar, tArray)));
        return;
    }
    auto* tChar = anComponent->As<char>();
    if (tChar != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a short" << eom);
        vtkSmartPointer<vtkCharArray> tArray = vtkSmartPointer<vtkCharArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new CharAction(tChar, tArray)));
        return;
    }

    auto* tUShort = anComponent->As<unsigned short>();
    if (tUShort != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an unsigned_short" << eom);
        vtkSmartPointer<vtkUnsignedShortArray> tArray = vtkSmartPointer<vtkUnsignedShortArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new UShortAction(tUShort, tArray)));
        return;
    }
    auto* tShort = anComponent->As<short>();
    if (tShort != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a short" << eom);
        vtkSmartPointer<vtkShortArray> tArray = vtkSmartPointer<vtkShortArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new ShortAction(tShort, tArray)));
        return;
    }

    auto* tUInt = anComponent->As<unsigned int>();
    if (tUInt != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a unsigned_int" << eom);
        vtkSmartPointer<vtkUnsignedIntArray> tArray = vtkSmartPointer<vtkUnsignedIntArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new UIntAction(tUInt, tArray)));
        return;
    }
    int* tInt = anComponent->As<int>();
    if (tInt != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an int" << eom);
        vtkSmartPointer<vtkIntArray> tArray = vtkSmartPointer<vtkIntArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new IntAction(tInt, tArray)));
        return;
    }

    auto* tULong = anComponent->As<unsigned long>();
    if (tULong != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is an unsigned_long" << eom);
        vtkSmartPointer<vtkUnsignedLongArray> tArray = vtkSmartPointer<vtkUnsignedLongArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new ULongAction(tULong, tArray)));
        return;
    }
    long* tLong = anComponent->As<long>();
    if (tLong != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a long" << eom);
        vtkSmartPointer<vtkLongArray> tArray = vtkSmartPointer<vtkLongArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new LongAction(tLong, tArray)));
        return;
    }
    long long* tLongLong = anComponent->As<long long>();
    if (tLongLong != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a long_long" << eom);
        vtkSmartPointer<vtkLongLongArray> tArray = vtkSmartPointer<vtkLongLongArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new LongLongAction(tLongLong, tArray)));
        return;
    }

    auto* tFloat = anComponent->As<float>();
    if (tFloat != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a float" << eom);
        vtkSmartPointer<vtkFloatArray> tArray = vtkSmartPointer<vtkFloatArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new FloatAction(tFloat, tArray)));
        return;
    }
    auto* tDouble = anComponent->As<double>();
    if (tDouble != nullptr) {
        wtrmsg_debug("  object <" << anComponent->GetName() << "> is a double" << eom);
        vtkSmartPointer<vtkDoubleArray> tArray = vtkSmartPointer<vtkDoubleArray>::New();
        tArray->SetName(anComponent->GetName().c_str());
        tArray->SetNumberOfComponents(1);
        fStepData->GetPointData()->AddArray(tArray);
        fStepDataActions.insert(ActionEntry(anComponent, new DoubleAction(tDouble, tArray)));
        return;
    }

    wtrmsg(eError) << "VTK writer cannot make data action for object <" << anComponent->GetName() << ">" << eom;

    return;
}

void KSWriteVTK::FillStep()
{
    if (fStepPointFlag == true) {
        if (fStepDataFlag == true) {
            wtrmsg_debug("VTK writer <" << GetName() << "> is filling a step" << eom);

            fStepPointAction.first->PullUpdate();
            fStepPointAction.second->Execute();
            fStepPointAction.first->PullDeupdate();

            for (auto& stepDataAction : fStepDataActions) {
                stepDataAction.first->PullUpdate();
                stepDataAction.second->Execute();
                stepDataAction.first->PullDeupdate();
            }
        }
    }

    return;
}

void KSWriteVTK::BreakStep()
{
    if (fStepPointFlag == true) {
        if (fStepDataFlag == true) {
            if (fStepIds.empty() == false) {
                wtrmsg_debug("VTK writer <" << GetName() << "> is breaking a step set with <" << fStepIds.size()
                                            << "> elements" << eom);

                vtkSmartPointer<vtkPolyLine> tPolyLine = vtkSmartPointer<vtkPolyLine>::New();
                tPolyLine->GetPointIds()->SetNumberOfIds(fStepIds.size());
                for (unsigned int tIndex = 0; tIndex < fStepIds.size(); tIndex++) {
                    tPolyLine->GetPointIds()->SetId(tIndex, fStepIds.at(tIndex));
                }
                fStepIds.clear();
                fStepLines->InsertNextCell(tPolyLine);
            }
        }
    }

    return;
}

STATICINT sKSWriteVTKDict =
    KSDictionary<KSWriteVTK>::AddCommand(&KSWriteVTK::SetStepPoint, &KSWriteVTK::ClearStepPoint, "set_step_point",
                                         "clear_step_point") +
    KSDictionary<KSWriteVTK>::AddCommand(&KSWriteVTK::SetStepData, &KSWriteVTK::ClearStepData, "set_step_data",
                                         "clear_step_data") +
    KSDictionary<KSWriteVTK>::AddCommand(&KSWriteVTK::SetTrackPoint, &KSWriteVTK::ClearTrackPoint, "set_track_point",
                                         "clear_track_point") +
    KSDictionary<KSWriteVTK>::AddCommand(&KSWriteVTK::SetTrackData, &KSWriteVTK::ClearTrackData, "set_track_data",
                                         "clear_track_data");

}  // namespace Kassiopeia
