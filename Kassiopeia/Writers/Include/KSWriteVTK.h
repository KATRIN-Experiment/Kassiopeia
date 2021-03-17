#ifndef _Kassiopeia_KSWriteVTK_h_
#define _Kassiopeia_KSWriteVTK_h_

#include "KSWriter.h"
#include "KThreeVector.hh"
#include "KTwoVector.hh"
#include "vtkCellArray.h"
#include "vtkCharArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkLongArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyLine.h"
#include "vtkShortArray.h"
#include "vtkSmartPointer.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkVertex.h"
#include "vtkXMLPolyDataWriter.h"

namespace Kassiopeia
{

class KSWriteVTK : public KSComponentTemplate<KSWriteVTK, KSWriter>
{
  private:
    class Action
    {
      public:
        Action() = default;
        virtual ~Action() = default;

        virtual void Execute() = 0;
    };

    class UIntAction : public Action
    {
      public:
        UIntAction(unsigned int* aData, vtkSmartPointer<vtkUnsignedIntArray> anArray) : fData(aData), fArray(anArray) {}
        ~UIntAction() override = default;

        void Execute() override
        {
            fArray->InsertNextValue(*fData);
            return;
        }

      private:
        unsigned int* fData;
        vtkSmartPointer<vtkUnsignedIntArray> fArray;
    };

    class IntAction : public Action
    {
      public:
        IntAction(int* aData, vtkSmartPointer<vtkIntArray> anArray) : fData(aData), fArray(anArray) {}
        ~IntAction() override = default;

        void Execute() override
        {
            fArray->InsertNextValue(*fData);
            return;
        }

      private:
        int* fData;
        vtkSmartPointer<vtkIntArray> fArray;
    };

    class FloatAction : public Action
    {
      public:
        FloatAction(float* aData, vtkSmartPointer<vtkFloatArray> anArray) : fData(aData), fArray(anArray) {}
        ~FloatAction() override = default;

        void Execute() override
        {
            fArray->InsertNextValue(*fData);
            return;
        }

      private:
        float* fData;
        vtkSmartPointer<vtkFloatArray> fArray;
    };

    class DoubleAction : public Action
    {
      public:
        DoubleAction(double* aData, vtkSmartPointer<vtkDoubleArray> anArray) : fData(aData), fArray(anArray) {}
        ~DoubleAction() override = default;

        void Execute() override
        {
            fArray->InsertNextValue(*fData);
            return;
        }

      private:
        double* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class TwoVectorAction : public Action
    {
      public:
        TwoVectorAction(KGeoBag::KTwoVector* aData, vtkSmartPointer<vtkDoubleArray> anArray) :
            fData(aData),
            fArray(anArray)
        {}
        ~TwoVectorAction() override = default;

        void Execute() override
        {
            fArray->InsertNextTuple2(fData->X(), fData->Y());
            return;
        }

      private:
        KGeoBag::KTwoVector* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class ThreeVectorAction : public Action
    {
      public:
        ThreeVectorAction(KGeoBag::KThreeVector* aData, vtkSmartPointer<vtkDoubleArray> anArray) :
            fData(aData),
            fArray(anArray)
        {}
        ~ThreeVectorAction() override = default;

        void Execute() override
        {
            fArray->InsertNextTuple3(fData->X(), fData->Y(), fData->Z());
            return;
        }

      private:
        KGeoBag::KThreeVector* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class PointAction : public Action
    {
      public:
        PointAction(KGeoBag::KThreeVector* aData, std::vector<vtkIdType>& anIds, vtkSmartPointer<vtkPoints> aPoints) :
            fData(aData),
            fIds(anIds),
            fPoints(aPoints)
        {}
        ~PointAction() override = default;

        void Execute() override
        {
            fIds.push_back(fPoints->InsertNextPoint(fData->X(), fData->Y(), fData->Z()));
            return;
        }

      private:
        KGeoBag::KThreeVector* fData;
        std::vector<vtkIdType>& fIds;
        vtkSmartPointer<vtkPoints> fPoints;
    };

    using ActionMap = std::map<KSComponent*, Action*>;
    using ActionEntry = std::pair<KSComponent*, Action*>;
    using ActionIt = ActionMap::iterator;
    using ActionCIt = ActionMap::const_iterator;

  public:
    KSWriteVTK();
    KSWriteVTK(const KSWriteVTK& aCopy);
    KSWriteVTK* Clone() const override;
    ~KSWriteVTK() override;

  public:
    void SetBase(const std::string& aBase);
    void SetPath(const std::string& aPath);

  private:
    std::string fBase;
    std::string fPath;

  public:
    void ExecuteRun() override;
    void ExecuteEvent() override;
    void ExecuteTrack() override;
    void ExecuteStep() override;

    void SetTrackPoint(KSComponent* aComponent);
    void ClearTrackPoint(KSComponent* aComponent);

    void SetTrackData(KSComponent* aComponent);
    void ClearTrackData(KSComponent* aComponent);

    void SetStepPoint(KSComponent* aComponent);
    void ClearStepPoint(KSComponent* aComponent);

    void SetStepData(KSComponent* aComponent);
    void ClearStepData(KSComponent* aComponent);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    void AddTrackPoint(KSComponent* aComponent);
    void AddTrackData(KSComponent* aComponent);
    void FillTrack();
    void BreakTrack();

    bool fTrackPointFlag;
    KSComponent* fTrackPointComponent;
    ActionEntry fTrackPointAction;

    bool fTrackDataFlag;
    KSComponent* fTrackDataComponent;
    ActionMap fTrackDataActions;

    std::vector<vtkIdType> fTrackIds;
    vtkSmartPointer<vtkPoints> fTrackPoints;
    vtkSmartPointer<vtkCellArray> fTrackVertices;
    vtkSmartPointer<vtkPolyData> fTrackData;

    void AddStepPoint(KSComponent* aComponent);
    void AddStepData(KSComponent* aComponent);
    void FillStep();
    void BreakStep();

    bool fStepPointFlag;
    KSComponent* fStepPointComponent;
    ActionEntry fStepPointAction;

    bool fStepDataFlag;
    KSComponent* fStepDataComponent;
    ActionMap fStepDataActions;

    std::vector<vtkIdType> fStepIds;
    vtkSmartPointer<vtkPoints> fStepPoints;
    vtkSmartPointer<vtkCellArray> fStepLines;
    vtkSmartPointer<vtkPolyData> fStepData;
};

}  // namespace Kassiopeia

#endif
