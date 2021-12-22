#ifndef _Kassiopeia_KSWriteVTK_h_
#define _Kassiopeia_KSWriteVTK_h_

#include "KSWriter.h"
#include "vtkCellArray.h"
#include "vtkCharArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkLongArray.h"
#include "vtkLongLongArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyLine.h"
#include "vtkShortArray.h"
#include "vtkSmartPointer.h"
#include "vtkStringArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkVertex.h"
#include "vtkXMLPolyDataWriter.h"

#include "KThreeMatrix.hh"
#include "KThreeVector.hh"
#include "KTwoMatrix.hh"
#include "KTwoVector.hh"

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

    template<typename T, typename V>
    class SimpleAction : public Action
    {
      public:
        SimpleAction(T* aData, vtkSmartPointer<V> anArray) : fData(aData), fArray(anArray) {}
        ~SimpleAction() override = default;

        void Execute() override
        {
            fArray->InsertNextValue(*fData);
            return;
        }

      private:
        T* fData;
        vtkSmartPointer<V> fArray;
    };

    using StringAction = SimpleAction<std::string, vtkStringArray>;
    using BoolAction = SimpleAction<bool, vtkUnsignedCharArray>;
    using UCharAction = SimpleAction<unsigned char, vtkUnsignedCharArray>;
    using CharAction = SimpleAction<char, vtkCharArray>;
    using UShortAction = SimpleAction<unsigned short, vtkUnsignedShortArray>;
    using ShortAction = SimpleAction<short, vtkShortArray>;
    using UIntAction = SimpleAction<unsigned int, vtkUnsignedIntArray>;
    using IntAction = SimpleAction<int, vtkIntArray>;
    using ULongAction = SimpleAction<unsigned long, vtkUnsignedLongArray>;
    using LongAction = SimpleAction<long, vtkLongArray>;
    using LongLongAction = SimpleAction<long long, vtkLongLongArray>;
    using FloatAction = SimpleAction<float, vtkFloatArray>;
    using DoubleAction = SimpleAction<double, vtkDoubleArray>;

    class TwoVectorAction : public Action
    {
      public:
        TwoVectorAction(katrin::KTwoVector* aData, vtkSmartPointer<vtkDoubleArray> anArray) :
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
        katrin::KTwoVector* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class ThreeVectorAction : public Action
    {
      public:
        ThreeVectorAction(katrin::KThreeVector* aData, vtkSmartPointer<vtkDoubleArray> anArray) :
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
        katrin::KThreeVector* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class TwoMatrixAction : public Action
    {
      public:
        TwoMatrixAction(katrin::KTwoMatrix* aData, vtkSmartPointer<vtkDoubleArray> anArray) :
            fData(aData),
            fArray(anArray)
        {}
        ~TwoMatrixAction() override = default;

        void Execute() override
        {
            fArray->InsertNextTuple4(fData->At(0), fData->At(1), fData->At(2), fData->At(3));
            return;
        }

      private:
        katrin::KTwoMatrix* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class ThreeMatrixAction : public Action
    {
      public:
        ThreeMatrixAction(katrin::KThreeMatrix* aData, vtkSmartPointer<vtkDoubleArray> anArray) :
            fData(aData),
            fArray(anArray)
        {}
        ~ThreeMatrixAction() override = default;

        void Execute() override
        {
            fArray->InsertNextTuple9(fData->At(0), fData->At(1), fData->At(2),
                                     fData->At(3), fData->At(4), fData->At(5),
                                     fData->At(6), fData->At(7), fData->At(8));
            return;
        }

      private:
        katrin::KThreeMatrix* fData;
        vtkSmartPointer<vtkDoubleArray> fArray;
    };

    class PointAction : public Action
    {
      public:
        PointAction(katrin::KThreeVector* aData, std::vector<vtkIdType>& anIds, vtkSmartPointer<vtkPoints> aPoints) :
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
        katrin::KThreeVector* fData;
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
