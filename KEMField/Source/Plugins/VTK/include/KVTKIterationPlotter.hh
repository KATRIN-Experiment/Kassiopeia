#ifndef KVTKITERATIONPLOTTER_DEF
#define KVTKITERATIONPLOTTER_DEF

#include "KIterativeSolver.hh"

#include <vtkAxis.h>
#include <vtkChartXY.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkPen.h>
#include <vtkPlot.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkVersion.h>

namespace KEMField
{
template<typename ValueType> class KVTKIterationPlotter : public KIterativeSolver<ValueType>::Visitor
{
  public:
    KVTKIterationPlotter(unsigned int i = 1)
    {
        KIterativeSolver<ValueType>::Visitor::Interval(i);
    }
    virtual ~KVTKIterationPlotter() = default;

    void InitializePlot();
    void CreatePlot();

    void AddPoint(float x, float y);

    void Initialize(KIterativeSolver<ValueType>&);
    void Visit(KIterativeSolver<ValueType>&);
    void Finalize(KIterativeSolver<ValueType>&);

  private:
    vtkSmartPointer<vtkFloatArray> fArrayX;
    vtkSmartPointer<vtkFloatArray> fArrayY;
    vtkSmartPointer<vtkContextView> fView;
    vtkSmartPointer<vtkChartXY> fChart;
};

template<typename ValueType> void KVTKIterationPlotter<ValueType>::InitializePlot()
{
    fArrayX = vtkSmartPointer<vtkFloatArray>::New();
    fArrayX->SetName("Iteration");

    fArrayY = vtkSmartPointer<vtkFloatArray>::New();
    fArrayY->SetName("|Residual|");

    // Set up the view
    fView = vtkSmartPointer<vtkContextView>::New();
    fView->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    fView->GetRenderWindow()->SetSize(500, 300);
    fView->GetRenderWindow()->SetMultiSamples(0);

    // Add line plot, setting the colors etc
    fChart = vtkSmartPointer<vtkChartXY>::New();
    fChart->GetAxis(vtkAxis::LEFT)->SetTitle("Log(|Residual|)");
    fChart->GetAxis(vtkAxis::LEFT)->SetLogScale(true);
    fChart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Iteration");

    // fView->GetRenderWindow()->SetPosition(2810,2000);
    fView->GetScene()->AddItem(fChart);
    fView->GetInteractor()->Initialize();
}

template<typename ValueType> void KVTKIterationPlotter<ValueType>::CreatePlot()
{
    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    table->AddColumn(fArrayX);
    table->AddColumn(fArrayY);

    vtkPlot* dots = fChart->AddPlot(vtkChart::POINTS);
#if VTK_MAJOR_VERSION <= 5
    dots->SetInput(table, 0, 1);
#else
    dots->SetInputData(table, 0, 1);
#endif
    dots->SetColor(0, 0, 0, 255);

    if (fArrayX->GetSize() >= 2) {
        vtkPlot* line = fChart->AddPlot(vtkChart::LINE);
#if VTK_MAJOR_VERSION <= 5
        line->SetInput(table, 0, 1);
#else
        line->SetInputData(table, 0, 1);
#endif
        line->SetColor(0, 0, 255, 255);
        line->SetWidth(1.0);
    }

    fChart->Modified();
    fView->GetRenderWindow()->Render();
}

template<typename ValueType> void KVTKIterationPlotter<ValueType>::AddPoint(float x, float y)
{
    std::cout << "adding points: " << x << ", " << y << std::endl;
    fArrayX->InsertNextValue(x);
    fArrayY->InsertNextValue(y);
    CreatePlot();
}

template<typename ValueType> void KVTKIterationPlotter<ValueType>::Initialize(KIterativeSolver<ValueType>&)
{
    InitializePlot();
}

template<typename ValueType> void KVTKIterationPlotter<ValueType>::Visit(KIterativeSolver<ValueType>& solver)
{
    AddPoint(solver.Iteration(), solver.ResidualNorm());
}

template<typename ValueType> void KVTKIterationPlotter<ValueType>::Finalize(KIterativeSolver<ValueType>&)
{
    kem_cout() << "KVTKIterationPlotter finished; waiting for key press ..." << eom;
    fView->GetInteractor()->Start();
    fView->GetRenderWindow()->Finalize();
}
}  // namespace KEMField

#endif /* KVTKITERATIONPLOTTER_DEF */
