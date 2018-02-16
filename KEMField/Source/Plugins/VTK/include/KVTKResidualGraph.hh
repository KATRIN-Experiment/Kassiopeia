#ifndef KVTKRESIDUALGRAPH_DEF
#define KVTKRESIDUALGRAPH_DEF

#include <vtkSmartPointer.h>
#include <vtkChartXY.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkContextView.h>

#include "KIterativeSolver.hh"
#include "KSimpleVector.hh"

#include <vtkVersion.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkContextScene.h>
#include <vtkPlot.h>
#include <vtkAxis.h>
#include <vtkPen.h>

namespace KEMField
{
  template <typename ValueType>
  class KVTKResidualGraph : public KIterativeSolver<ValueType>::Visitor
  {
  public:
    KVTKResidualGraph(unsigned int i=1)
    { KIterativeSolver<ValueType>::Visitor::Interval(i); }
    virtual ~KVTKResidualGraph() {}

    void InitializeGraph(unsigned int);
    void CreateGraph();

    void UpdateGraph();

    void Initialize(KIterativeSolver<ValueType>&);
    void Visit(KIterativeSolver<ValueType>&);
    void Finalize(KIterativeSolver<ValueType>&);

  private:
    KSimpleVector<ValueType> fResidual;

    vtkSmartPointer<vtkFloatArray> fArrayX;
    vtkSmartPointer<vtkFloatArray> fArrayY;
    vtkSmartPointer<vtkContextView> fView;
    vtkSmartPointer<vtkChartXY> fChart;
  };

  template <typename ValueType>
  void KVTKResidualGraph<ValueType>::InitializeGraph(unsigned int dimension)
  {
    fResidual.resize(dimension,1.);

    fArrayX = vtkSmartPointer<vtkFloatArray>::New();
    fArrayX->SetName("Dimension");

    fArrayY = vtkSmartPointer<vtkFloatArray>::New();
    fArrayY->SetName("Residual_i");

    for (unsigned int i=0;i<dimension;i++)
    {
      fArrayX->InsertNextValue(i);
      fArrayY->InsertNextValue(fResidual(i));
    }

    // Set up the view
    fView = vtkSmartPointer<vtkContextView>::New();
    fView->GetRenderer()->SetBackground(1.0, 1.0, 1.0); 

    // Add line plot, setting the colors etc
    fChart = vtkSmartPointer<vtkChartXY>::New();
    fChart->GetAxis(vtkAxis::LEFT)->SetTitle("Log(Residual_i)");
    fChart->GetAxis(vtkAxis::LEFT)->SetLogScale(true);
    fChart->GetAxis(vtkAxis::LEFT)->SetRange(1.e-16,10.);
    fChart->GetAxis(vtkAxis::LEFT)->SetBehavior(vtkAxis::FIXED);
    fChart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Dimension");

    // fView->GetRenderWindow()->SetPosition(2810,2000);
    fView->GetRenderWindow()->SetSize(1000,256);
    fView->GetScene()->AddItem(fChart);
 }

  template <typename ValueType>
  void KVTKResidualGraph<ValueType>::CreateGraph()
  {
    fChart->ClearPlots();

    vtkSmartPointer<vtkTable> table =
      vtkSmartPointer<vtkTable>::New();
    table->AddColumn(fArrayX);
    table->AddColumn(fArrayY);

    vtkPlot *line = fChart->AddPlot(vtkChart::LINE);
#if VTK_MAJOR_VERSION <= 5
    line->SetInput(table, 0, 1);
#else
    line->SetInputData(table, 0, 1);
#endif
    line->SetColor(0, 255, 0, 255);
    line->SetWidth(1.0);
 
    fChart->Modified();
    fView->GetRenderWindow()->Render();
  }

  template <typename ValueType>
  void KVTKResidualGraph<ValueType>::UpdateGraph()
  {
    for (unsigned int i=0;i<fResidual.Dimension();i++)
      fArrayY->InsertValue(i,(fResidual(i) ? fabs(fResidual(i)) : 1.e-16));
    CreateGraph();
  }

  template <typename ValueType>
  void KVTKResidualGraph<ValueType>::Initialize(KIterativeSolver<ValueType>& solver)
  {
    InitializeGraph(solver.Dimension());
    UpdateGraph();
  }

  template <typename ValueType>
  void KVTKResidualGraph<ValueType>::Visit(KIterativeSolver<ValueType>& solver)
  {    
    solver.GetResidualVector(fResidual);
    UpdateGraph();
  }

  template <typename ValueType>
  void KVTKResidualGraph<ValueType>::Finalize(KIterativeSolver<ValueType>&)
  {
    fView->GetRenderWindow()->Finalize();
  }
}

#endif /* KVTKRESIDUALGRAPH_DEF */
