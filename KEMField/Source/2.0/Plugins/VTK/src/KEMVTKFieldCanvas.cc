#include <cmath>

#include "KEMVTKFieldCanvas.hh"

namespace KEMField
{
  KEMVTKFieldCanvas::KEMVTKFieldCanvas(double x_1,
				       double x_2,
				       double y_1,
				       double y_2,
				       double zmir,
				       bool   isfull)
    : KEMFieldCanvas(x_1,
		     x_2,
		     y_1,
		     y_2,
		     zmir,
		     isfull)
  {
    InitializeCanvas();
  }

  void KEMVTKFieldCanvas::InitializeCanvas()
  {
    int SCALE1 = 800;
    int SCALE2;
    if (full)
      SCALE2 = 800;
    else
      SCALE2 = 400;

    view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderWindow()->SetSize(SCALE1,SCALE2);
  }

  //______________________________________________________________________________

  KEMVTKFieldCanvas::~KEMVTKFieldCanvas()
  {
  }

  //______________________________________________________________________________

  void KEMVTKFieldCanvas::DrawFieldMap(std::vector<double> x,
				       std::vector<double> y,
				       std::vector<double> V,
				       bool xy,
				       double z)
  {
    (void)xy;
    (void)z;

    chart = vtkSmartPointer<vtkChartHistogram2D>::New();

    view->GetScene()->AddItem(chart);

    data = vtkSmartPointer<vtkImageData>::New();
    data->SetExtent(0, x.size()-1, 0, y.size()-1, 0, 0);
#ifdef VTK6
    data->AllocateScalars(VTK_DOUBLE,1);
#else
    data->SetScalarTypeToDouble();
    data->SetNumberOfScalarComponents(1);
    data->AllocateScalars();
#endif

    data->SetOrigin(x.at(0), y.at(0), 0.0);
    double dx = (x.at(x.size()-1)-x.at(0))/x.size();
    double dy = (y.at(y.size()-1)-y.at(0))/y.size();
    data->SetSpacing(dx, dy, 1.0);

    double min = 1.e10;
    double max = -1.e10;

    double *dPtr = static_cast<double *>(data->GetScalarPointer(0, 0, 0));
    int V_val = 0;
    for (int i = 0; i < (int)x.size(); ++i)
    {
      for (int j = 0; j < (int)y.size(); ++j)
      {
	dPtr[i + x.size()*j] = V[V_val];
	if (V[V_val]<min) min = V[V_val];
	if (V[V_val]>max) max = V[V_val];
	V_val++;
      }
    }
#ifdef VTK6
    chart->SetInputData(data);
#else
    chart->SetInput(data);
#endif

    double range = max-min;

    transferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
    transferFunction->AddRGBSegment(min + 0.0*range, 0.5, 0.0, 1.0,
				    min + 0.2*range, 0.0, 0.0, 1.0);
    transferFunction->AddRGBSegment(min + 0.2*range, 0.0, 0.0, 1.0,
				    min + 0.4*range, 0.0, 1.0, 1.0);
    transferFunction->AddRGBSegment(min + 0.4*range, 0.0, 1.0, 1.0,
				    min + 0.6*range, 0.0, 1.0, 0.0 );
    transferFunction->AddRGBSegment(min + 0.6*range, 0.0, 1.0, 0.0,
				    min + 0.8*range, 1.0, 1.0, 0.0 );
    transferFunction->AddRGBSegment(min + 0.8*range, 1.0, 1.0, 0.0,
				    min + 1.0*range, 1.0, 0.0, 0.0 );
    transferFunction->Build();
    chart->SetTransferFunction(transferFunction);
  }

  //______________________________________________________________________________

  void KEMVTKFieldCanvas::DrawComparisonMap(int nPoints,
					    std::vector<double> x,
					    std::vector<double> y,
					    std::vector<double> V1,
					    std::vector<double> V2)
  {
    (void)nPoints;
    chart = vtkSmartPointer<vtkChartHistogram2D>::New();

    view->GetScene()->AddItem(chart);

    data = vtkSmartPointer<vtkImageData>::New();
    data->SetExtent(0, x.size()-1, 0, y.size()-1, 0, 0);
#ifdef VTK6
    data->AllocateScalars(VTK_DOUBLE,1);
#else
    data->SetScalarTypeToDouble();
    data->SetNumberOfScalarComponents(1);
    data->AllocateScalars();
#endif

    data->SetOrigin(x.at(0), y.at(0), 0.0);
    double dx = (x.at(x.size()-1)-x.at(0))/x.size();
    double dy = (y.at(y.size()-1)-y.at(0))/y.size();
    data->SetSpacing(dx, dy, 1.0);

    double min = 1.e10;
    double max = -1.e10;

    double *dPtr = static_cast<double *>(data->GetScalarPointer(0, 0, 0));
    int V_val = 0;
    for (int i = 0; i < (int)x.size(); ++i)
    {
      for (int j = 0; j < (int)y.size(); ++j)
      {
	double V = V2[V_val] - V1[V_val];
	dPtr[i + x.size()*j] = V;
	if (V<min) min = V;
	if (V>max) max = V;
	V_val++;
      }
    }
#ifdef VTK6
    chart->SetInputData(data);
#else
    chart->SetInput(data);
#endif

    double range = max-min;

    transferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
    transferFunction->AddRGBSegment(min + 0.0*range, 0.5, 0.0, 1.0,
				    min + 0.2*range, 0.0, 0.0, 1.0);
    transferFunction->AddRGBSegment(min + 0.2*range, 0.0, 0.0, 1.0,
				    min + 0.4*range, 0.0, 1.0, 1.0);
    transferFunction->AddRGBSegment(min + 0.4*range, 0.0, 1.0, 1.0,
				    min + 0.6*range, 0.0, 1.0, 0.0 );
    transferFunction->AddRGBSegment(min + 0.6*range, 0.0, 1.0, 0.0,
				    min + 0.8*range, 1.0, 1.0, 0.0 );
    transferFunction->AddRGBSegment(min + 0.8*range, 1.0, 1.0, 0.0,
				    min + 1.0*range, 1.0, 0.0, 0.0 );
    transferFunction->Build();
    chart->SetTransferFunction(transferFunction);
  }

  //______________________________________________________________________________

  void KEMVTKFieldCanvas::DrawFieldLines(std::vector<double> x,
					 std::vector<double> y)
  {
    (void)x;
    (void)y;
  }

  //______________________________________________________________________________

  void KEMVTKFieldCanvas::LabelAxes(std::string xname,
				    std::string yname,
				    std::string zname)
  {
    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle(xname);
    chart->GetAxis(vtkAxis::LEFT)->SetTitle(yname);
    chart->GetPlot(0)->SetLabel(zname);
  }

  //______________________________________________________________________________

  void KEMVTKFieldCanvas::LabelCanvas(std::string name)
  {
    chart->SetTitle(name);
  }

  //______________________________________________________________________________

  void KEMVTKFieldCanvas::SaveAs(std::string savename)
  {
    view->GetRenderWindow()->SetMultiSamples(0);
    view->GetInteractor()->Initialize();

    // Screenshot  
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = 
      vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(view->GetRenderWindow());
    windowToImageFilter->Update();
 
    vtkSmartPointer<vtkPNGWriter> writer = 
      vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(savename.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();
  }
}
