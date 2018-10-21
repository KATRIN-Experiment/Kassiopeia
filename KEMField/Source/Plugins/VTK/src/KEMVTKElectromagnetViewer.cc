#include "KEMVTKElectromagnetViewer.hh"

#include "KEMCout.hh"

namespace KEMField
{
  KEMVTKElectromagnetViewer::KEMVTKElectromagnetViewer(KElectromagnetContainer& anElectromagnetContainer)
  {
    for (unsigned int i=0;i<anElectromagnetContainer.size();i++)
    {
      anElectromagnetContainer.at(i)->Accept(*this);
    }
  }

  void KEMVTKElectromagnetViewer::Visit(KLineCurrent& aLineCurrent)
  {
    KEMField::cout<<"Pulling elements from "<<KLineCurrent::Name()<<" and putting them in VTK arrays"<<KEMField::endl;
    KEMField::cout<<aLineCurrent<<KEMField::endl;
  }

  void KEMVTKElectromagnetViewer::Visit(KCurrentLoop& aCurrentLoop)
  {
    KEMField::cout<<"Pulling elements from "<<KCurrentLoop::Name()<<" and putting them in VTK arrays"<<KEMField::endl;
    KEMField::cout<<aCurrentLoop<<KEMField::endl;
  }

  void KEMVTKElectromagnetViewer::Visit(KSolenoid& aSolenoid)
  {
    KEMField::cout<<"Pulling elements from "<<KSolenoid::Name()<<" and putting them in VTK arrays"<<KEMField::endl;
    KEMField::cout<<aSolenoid<<KEMField::endl;
  }

  void KEMVTKElectromagnetViewer::Visit(KCoil& aCoil)
  {
    KEMField::cout<<"Pulling elements from "<<KCoil::Name()<<" and putting them in VTK arrays"<<KEMField::endl;
    KEMField::cout<<aCoil<<KEMField::endl;
  }

  void KEMVTKElectromagnetViewer::GenerateGeometryFile(std::string fileName)
  {
    KEMField::cout<<"Generating file "<<fileName<<KEMField::endl;
  }

  void KEMVTKElectromagnetViewer::ViewGeometry()
  {
    KEMField::cout<<"Rendering electromagnets to screen"<<KEMField::endl;
  }
}
