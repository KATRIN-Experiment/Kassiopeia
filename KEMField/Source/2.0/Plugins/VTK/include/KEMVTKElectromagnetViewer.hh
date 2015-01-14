#ifndef KEMVTKELECTROMAGNETVIEWER_DEF
#define KEMVTKELECTROMAGNETVIEWER_DEF

#include "KTypeManipulation.hh"

#include "KLineCurrent.hh"
#include "KCurrentLoop.hh"
#include "KSolenoid.hh"
#include "KCoil.hh"

#include "KElectromagnetVisitor.hh"
#include "KElectromagnetContainer.hh"

namespace KEMField
{

  /**
   * @class KEMVTKElectromagnetViewer
   *
   * @brief A class for rendering electromagnets with VTK. 
   *
   * KEMVTKViewer is a class for rendering electromagnets with VTK.  The result
   * can be saved to a file, or displayed on the screen.
   *
   * @author Stefan Groh
   */

  class KEMVTKElectromagnetViewer : public KElectromagnetVisitor
  {
  public:
    KEMVTKElectromagnetViewer(KElectromagnetContainer& anElectromagnetContainer);
    ~KEMVTKElectromagnetViewer() {}

    void GenerateGeometryFile(std::string fileName = "Electromagnets.vtp");

    void ViewGeometry();

  private:

    void Visit(KLineCurrent&);
    void Visit(KCurrentLoop&);
    void Visit(KSolenoid&);
    void Visit(KCoil&);

  };

}

#endif /* KEMVTKELECTROMAGNETVIEWER_DEF */
