#ifndef KEMVTKELECTROMAGNETVIEWER_DEF
#define KEMVTKELECTROMAGNETVIEWER_DEF

#include "KCoil.hh"
#include "KCurrentLoop.hh"
#include "KElectromagnetContainer.hh"
#include "KElectromagnetVisitor.hh"
#include "KLineCurrent.hh"
#include "KSolenoid.hh"
#include "KTypeManipulation.hh"

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
    ~KEMVTKElectromagnetViewer() override = default;

    void GenerateGeometryFile(const std::string& fileName = "Electromagnets.vtp");

    void ViewGeometry();

  private:
    void Visit(KLineCurrent&) override;
    void Visit(KCurrentLoop&) override;
    void Visit(KSolenoid&) override;
    void Visit(KCoil&) override;
};

}  // namespace KEMField

#endif /* KEMVTKELECTROMAGNETVIEWER_DEF */
