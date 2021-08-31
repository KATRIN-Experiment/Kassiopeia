#ifndef _katrin_KVTKPainter_h_
#define _katrin_KVTKPainter_h_

#include "KPainter.h"

#include <vtkObject.h>

namespace katrin
{
class KVTKWindow;

class KVTKPainter : public KPainter
{
  public:
    KVTKPainter();
    ~KVTKPainter() override;

  public:
    void SetWindow(KWindow* aWindow) override;
    void ClearWindow(KWindow* aWindow) override;

    void SetDisplayMode(bool aMode);
    void SetWriteMode(bool aMode);

    virtual std::string HelpText();
    virtual void OnKeyPress(vtkObject* /*caller*/, long unsigned int /*eventId*/, void* /*client*/, void* /*callData*/);

  protected:
    KVTKWindow* fWindow;
    bool fDisplayEnabled;
    bool fWriteEnabled;
};

}  // namespace katrin

#endif
