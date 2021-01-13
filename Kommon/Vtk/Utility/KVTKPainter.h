#ifndef _katrin_KVTKPainter_h_
#define _katrin_KVTKPainter_h_

#include "KPainter.h"

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

  protected:
    KVTKWindow* fWindow;
    bool fDisplayEnabled;
    bool fWriteEnabled;
};

}  // namespace katrin

#endif
