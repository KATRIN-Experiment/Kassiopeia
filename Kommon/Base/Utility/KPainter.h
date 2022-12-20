#ifndef _katrin_KPainter_h_
#define _katrin_KPainter_h_

#include "KNamed.h"

namespace katrin
{

class KWindow;

class KPainter : public KNamed
{
  public:
    KPainter();
    ~KPainter() override;

  public:
    virtual void Render() = 0;
    virtual void Display() = 0;
    virtual void Write() = 0;

    virtual void SetWindow(KWindow* aWindow) = 0;
    virtual void ClearWindow(KWindow* aWindow) = 0;
};

}  // namespace katrin

#endif
