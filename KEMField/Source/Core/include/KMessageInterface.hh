#ifndef KMESSAGEINTERFACE_DEF
#define KMESSAGEINTERFACE_DEF

#include <ostream>

#include "KMessage.h"

#include "KDataDisplay.hh"

namespace KEMField
{
  class KMessage_KEMField;

  class KMessage_KEMField : public katrin::KMessage
  {
  public:
    KMessage_KEMField();
    KMessage_KEMField(std::streambuf*);
    virtual ~KMessage_KEMField();

    void flush() { GetStream()<<katrin::reom; }

    katrin::KMessage& GetStream() { return *this; }
  };

  using katrin::eDebug;
  using katrin::eNormal;
  using katrin::eWarning;
  using katrin::eError;

  using katrin::ret;
  using katrin::rret;
  using katrin::eom;
  using katrin::reom;

  // define the custom endl for this stream
  template <>
  inline KDataDisplay<KMessage_KEMField>& endl(KDataDisplay<KMessage_KEMField>& myDisplay)
  {
    myDisplay.GetStream()<<eom;
    return myDisplay;
  }

}

#endif /* KMESSAGEINTERFACE_DEF */
