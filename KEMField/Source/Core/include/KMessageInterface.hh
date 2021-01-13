#ifndef KMESSAGEINTERFACE_DEF
#define KMESSAGEINTERFACE_DEF

#include "KDataDisplay.hh"
#include "KMessage.h"

#include <ostream>

namespace KEMField
{
class KMessage_KEMField;

class KMessage_KEMField : public katrin::KMessage
{
  public:
    KMessage_KEMField();
    KMessage_KEMField(std::streambuf*);
    ~KMessage_KEMField() override;

    void flush()
    {
        GetStream() << katrin::reom;
    }

    katrin::KMessage& GetStream()
    {
        return *this;
    }
};


using katrin::eom;

// define the custom endl for this stream
template<> inline KDataDisplay<KMessage_KEMField>& endl(KDataDisplay<KMessage_KEMField>& myDisplay)
{
    myDisplay.GetStream() << eom;
    return myDisplay;
}

}  // namespace KEMField

#endif /* KMESSAGEINTERFACE_DEF */
