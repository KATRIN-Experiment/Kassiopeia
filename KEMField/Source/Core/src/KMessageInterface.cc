#include "KMessageInterface.hh"

#include <cmath>

namespace KEMField
{

KMessage_KEMField::KMessage_KEMField() : katrin::KMessage("KEMField", "KEMFIELD", "", "") {}

KMessage_KEMField::KMessage_KEMField(std::streambuf*) : katrin::KMessage("KEMField", "KEMFIELD", "", "") {}

KMessage_KEMField::~KMessage_KEMField() {}

}  // namespace KEMField
