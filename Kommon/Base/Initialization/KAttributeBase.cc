#include "KAttributeBase.hh"

#include "KInitializationMessage.hh"

namespace katrin
{

KAttributeBase::KAttributeBase() : fParentElement(nullptr) {}
KAttributeBase::~KAttributeBase() = default;

void KAttributeBase::ProcessToken(KAttributeDataToken* aToken)
{
    //add value to this element
    if (SetValue(aToken) == false) {
        initmsg(eError) << "element <" << GetName() << "> could not process value <" << aToken->GetValue() << ">"
                        << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    return;
}
void KAttributeBase::ProcessToken(KErrorToken* aToken)
{
    initmsg(eError) << "element <" << GetName() << "> encountered an error <" << aToken->GetValue() << ">" << ret;
    initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                    << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
    return;
}

}  // namespace katrin
