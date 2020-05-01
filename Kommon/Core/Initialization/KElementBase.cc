#include "KElementBase.hh"

#include "KAttribute.hh"
#include "KInitializationMessage.hh"

namespace katrin
{
KElementBase::KElementBase() :
    fParentElement(nullptr),
    fAttributes(nullptr),
    fChildAttribute(nullptr),
    fAttributeDepth(0),
    fElements(nullptr),
    fChildElement(nullptr),
    fElementDepth(0)
{}
KElementBase::~KElementBase() {}

void KElementBase::ProcessToken(KBeginElementToken* aToken)
{
    //initmsg_debug( "processing begin element token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if ((fElementDepth == 0) && (fAttributeDepth == 0)) {
        //look up constructor method in the map, complain and exit if not found
        auto It = fElements->find(aToken->GetValue());
        if (It == fElements->end()) {
            // show only elements with similar beginning, and limit to maximum of 10 entries
            std::string token = aToken->GetValue();
            std::set<std::string> elements;
            while (token.length() > 0) {
                for (auto& it : GetElements()) {
                    if (elements.size() >= 10)
                        break;
                    if (it.substr(0, token.length()) == token)
                        elements.insert(it);
                }
                token.resize(token.size() - 1);
            }

            initmsg(eWarning) << "nothing registered for element <" << aToken->GetValue() << "> in element <"
                              << GetName() << ">" << ret;
            initmsg(eWarning) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                              << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << ret;
            initmsg(eWarning) << "available elements: " << elements;
            if (elements.size() < fElements->size())
                initmsg(eWarning) << " and " << (fElements->size() - elements.size()) << " more";
            initmsg(eWarning) << eom;

            fChild = nullptr;
            fChildElement = nullptr;
            fElementDepth = 1;
            return;
        }

        //construct and label element
        KElementBase* tChildElement = (It->second)(this);
        tChildElement->SetName(aToken->GetValue());

        //begin element
        if (tChildElement->Begin() == false) {
            initmsg(eError) << "could not begin element <" << aToken->GetValue() << ">" << ret;
            initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                            << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        fChild = tChildElement;
        fChildElement = tChildElement;
        fElementDepth = 1;
        return;
    }
    else {
        fElementDepth++;
        KProcessor::ProcessToken(aToken);
        return;
    }
}
void KElementBase::ProcessToken(KBeginAttributeToken* aToken)
{
    //initmsg_debug( "processing begin attribute token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if ((fElementDepth == 0) && (fAttributeDepth == 0)) {
        //look up constructor method in the map, complain and exit if not found
        auto It = fAttributes->find(aToken->GetValue());
        if (It == fAttributes->end()) {
            initmsg(eError) << "nothing registered for attribute <" << aToken->GetValue() << "> in element <"
                            << GetName() << ">" << ret;
            initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                            << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << ret;
            initmsg(eError) << "available attributes: " << GetAttributes() << eom;
            return;
        }

        //construct attribute
        KAttributeBase* tChildAttribute = (It->second)(this);

        //label attribute
        tChildAttribute->SetName(aToken->GetValue());

        fChild = tChildAttribute;
        fChildAttribute = tChildAttribute;
        fAttributeDepth = 1;
        return;
    }
    else {
        KProcessor::ProcessToken(aToken);
        return;
    }
}
void KElementBase::ProcessToken(KEndAttributeToken* aToken)
{
    //initmsg_debug( "processing end attribute token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if ((fAttributeDepth == 1) && (fElementDepth == 0)) {
        if (fChildAttribute) {
            //add attribute to this element
            if (AddAttribute(fChildAttribute) == false) {
                initmsg(eError) << "element <" << GetName() << "> could not process attribute <"
                                << fChildAttribute->GetName() << ">" << ret;
                initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                                << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">"
                                << eom;
            }

            //delete attribute
            delete fChildAttribute;
        }

        //reset child information
        fChild = nullptr;
        fChildAttribute = nullptr;
        fAttributeDepth = 0;
        return;
    }
    else {
        KProcessor::ProcessToken(aToken);
        return;
    }
}
void KElementBase::ProcessToken(KMidElementToken* aToken)
{
    //initmsg_debug( "processing mid element token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if ((fAttributeDepth == 0) && (fElementDepth == 1)) {
        if (fChildElement) {
            //start body of child element
            if (fChildElement->Body() == false) {
                initmsg(eError) << "could not begin body of element <" << aToken->GetValue() << ">" << ret;
                initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                                << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">"
                                << eom;
                return;
            }
        }

        //nothing else to do
        return;
    }
    else {
        KProcessor::ProcessToken(aToken);
        return;
    }
}
void KElementBase::ProcessToken(KElementDataToken* aToken)
{
    //initmsg_debug( "processing element data token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if (fElementDepth == 0) {
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
    else {
        KProcessor::ProcessToken(aToken);
    }
}
void KElementBase::ProcessToken(KEndElementToken* aToken)
{
    //initmsg_debug( "processing end element token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if ((fAttributeDepth == 0) && (fElementDepth == 1)) {
        if (fChildElement) {
            //end child element
            if (fChildElement->End() == false) {
                initmsg(eError) << "could not end child element <" << aToken->GetValue() << ">" << ret;
                initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                                << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">"
                                << eom;
                return;
            }

            //add child element to this element
            if (AddElement(fChildElement) == false) {
                initmsg(eError) << "element <" << GetName() << "> could not process element <" << aToken->GetValue()
                                << ">" << ret;
                initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                                << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">"
                                << eom;
                return;
            }

            //delete child element
            delete fChildElement;
        }

        fChild = nullptr;
        fChildElement = nullptr;
        fElementDepth = 0;
        return;
    }
    else {
        fElementDepth--;
        KProcessor::ProcessToken(aToken);
        return;
    }
}
void KElementBase::ProcessToken(KErrorToken* aToken)
{
    //initmsg_debug( "processing error token <" << aToken->GetValue() << "> in element <" << GetName() << ">" << eom );
    if (fElementDepth == 0 && fAttributeDepth == 0) {
        initmsg(eError) << "element <" << GetName() << "> encountered an error <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
        return;
    }
    else {
        KProcessor::ProcessToken(aToken);
        return;
    }
}

}  // namespace katrin
