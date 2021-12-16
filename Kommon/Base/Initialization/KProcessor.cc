#include "KProcessor.hh"

#include <cstdlib>

namespace katrin
{

KProcessor::KProcessor() : fParent(nullptr), fChild(nullptr) {}
KProcessor::~KProcessor() = default;

void KProcessor::Connect(KProcessor* aParent, KProcessor* aChild)
{
    if ((aParent == nullptr) || (aChild == nullptr)) {
        return;
    }

    if ((aParent->fChild != nullptr) || (aChild->fParent != nullptr)) {
        return;
    }

    aParent->fChild = aChild;
    aChild->fParent = aParent;

    return;
}
void KProcessor::Disconnect(KProcessor* aParent, KProcessor* aChild)
{
    if ((aParent == nullptr) || (aChild == nullptr)) {
        return;
    }

    if ((aParent->fChild != aChild) || (aChild->fParent != aParent)) {
        return;
    }

    aParent->fChild = nullptr;
    aChild->fParent = nullptr;

    return;
}

void KProcessor::InsertBefore(KProcessor* aTarget)
{
    if ((fParent != nullptr) || (fChild != nullptr) || (aTarget == nullptr)) {
        return;
    }

    if (aTarget->fParent != nullptr) {
        fParent = aTarget->fParent;
        fParent->fChild = this;
    }

    fChild = aTarget;
    aTarget->fParent = this;

    return;
}
void KProcessor::InsertAfter(KProcessor* aTarget)
{
    if ((fParent != nullptr) || (fChild != nullptr) || (aTarget == nullptr)) {
        return;
    }

    if (aTarget->fChild != nullptr) {
        fChild = aTarget->fChild;
        fChild->fParent = this;
    }

    fParent = aTarget;
    aTarget->fChild = this;

    return;
}
void KProcessor::Remove()
{
    if ((fParent != nullptr) && (fChild != nullptr)) {
        fParent->fChild = fChild;
        fChild->fParent = fParent;

        fParent = nullptr;
        fChild = nullptr;

        return;
    }

    if (fParent != nullptr) {
        fParent->fChild = nullptr;
        fParent = nullptr;
    }

    if (fChild != nullptr) {
        fChild->fParent = nullptr;
        fChild = nullptr;
    }

    return;
}

KProcessor* KProcessor::GetFirstParent()
{
    if (fParent != nullptr) {
        return fParent->GetFirstParent();
    }
    return this;
}
KProcessor* KProcessor::GetParent()
{
    return fParent;
}

KProcessor* KProcessor::GetLastChild()
{
    if (fChild != nullptr) {
        return fChild->GetLastChild();
    }
    return this;
}
KProcessor* KProcessor::GetChild()
{
    return fChild;
}

void KProcessor::ProcessToken(KBeginParsingToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KBeginFileToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KBeginElementToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KBeginAttributeToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KAttributeDataToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KEndAttributeToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KMidElementToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KElementDataToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KEndElementToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KEndFileToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KEndParsingToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KCommentToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}
void KProcessor::ProcessToken(KErrorToken* aToken)
{
    if (fChild == nullptr) {
        return;
    }
    else {
        fChild->ProcessToken(aToken);
        return;
    }
}

}  // namespace katrin
