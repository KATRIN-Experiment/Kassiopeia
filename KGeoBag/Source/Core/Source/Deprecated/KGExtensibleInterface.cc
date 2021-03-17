#include "KGExtensibleInterface.hh"

namespace KGeoBag
{

KGExtensibleInterface::KGExtensibleInterface() : fLocked(false), fPrevious(NULL), fNext(NULL) {}
KGExtensibleInterface::KGExtensibleInterface(const KGExtensibleInterface& aCopy) :
    fLocked(false),
    fPrevious(NULL),
    fNext(NULL)
{}
KGExtensibleInterface::~KGExtensibleInterface()
{
    if (fLocked == false) {
        Lock();
        delete Head();
    }
    else {
        delete fNext;
    }
}

KGExtensibleInterface* KGExtensibleInterface::Head()
{
    if (fPrevious != NULL) {
        return fPrevious->Head();
    }
    else {
        return this;
    }
}
const KGExtensibleInterface* KGExtensibleInterface::Head() const
{
    if (fPrevious != NULL) {
        return fPrevious->Head();
    }
    else {
        return this;
    }
}
KGExtensibleInterface* KGExtensibleInterface::Tail()
{
    if (fNext != NULL) {
        return fNext->Tail();
    }
    else {
        return this;
    }
}
const KGExtensibleInterface* KGExtensibleInterface::Tail() const
{
    if (fNext != NULL) {
        return fNext->Tail();
    }
    else {
        return this;
    }
}

void KGExtensibleInterface::Lock() const
{
    fLocked = true;
    LockPrevious();
    LockNext();
    return;
}
void KGExtensibleInterface::LockPrevious() const
{
    if (fPrevious != NULL) {
        fPrevious->fLocked = true;
        fPrevious->LockPrevious();
    }
    return;
}
void KGExtensibleInterface::LockNext() const
{
    if (fNext != NULL) {
        fNext->fLocked = true;
        fNext->LockNext();
    }
    return;
}

void KGExtensibleInterface::Unlock() const
{
    fLocked = false;
    UnlockPrevious();
    UnlockNext();
    return;
}
void KGExtensibleInterface::UnlockPrevious() const
{
    if (fPrevious != NULL) {
        fPrevious->fLocked = false;
        fPrevious->UnlockPrevious();
    }
    return;
}
void KGExtensibleInterface::UnlockNext() const
{
    if (fNext != NULL) {
        fNext->fLocked = false;
        fNext->UnlockNext();
    }
    return;
}

}  // namespace KGeoBag
