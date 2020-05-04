#ifndef KGCORE_HH_
#error "do not include KGSpaceFunctions.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

//**********
//extensible
//**********

template<class XExtension> inline bool KGSpace::HasExtension() const
{
    KGExtensibleSpace* tExtensible;
    KGExtendedSpace<XExtension>* tExtended;
    std::vector<KGExtensibleSpace*>::const_iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtensible = *tIt;
        tExtended = dynamic_cast<KGExtendedSpace<XExtension>*>(tExtensible);
        if (tExtended != nullptr) {
            return true;
        }
    }
    return false;
}

template<class XExtension> inline const KGExtendedSpace<XExtension>* KGSpace::AsExtension() const
{
    KGExtensibleSpace* tExtensible;
    KGExtendedSpace<XExtension>* tExtended;
    std::vector<KGExtensibleSpace*>::const_iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtensible = *tIt;
        tExtended = dynamic_cast<KGExtendedSpace<XExtension>*>(tExtensible);
        if (tExtended != NULL) {
            return tExtended;
        }
    }
    return NULL;
}

template<class XExtension> inline KGExtendedSpace<XExtension>* KGSpace::AsExtension()
{
    KGExtensibleSpace* tExtensible;
    KGExtendedSpace<XExtension>* tExtended;
    std::vector<KGExtensibleSpace*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtensible = *tIt;
        tExtended = dynamic_cast<KGExtendedSpace<XExtension>*>(tExtensible);
        if (tExtended != nullptr) {
            return tExtended;
        }
    }
    return nullptr;
}

template<class XExtension> inline KGExtendedSpace<XExtension>* KGSpace::MakeExtension()
{
    KGExtensibleSpace* tExtensible;
    KGExtendedSpace<XExtension>* tExtended;
    std::vector<KGExtensibleSpace*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtensible = *tIt;
        tExtended = dynamic_cast<KGExtendedSpace<XExtension>*>(tExtensible);
        if (tExtended != nullptr) {
            delete tExtended;
            fExtensions.erase(tIt);
            break;
        }
    }
    tExtended = new KGExtendedSpace<XExtension>(this);
    fExtensions.push_back(tExtended);
    return tExtended;
}

template<class XExtension>
inline KGExtendedSpace<XExtension>* KGSpace::MakeExtension(const typename XExtension::Space& aCopy)
{
    KGExtensibleSpace* tExtensible;
    KGExtendedSpace<XExtension>* tExtended;
    std::vector<KGExtensibleSpace*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtensible = *tIt;
        tExtended = dynamic_cast<KGExtendedSpace<XExtension>*>(tExtensible);
        if (tExtended != NULL) {
            delete tExtended;
            fExtensions.erase(tIt);
            break;
        }
    }
    tExtended = new KGExtendedSpace<XExtension>(this, aCopy);
    fExtensions.push_back(tExtended);
    return tExtended;
}

}  // namespace KGeoBag

#endif
