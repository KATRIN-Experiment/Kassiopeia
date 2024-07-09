#ifndef KSLIST_H_
#define KSLIST_H_

#include <typeinfo>

namespace Kassiopeia
{

template<class XType> class KSList
{
  public:
    KSList(const int aSize = 64);
    KSList(const KSList& aCopy);
    ~KSList();

    int End() const;
    bool Full() const;

    void ClearContent();
    void DeleteContent();

    int AddElement(XType* anElement);
    int FindElement(XType* anElement);
    int FindElementByType(XType* anElement);
    int RemoveElement(XType* anElement);
    XType* ElementAt(int& anIndex) const;

    template<class XReturnType, class XClassType> void ForEach(XReturnType (XClassType::*aMember)());
    template<class XReturnType, class XClassType> void ForEach(XReturnType (XClassType::*aMember)() const);
    template<class XReturnType, class XClassType, class XArgumentType>
    void ForEach(XReturnType (XClassType::*aMember)(XArgumentType), XArgumentType anArgument);
    template<class XReturnType, class XClassType, class XArgumentType>
    void ForEach(XReturnType (XClassType::*aMember)(XArgumentType) const, XArgumentType anArgument);
    template<class XReturnType, class XClassType, class XArgumentType1, class XArgumentType2>
    void ForEach(XReturnType (XClassType::*aMember)(XArgumentType1, XArgumentType2), XArgumentType1 anArgument1,
                 XArgumentType2 anArgument2);
    template<class XReturnType, class XClassType, class XArgumentType1, class XArgumentType2>
    void ForEach(XReturnType (XClassType::*aMember)(XArgumentType1, XArgumentType2) const, XArgumentType1 anArgument1,
                 XArgumentType2 anArgument2);

    template<class XClassType> bool ForEachUntilTrue(bool (XClassType::*aMember)());
    template<class XClassType> bool ForEachUntilTrue(bool (XClassType::*aMember)() const);
    template<class XClassType, class XArgumentType>
    bool ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType), XArgumentType anArgument);
    template<class XClassType, class XArgumentType>
    bool ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType) const, XArgumentType anArgument);
    template<class XClassType, class XArgumentType1, class XArgumentType2>
    bool ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2), XArgumentType1 anArgument1,
                          XArgumentType2 anArgument2);
    template<class XClassType, class XArgumentType1, class XArgumentType2>
    bool ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2) const, XArgumentType1 anArgument1,
                          XArgumentType2 anArgument2);

    template<class XClassType> bool ForEachUntilFalse(bool (XClassType::*aMember)());
    template<class XClassType> bool ForEachUntilFalse(bool (XClassType::*aMember)() const);
    template<class XClassType, class XArgumentType>
    bool ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType), XArgumentType anArgument);
    template<class XClassType, class XArgumentType>
    bool ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType) const, XArgumentType anArgument);
    template<class XClassType, class XArgumentType1, class XArgumentType2>
    bool ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2), XArgumentType1 anArgument1,
                           XArgumentType2 anArgument2);
    template<class XClassType, class XArgumentType1, class XArgumentType2>
    bool ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2) const,
                           XArgumentType1 anArgument1, XArgumentType2 anArgument2);

    template<class XClassType, class XReturnType> XReturnType LargestOfAll(XReturnType (XClassType::*aMember)());

    template<class XClassType, class XReturnType> XReturnType SmallestOfAll(XReturnType (XClassType::*aMember)());

    template<class XClassType, class XReturnType> XReturnType SumOfAll(XReturnType (XClassType::*aMember)());

  private:
    XType** fElements;
    int fEndElement;
    const int fLastElement;
};

template<class XType>
KSList<XType>::KSList(const int aMaxSize) : fEndElement(0), fLastElement(aMaxSize)
{
    fElements = new XType*[fLastElement];
    for (int currentElement = 0; currentElement < fLastElement; currentElement++) {
        fElements[currentElement] = nullptr;
    }
}
template<class XType>
KSList<XType>::KSList(const KSList& aCopy) : fEndElement(0), fLastElement(aCopy.fLastElement)
{
    fElements = new XType*[fLastElement];
    for (int currentElement = 0; currentElement < fLastElement; currentElement++) {
        fElements[currentElement] = aCopy.fElements[currentElement];
    }
}
template<class XType> KSList<XType>::~KSList()
{
    delete[] fElements;
}

template<class XType> int KSList<XType>::End() const
{
    return fEndElement;
}
template<class XType> bool KSList<XType>::Full() const
{
    return (fEndElement == fLastElement);
}

template<class XType> void KSList<XType>::ClearContent()
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if (fElements[currentElement] != nullptr) {
            fElements[currentElement] = nullptr;
        }
    }
    fEndElement = 0;
    return;
}
template<class XType> void KSList<XType>::DeleteContent()
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if (fElements[currentElement] != nullptr) {
            delete fElements[currentElement];
            fElements[currentElement] = nullptr;
        }
    }
    fEndElement = 0;
    return;
}

template<class XType> int KSList<XType>::AddElement(XType* anElement)
{
    if (fEndElement == fLastElement) {
        return -1;
    }
    const int currentElement = fEndElement;
    fElements[currentElement] = anElement;
    fEndElement++;
    return currentElement;
}
template<class XType> int KSList<XType>::FindElement(XType* anElement)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if (fElements[currentElement] == anElement) {
            return currentElement;
        }
    }
    return -1;
}
template<class XType> int KSList<XType>::FindElementByType(XType* anElement)
{
    const std::type_info& elementType = typeid(*anElement);
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        const std::type_info& compareType = typeid(*fElements[currentElement]);
        if (elementType == compareType) {
            return currentElement;
        }
    }
    return -1;
}
template<class XType> int KSList<XType>::RemoveElement(XType* anElement)
{
    int tIndex = -1;
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if (tIndex == -1) {
            if (fElements[currentElement] == anElement) {
                fElements[currentElement] = nullptr;
                tIndex = currentElement;
            }
            continue;
        }
        fElements[currentElement - 1] = fElements[currentElement];
    }
    if (tIndex != -1) {
        fEndElement--;
        fElements[fEndElement] = nullptr;
    }
    return tIndex;
}
template<class XType> XType* KSList<XType>::ElementAt(int& anIndex) const
{
    if (anIndex >= fEndElement || anIndex < 0) {
        return nullptr;
    }
    return fElements[anIndex];
}

template<class XType>
template<class XReturnType, class XClassType>
void KSList<XType>::ForEach(XReturnType (XClassType::*aMember)())
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        (fElements[currentElement]->*aMember)();
    }
    return;
}
template<class XType>
template<class XReturnType, class XClassType>
void KSList<XType>::ForEach(XReturnType (XClassType::*aMember)() const)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        (fElements[currentElement]->*aMember)();
    }
    return;
}
template<class XType>
template<class XReturnType, class XClassType, class XArgumentType>
void KSList<XType>::ForEach(XReturnType (XClassType::*aMember)(XArgumentType), XArgumentType anArgument)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        (fElements[currentElement]->*aMember)(anArgument);
    }
    return;
}
template<class XType>
template<class XReturnType, class XClassType, class XArgumentType>
void KSList<XType>::ForEach(XReturnType (XClassType::*aMember)(XArgumentType) const, XArgumentType anArgument)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        (fElements[currentElement]->*aMember)(anArgument);
    }
    return;
}

template<class XType>
template<class XReturnType, class XClassType, class XArgumentType1, class XArgumentType2>
void KSList<XType>::ForEach(XReturnType (XClassType::*aMember)(XArgumentType1, XArgumentType2),
                            XArgumentType1 anArgument1, XArgumentType2 anArgument2)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        (fElements[currentElement]->*aMember)(anArgument1, anArgument2);
    }
    return;
}
template<class XType>
template<class XReturnType, class XClassType, class XArgumentType1, class XArgumentType2>
void KSList<XType>::ForEach(XReturnType (XClassType::*aMember)(XArgumentType1, XArgumentType2) const,
                            XArgumentType1 anArgument1, XArgumentType2 anArgument2)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        (fElements[currentElement]->*aMember)(anArgument1, anArgument2);
    }
    return;
}

template<class XType> template<class XClassType> bool KSList<XType>::ForEachUntilTrue(bool (XClassType::*aMember)())
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)() == true) {
            return true;
        }
    }
    return false;
}
template<class XType>
template<class XClassType>
bool KSList<XType>::ForEachUntilTrue(bool (XClassType::*aMember)() const)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)() == true) {
            return true;
        }
    }
    return false;
}
template<class XType>
template<class XClassType, class XArgumentType>
bool KSList<XType>::ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType), XArgumentType anArgument)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument) == true) {
            return true;
        }
    }
    return false;
}
template<class XType>
template<class XClassType, class XArgumentType>
bool KSList<XType>::ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType) const, XArgumentType anArgument)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument) == true) {
            return true;
        }
    }
    return false;
}
template<class XType>
template<class XClassType, class XArgumentType1, class XArgumentType2>
bool KSList<XType>::ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2),
                                     XArgumentType1 anArgument1, XArgumentType2 anArgument2)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument1, anArgument2) == true) {
            return true;
        }
    }
    return false;
}
template<class XType>
template<class XClassType, class XArgumentType1, class XArgumentType2>
bool KSList<XType>::ForEachUntilTrue(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2) const,
                                     XArgumentType1 anArgument1, XArgumentType2 anArgument2)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument1, anArgument2) == true) {
            return true;
        }
    }
    return false;
}

template<class XType> template<class XClassType> bool KSList<XType>::ForEachUntilFalse(bool (XClassType::*aMember)())
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)() == false) {
            return false;
        }
    }
    return true;
}
template<class XType>
template<class XClassType>
bool KSList<XType>::ForEachUntilFalse(bool (XClassType::*aMember)() const)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)() == false) {
            return false;
        }
    }
    return true;
}
template<class XType>
template<class XClassType, class XArgumentType>
bool KSList<XType>::ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType), XArgumentType anArgument)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument) == false) {
            return false;
        }
    }
    return true;
}
template<class XType>
template<class XClassType, class XArgumentType>
bool KSList<XType>::ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType) const, XArgumentType anArgument)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument) == false) {
            return false;
        }
    }
    return true;
}
template<class XType>
template<class XClassType, class XArgumentType1, class XArgumentType2>
bool KSList<XType>::ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2),
                                      XArgumentType1 anArgument1, XArgumentType2 anArgument2)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument1, anArgument2) == false) {
            return false;
        }
    }
    return true;
}
template<class XType>
template<class XClassType, class XArgumentType1, class XArgumentType2>
bool KSList<XType>::ForEachUntilFalse(bool (XClassType::*aMember)(XArgumentType1, XArgumentType2) const,
                                      XArgumentType1 anArgument1, XArgumentType2 anArgument2)
{
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        if ((fElements[currentElement]->*aMember)(anArgument1, anArgument2) == false) {
            return false;
        }
    }
    return true;
}

template<class XType>
template<class XClassType, class XReturnType>
XReturnType KSList<XType>::LargestOfAll(XReturnType (XClassType::*aMember)())
{
    XReturnType Largest = -1.e300;
    XReturnType CurrentValue;
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        CurrentValue = (fElements[currentElement]->*aMember)();
        if (CurrentValue > Largest) {
            Largest = CurrentValue;
        }
    }
    return Largest;
}

template<class XType>
template<class XClassType, class XReturnType>
XReturnType KSList<XType>::SmallestOfAll(XReturnType (XClassType::*aMember)())
{
    XReturnType Smallest = 1.e300;
    XReturnType CurrentValue;
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        CurrentValue = (fElements[currentElement]->*aMember)();
        if (CurrentValue < Smallest) {
            Smallest = CurrentValue;
        }
    }
    return Smallest;
}

template<class XType>
template<class XClassType, class XReturnType>
XReturnType KSList<XType>::SumOfAll(XReturnType (XClassType::*aMember)())
{
    XReturnType Sum = 0.;
    for (int currentElement = 0; currentElement < fEndElement; currentElement++) {
        Sum += (fElements[currentElement]->*aMember)();
    }
    return Sum;
}

}  // namespace Kassiopeia

#endif /* KSLIST_H_ */
