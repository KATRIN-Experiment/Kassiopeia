#ifndef Kassiopeia_KSObject_h_
#error "do not include KSOutput.h directly.  include KSObject.h instead."
#else
#ifndef Kassiopeia_KSOutput_h_
#define Kassiopeia_KSOutput_h_

namespace Kassiopeia
{

template<class XParentType, class XMemberType> class KSOutput;

template<class XParentType, class XMemberType> class KSOutputFactoryTemplate;

//**********************
//const-reference getter
//**********************

template<class XParentType, class XChildType>
class KSOutput<XParentType, const XChildType& (XParentType::*) (void) const> : public KSObject
{
  public:
    KSOutput(KSObject* aParentObject, XParentType* aParentPointer,
             const XChildType& (XParentType::*aMember)(void) const) :
        KSObject(),
        fState(sActivated),
        fParentObject(aParentObject),
        fChildObjects(),
        fUpdated(false),
        fTarget(aParentPointer),
        fMember(aMember),
        fValue()
    {
        Set(&fValue);
    }
    KSOutput(const KSOutput<XParentType, const XChildType& (XParentType::*) (void) const>& aCopy) :
        KSObject(aCopy),
        fState(aCopy.fState),
        fParentObject(aCopy.fParentObject),
        fChildObjects(aCopy.fChildObjects),
        fUpdated(true),
        fTarget(aCopy.fTarget),
        fMember(aCopy.fMember),
        fValue(aCopy.fValue)
    {
        Set(&fValue);
    }
    virtual ~KSOutput() {}

    //********
    //KSObject
    //********

  public:
    KSOutput* Clone() const
    {
        return new KSOutput<XParentType, const XChildType& (XParentType::*) (void) const>(*this);
    }
    KSObject* Command(KSObject*, const std::string&)
    {
        return NULL;
    }
    KSObject* Output(const std::string& aLabel)
    {
        KSObject* tChild = KSDictionary<XChildType>::GetOutput(this, aLabel);
        if (tChild == NULL) {
            objctmsg(eError) << "const-reference output <" << this->GetName() << "> has no output named <" << aLabel
                             << ">" << eom;
        }
        else {
            fChildObjects.push_back(tChild);
        }
        return tChild;
    }
    void PushUpdate()
    {
        if (fUpdated == false) {
            fValue = (fTarget->*fMember)();
            fUpdated = true;
            for (std::vector<KSObject*>::iterator tIt = fChildObjects.begin(); tIt != fChildObjects.end(); tIt++) {
                (*tIt)->PushUpdate();
            }
        }
        return;
    }
    void PushReset()
    {
        if (fUpdated == true) {
            fUpdated = false;
            for (std::vector<KSObject*>::iterator tIt = fChildObjects.begin(); tIt != fChildObjects.end(); tIt++) {
                (*tIt)->PushReset();
            }
        }
        return;
    }
    void PullUpdate()
    {
        if (fUpdated == false) {
            fParentObject->PullUpdate();
            fValue = (fTarget->*fMember)();
            fUpdated = true;
        }
        return;
    }
    void PullReset()
    {
        if (fUpdated == true) {
            fParentObject->PullReset();
            fUpdated = false;
        }
        return;
    }

    const unsigned int& State() const
    {
        return fParentObject->State();
    }

    void Initialize()
    {
        if (fParentObject->State() != sInitialized) {
            objctmsg(eError) << "tried to initialize const-reference output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }
    void Deinitialize()
    {
        if (fParentObject->State() != sInitialized) {
            objctmsg(eError) << "tried to deinitialize const-reference output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }
    void Activate()
    {
        if (fParentObject->State() != sActivated) {
            objctmsg(eError) << "tried to activate const-reference output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }
    void Deactivate()
    {
        if (fParentObject->State() != sActivated) {
            objctmsg(eError) << "tried to deactivate const-reference output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }

  private:
    unsigned int fState;
    KSObject* fParentObject;
    std::vector<KSObject*> fChildObjects;
    bool fUpdated;
    XParentType* fTarget;
    const XChildType& (XParentType::*fMember)(void) const;
    XChildType fValue;
};

template<class XParentType, class XChildType>
class KSOutputFactoryTemplate<XParentType, const XChildType& (XParentType::*) () const> : public KSOutputFactory
{
  public:
    KSOutputFactoryTemplate(const XChildType& (XParentType::*aMember)() const) : fMember(aMember) {}
    virtual ~KSOutputFactoryTemplate() {}

  public:
    KSObject* CreateOutput(KSObject* aParent) const
    {
        XParentType* tParent = aParent->As<XParentType>();
        if (tParent == NULL) {
            objctmsg_debug("output parent <" << aParent->GetName() << "> could not be cast to type <"
                                             << typeid(XParentType).name() << ">" << eom);
            return NULL;
        }

        objctmsg_debug("output built" << eom);
        return new KSOutput<XParentType, const XChildType& (XParentType::*) () const>(aParent, tParent, fMember);
    }

  private:
    const XChildType& (XParentType::*fMember)() const;
};

//*****************
//copy-value getter
//*****************

template<class XParentType, class XChildType>
class KSOutput<XParentType, XChildType (XParentType::*)(void) const> : public KSObject
{
  public:
    KSOutput(KSObject* aParentObject, XParentType* aParentPointer, XChildType (XParentType::*aMember)(void) const) :
        KSObject(),
        fState(sActivated),
        fParentObject(aParentObject),
        fChildObjects(),
        fUpdated(false),
        fTarget(aParentPointer),
        fMember(aMember),
        fValue()
    {
        Set(&fValue);
    }
    KSOutput(const KSOutput<XParentType, XChildType (XParentType::*)(void) const>& aCopy) :
        KSObject(aCopy),
        fState(aCopy.fState),
        fParentObject(aCopy.fParentObject),
        fChildObjects(aCopy.fChildObjects),
        fUpdated(false),
        fTarget(aCopy.fTarget),
        fMember(aCopy.fMember),
        fValue(aCopy.fValue)
    {
        Set(&fValue);
    }
    virtual ~KSOutput() {}

    //********
    //KSObject
    //********

  public:
    KSOutput* Clone() const
    {
        return new KSOutput<XParentType, XChildType (XParentType::*)(void) const>(*this);
    }
    KSObject* Command(KSObject*, const std::string&)
    {
        return NULL;
    }
    KSObject* Output(const std::string& aLabel)
    {
        KSObject* tChild = KSDictionary<XChildType>::GetOutput(this, aLabel);
        if (tChild == NULL) {
            objctmsg(eError) << "copy-value output <" << this->GetName() << "> has no output named <" << aLabel << ">"
                             << eom;
        }
        else {
            fChildObjects.push_back(tChild);
        }
        return tChild;
    }
    void PushUpdate()
    {
        if (fUpdated == false) {
            fValue = (fTarget->*fMember)();
            fUpdated = true;
            for (std::vector<KSObject*>::iterator tIt = fChildObjects.begin(); tIt != fChildObjects.end(); tIt++) {
                (*tIt)->PushUpdate();
            }
        }
        return;
    }
    void PushReset()
    {
        if (fUpdated == true) {
            fUpdated = false;
            for (std::vector<KSObject*>::iterator tIt = fChildObjects.begin(); tIt != fChildObjects.end(); tIt++) {
                (*tIt)->PushReset();
            }
        }
        return;
    }
    void PullUpdate()
    {
        if (fUpdated == false) {
            fParentObject->PullUpdate();
            fValue = (fTarget->*fMember)();
            fUpdated = true;
        }
        return;
    }
    void PullReset()
    {
        if (fUpdated == true) {
            fParentObject->PullReset();
            fUpdated = false;
        }
        return;
    }

    const unsigned int& State() const
    {
        return fParentObject->State();
    }

    void Initialize()
    {
        if (fParentObject->State() != sInitialized) {
            objctmsg(eError) << "tried to initialize copy-value output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }
    void Deinitialize()
    {
        if (fParentObject->State() != sIdle) {
            objctmsg(eError) << "tried to deinitialize copy-value output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }
    void Activate()
    {
        if (fParentObject->State() != sActivated) {
            objctmsg(eError) << "tried to activate copy-value output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }
    void Deactivate()
    {
        if (fParentObject->State() != sInitialized) {
            objctmsg(eError) << "tried to deactivate copy-value output <" << this->GetName() << "> from state <"
                             << this->State() << ">" << eom;

            return;
        }

        return;
    }

  private:
    unsigned int fState;
    KSObject* fParentObject;
    std::vector<KSObject*> fChildObjects;
    bool fUpdated;
    XParentType* fTarget;
    XChildType (XParentType::*fMember)(void) const;
    XChildType fValue;
};

template<class XParentType, class XChildType>
class KSOutputFactoryTemplate<XParentType, XChildType (XParentType::*)() const> : public KSOutputFactory
{
  public:
    KSOutputFactoryTemplate(XChildType (XParentType::*aMember)() const) : fMember(aMember) {}
    virtual ~KSOutputFactoryTemplate() {}

  public:
    KSObject* CreateOutput(KSObject* aParent) const
    {
        XParentType* tParent = aParent->As<XParentType>();
        if (tParent == NULL) {
            objctmsg_debug("output parent <" << aParent->GetName() << "> could not be cast to type <"
                                             << typeid(XParentType).name() << ">" << eom);
            return NULL;
        }

        objctmsg_debug("output built" << eom);
        return new KSOutput<XParentType, XChildType (XParentType::*)() const>(aParent, tParent, fMember);
    }

  private:
    XChildType (XParentType::*fMember)() const;
};

//**********
//dictionary
//**********

template<class XType>
template<class XMemberType>
int KSDictionary<XType>::AddOutput(XMemberType aMember, const std::string& aLabel)
{
    if (fOutputFactories == NULL) {
        fOutputFactories = new OutputFactoryMap();
    }

    KSOutputFactoryTemplate<XType, XMemberType>* tOutputMemberFactory =
        new KSOutputFactoryTemplate<XType, XMemberType>(aMember);
    fOutputFactories->insert(OutputFactoryEntry(aLabel, tOutputMemberFactory));
    return 0;
}

}  // namespace Kassiopeia

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KComplexElement.hh"

using namespace Kassiopeia;
namespace katrin
{

class KSOutputData
{
  public:
    std::string fName;
    KSObject* fParent;
    std::string fField;
};

typedef KComplexElement<KSOutputData> KSOutputBuilder;

template<> inline bool KSOutputBuilder::Begin()
{
    fObject = new KSOutputData;
    return true;
}

template<> inline bool KSOutputBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string tName = aContainer->AsReference<std::string>();
        fObject->fName = tName;
        return true;
    }
    if (aContainer->GetName() == "object") {
        KTagged* tObject = KToolbox::GetInstance().Get<KTagged>(aContainer->AsReference<std::string>());
        fObject->fParent = tObject;
        return true;
    }
    if (aContainer->GetName() == "field") {
        std::string tField = aContainer->AsReference<std::string>();
        fObject->fField = tField;
        return true;
    }
    return false;
}

template<> inline bool KSOutputBuilder::End()
{
    KSObject* tObject = fObject->fParent->Output(fObject->fField);
    tObject->SetName(fObject->fName);
    delete fObject;
    Set(tObject);
    return true;
}

}  // namespace katrin

#endif
#endif
