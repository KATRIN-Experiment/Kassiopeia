#include "KChattyProcessor.hh"
#include "KCommandLineTokenizer.hh"
#include "KComplexElement.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KMessage.h"
#include "KPrintProcessor.hh"
#include "KSerializationProcessor.hh"
#include "KTextFile.h"
#include "KVariableProcessor.hh"
#include "KXMLTokenizer.hh"
#include "UnitTest.h"

using namespace katrin;

namespace katrin
{

class TestChild
{
  public:
    TestChild() : fName("(anonymous)") {}
    virtual ~TestChild()
    {
        std::cout << "a test child is destroyed" << std::endl;
    }

    void SetName(const std::string& aName)
    {
        fName = aName;
        return;
    }

    void SetValue(const int& aValue)
    {
        fValue = aValue;
        return;
    }

    void Print(std::ostream& output) const
    {
        output << "  *" << fName << ", " << fValue << std::endl;
        return;
    }

  private:
    std::string fName;
    int fValue{0};
};

typedef KComplexElement<TestChild> KiTestChildElement;

template<> bool KiTestChildElement::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &TestChild::SetName);
        return true;
    }
    if (anAttribute->GetName() == "value") {
        anAttribute->CopyTo(fObject, &TestChild::SetValue);
        return true;
    }
    return false;
}

template<> bool KiTestChildElement::End()
{
    fObject->Print(std::cout);
    return true;
}

static const int __attribute__((unused)) sTestChildElementStructure =
    KiTestChildElement::Attribute<std::string>("name") + KiTestChildElement::Attribute<int>("value");

class TestParent : public KContainer
{
  public:
    TestParent() = default;
    ~TestParent() override
    {
        for (auto tIter : fChildren) {
            delete tIter;
        }
        std::cout << "a test parent is destroyed" << std::endl;
    }

    void SetName(const std::string& aName)
    {
        fName = aName;
        return;
    }
    void AddChild(const TestChild* aChild)
    {
        fChildren.push_back(aChild);
    }

    void Print(std::ostream& output) const override
    {
        output << "*" << fName << std::endl;
        for (auto tIter : fChildren) {
            tIter->Print(output);
        }
        return;
    }

  private:
    std::string fName;
    std::vector<const TestChild*> fChildren;
};

using KiTestParentElement = KComplexElement<TestParent>;

template<> bool KiTestParentElement::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &TestParent::SetName);
        return true;
    }
    return false;
}

template<> bool KiTestParentElement::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "child") {
        anElement->ReleaseTo(fObject, &TestParent::AddChild);
        return true;
    }
    return false;
}

template<> bool KiTestParentElement::End()
{
    fObject->Print(std::cout);
    return true;
}

STATICINT sTestParentElementStructure =
    KiTestParentElement::Attribute<std::string>("name") + KiTestParentElement::ComplexElement<TestChild>("child");

STATICINT sTestParentElement = KElementProcessor::ComplexElement<TestParent>("parent");

}  // namespace katrin

TEST(XML, Includes)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLIncludes.xml");

    // resolve file path
    tFile->Open(KFile::eRead);
    const std::string configPath = tFile->GetPath();
    tFile->Close();

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KIncludeProcessor tIncludeProcessor;

    // add currents file path to include processor
    tIncludeProcessor.SetPath(configPath);
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tIncludeProcessor.InsertAfter(&tVariableProcessor);
    tChattyProcessor.InsertAfter(&tIncludeProcessor);

    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Loops)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLLoops.xml");

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tIncludeProcessor.InsertAfter(&tVariableProcessor);
    tLoopProcessor.InsertAfter(&tIncludeProcessor);
    tChattyProcessor.InsertAfter(&tLoopProcessor);

    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Serialization)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLSerialization.xml");

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KSerializationProcessor tKSerializationProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tKSerializationProcessor.InsertAfter(&tVariableProcessor);

    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Tokenizer)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLTokenizer.xml");

    KXMLTokenizer tTokenizer;
    KChattyProcessor tChattyProcessor;

    tChattyProcessor.InsertAfter(&tTokenizer);
    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Variables)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eNormal);
    KMessageTable::GetInstance().SetLogVerbosity(eNormal);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLVariables.xml");

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine();

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor(tCommandLine.GetVariables());
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tChattyProcessor.InsertAfter(&tVariableProcessor);

    tTokenizer.ProcessFile(tFile);
}

#ifdef ROOT

#include "KFormulaProcessor.hh"

TEST(XML, Formulas)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLFormulas.xml");

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tChattyProcessor.InsertAfter(&tFormulaProcessor);

    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Print)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLPrint.xml");

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KPrintProcessor tPrintProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tPrintProcessor.InsertAfter(&tFormulaProcessor);

    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Conditions)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLConditions.xml");

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KConditionProcessor tConditionProcessor;
    KIncludeProcessor tIncludeProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tConditionProcessor.InsertAfter(&tFormulaProcessor);
    tIncludeProcessor.InsertAfter(&tConditionProcessor);
    tChattyProcessor.InsertAfter(&tIncludeProcessor);

    tTokenizer.ProcessFile(tFile);
}

TEST(XML, Elements)
{
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);
    KTextFile* tFile = KTextFile::CreateConfigTextFile("TestXMLElements.xml");

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter(&tTokenizer);
    tFormulaProcessor.InsertAfter(&tVariableProcessor);
    tIncludeProcessor.InsertAfter(&tFormulaProcessor);
    tLoopProcessor.InsertAfter(&tIncludeProcessor);
    tElementProcessor.InsertAfter(&tLoopProcessor);

    tTokenizer.ProcessFile(tFile);
}

#endif  // ROOT
