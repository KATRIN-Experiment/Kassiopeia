// KArgumentList.cpp //
// Author: Sanshiro Enomoto <sanshiro@uw.edu> //


#include "KArgumentList.h"

#include "KVariant.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "KLogger.h"
KLOGGER(logger, "ArgumentList");


using namespace std;
using namespace katrin;


static const bool gIsSpaceAllowedAfterEqual = true;              // if false, "--foo= bar" becomes {"--foo": ""} and ["bar"]
static const bool gIsSpaceBeforeEqualAnError = true;             // if true,  "--foo =bar"  will generate an error (next)
static const bool gIsErrorToBeThrown = false;                    // if false, errors will be logging only (and proceed; next)
static const bool gIsStartingWithEqualAlwaysAParameter = false;  // if true,  "--foo =bar" becomes {"--foo":""} & ["=bar"]

/*****
# Note
- If gIsSpaceAllowedAferEqual is true,
    - "--foo= bar" becomes {"--foo": "bar"}, 
    - however, even in this canse, if the next argv is an option, the next one becomes a separate option, 
      e.g., "--foo= --bar= buz" becomes {"--foo": "", "--bar": "buz"}.
    - Note that, "--foo=--bar=buz" (no space after equal) is still {"--foo": "--bar=buz"}

- If gIsSpaceBeforeEqualAnError is false, or gIsErrorToBeThrown is false (BE CAREFUL FOR THIS SETTING!!)
    - "--foo =bar" will be parsed:
      - if gIsStartingWithEqualAlwaysAParameter is true, it becomes {"--foo": ""} and ["=bar"].
      - otherwise, it becomes {"--foo": "bar"}
        - Also note more complex cases in this configuration:
          - "--foo =bar" -->  { "--foo": "bar" }
          - "--foo =--bar=buz"  -->  { "--foo": "--bar=buz" }
          - "--foo =--bar =buz"  -->  { "--foo": "--bar" } & [ "=buz" ]
        - if gIsSpacedAllowedAfterEqual is also true:
          - "--foo = --bar" -->  { "--foo": "", "--bar": "" }
          - "--foo = --bar = buz"  -->  { "--foo": "", "--bar": "buz" }
        - otherwise
          - "--foo = --bar" -->  { "--foo": "", "--bar": "" }
          - "--foo = --bar = buz"  -->  { "--foo": "", "--bar": "" } & ["buz"]
*****/




KArgumentList::KArgumentList(int argc, char** argv)
{
        
    if (argc > 0) {
        fProgramPathName = argv[0];
        fCommandLine = argv[0];
        string::size_type slash = fProgramPathName.find_last_of('/');
        if (slash != string::npos) {
            fProgramName = fProgramPathName.substr(slash + 1, string::npos);
        }
        else {
            fProgramName = fProgramPathName;
        }
    }

    auto IsOption = [](const string& Argument)->bool {
        if ((Argument[0] != '-') || (Argument == "-") || (Argument == "--")) {
            return false;
        }
        else if ((Argument.size() > 1) && (isdigit(Argument[1]))) {
            // negative number parameter //
            return false;
        }
        return true;
    };
    
    for (int i = 1; i < argc; i++) {
        string Argument = argv[i];
        fCommandLine += " " + Argument;
        if (! IsOption(Argument)) {
            fParameterList.push_back(Argument);
        }
        else {
            string::size_type NameLength = Argument.find_first_of('=');
            string Name = Argument.substr(0, NameLength);
            string Value = "";

            if (NameLength != string::npos) {      //// --foo=*
                // --foo=bar  -->  { "--foo": "bar" }
                // --foo=--bar=buz  -->  { "--foo": "--bar=buz" }
                // also: --foo=--bar =buz  -->  { "--foo": "--bar" } & [ "=buz" ]
                Value = Argument.substr(NameLength + 1, Argument.size());
            }

            if (Value.empty() && (i+1 < argc)) {
                if (NameLength != string::npos) {   //// --foo= *
                    if (! gIsSpaceAllowedAfterEqual) {
                        ;     // --foo= bar --> { "--foo": "" } & [ "bar" ]
                    }
                    else if (IsOption(argv[i+1])) {
                        ;     // --foo= --bar=buz  -->  { "--foo": "", "--bar": "buz" }
                    }
                    else {    // --foo= bar -->  { "--foo": "bar" }
                        i++;
                        Value = argv[i];
                    }
                }
                else if (argv[i+1][0] == '=') {     //// --foo =*
                    if (gIsSpaceBeforeEqualAnError) {
                        KERROR(logger, "arguments starting with '=' are not allowed");
                        if (gIsErrorToBeThrown) {
                            throw KException() << "arguments starting with '=' are not allowed";
                        }
                    }
                    i++;
                    if (gIsStartingWithEqualAlwaysAParameter) {
                        // --foo =bar --> {"--foo": ""} & ["=bar"]
                        fParameterList.push_back(argv[i]);
                    }
                    else if (argv[i][1] != '\0') {     //// "--foo =..."
                        // --foo =bar -->  { "--foo": "bar" }
                        // --foo =--bar=buz  -->  { "--foo": "--bar=buz" }
                        // also: --foo =--bar =buz  -->  { "--foo": "--bar" } & [ "=buz" ]
                        Value = (argv[i] + 1);
                    }
                    else {      //// --foo = ...
                        if (i+1 >= argc) {
                            // --foo =$ -->  { "--foo": "" }
                            ;
                        }
                        else if (! gIsSpaceAllowedAfterEqual) {
                            // --foo = bar --> { "--foo": "" } & [ "bar" ]
                            ;
                        }
                        else if (IsOption(argv[i+1])) {
                            // --foo = --bar -->  { "--foo": "", "--bar": "" }
                            // --foo = --bar = buz  -->  { "--foo": "", "--bar": "buz" }
                            ;
                        }
                        else {
                            // --foo = bar  --> { "--foo": "bar" }
                            i++;
                            Value = argv[i];
                        }
                    }
                }
                else {
                    // --foo  -->  { "--foo": "" }
                    ;
                }
            }

            fOptionTable[Name] = Value;
            fOptionNameList.push_back(Name);
        }
    }

    fArgvBuffer = nullptr;
    fArgvBufferSize = 0;
}

KArgumentList::~KArgumentList()
{
    if (fArgvBuffer) {
        for (unsigned i = 0; i < fArgvBufferSize; i++) {
            delete[] fArgvBuffer[i];
            delete[] fArgvBuffer;
        }
    }
}

KVariant KArgumentList::GetParameter(unsigned int Index) const
{
    if (Index >= fParameterList.size()) {
        return KVariant();
    }
    else {
        return KVariant(fParameterList[Index]);
    }
}

KVariant KArgumentList::GetOption(const std::string& Name) const
{
    fUsedOptionSet.insert(Name);
    
    auto Option = fOptionTable.find(Name);
    if (Option == fOptionTable.end()) {
        return KVariant();
    }
    else {
        return KVariant(Option->second);
    }
}

void KArgumentList::SetParameter(unsigned int Index, const std::string& Value)
{
    while (Index >= fParameterList.size()) {
        fParameterList.emplace_back("");
    }
    fParameterList[Index] = Value;
}

void KArgumentList::SetOption(const std::string& Name, const std::string& Value)
{
    if (fOptionTable.find(Name) == fOptionTable.end()) {
        fOptionNameList.push_back(Name);
    }
    fOptionTable[Name] = Value;
}

void KArgumentList::Dump(ostream& os) const
{
    os << "Parameters:" << endl;
    for (const auto& i : fParameterList) {
        os << "    " << i << endl;
    }

    os << "Options:" << endl;
    for (auto Name : fOptionNameList) {
        os << "    " << Name << ": " << fOptionTable.find(Name)->second << endl;
    }
}

void KArgumentList::PullBack(int& argc, char**& argv) const
{
    if (fArgvBuffer) {
        for (unsigned i = 0; i < fArgvBufferSize; i++) {
            delete[] fArgvBuffer[i];
            delete[] fArgvBuffer;
        }
    }

    fArgvBufferSize = fParameterList.size() + 1;
    fArgvBuffer = new char*[fArgvBufferSize];
    fArgvBuffer[0] = strdup(fProgramName.c_str());
    for (unsigned i = 0; i < fParameterList.size(); i++) {
        fArgvBuffer[i + 1] = strdup(fParameterList[i].c_str());
    }

    argc = (int) fParameterList.size() + 1;
    argv = fArgvBuffer;
}

std::vector<std::string> KArgumentList::UnusedOptionList() const
{
    std::vector<std::string> List;
    for (const auto& Option: fOptionNameList) {
        if (fUsedOptionSet.count(Option) == 0) {
            List.push_back(Option);
        }
    }

    return List;
}


KArgumentSchema::KElement::KElement(std::string Name)
{
    fName = Name;
    fIsDefaultValueEnabled = false;
}

KArgumentSchema::KElement::~KElement() = default;

KArgumentSchema::KElement& KArgumentSchema::KElement::WhichIs(const std::string& Description)
{
    fDescription = Description;
    return *this;
}

KArgumentSchema::KElement& KArgumentSchema::KElement::InTypeOf(const KVariant& Prototype)
{
    fPrototype = Prototype;
    return *this;
}

KArgumentSchema::KElement& KArgumentSchema::KElement::WithDefault(const KVariant& Prototype)
{
    fPrototype = Prototype;
    fIsDefaultValueEnabled = true;
    return *this;
}

void KArgumentSchema::KElement::Print(std::ostream& os, size_t NameWidth)
{
    os << fName;
    os << string(NameWidth - fName.size(), ' ');

    if (fIsDefaultValueEnabled) {
        os << "[default: " << fPrototype << "] ";
    }

    os << fDescription << endl;
}

void KArgumentSchema::KElement::Validate(const std::string& Value, std::string Name)
{
    if (Name.empty()) {
        Name = fName;
    }

    if (fPrototype.IsVoid()) {
        if (!Value.empty()) {
            throw KException() << "argument does not take value: " << Name;
        }
    }
    else {
        try {
            KVariant TestValue = fPrototype;
            TestValue.Assign(Value);
        }
        catch (KException& e) {
            throw KException() << "invalid argument value: " << Name << "=" << Value;
        }
    }
}

KVariant KArgumentSchema::KElement::DefaultValue() const
{
    if (fIsDefaultValueEnabled) {
        return fPrototype;
    }
    else {
        return KVariant();
    }
}


KArgumentSchema::KArgumentSchema()
{
    fNameLength = 0;

    fIsExcessAllowed = false;
    fIsUnknownAllowed = false;
    fTakesMultipleParameters = false;
    fNumberOfRequiredParameters = 0;
}

KArgumentSchema::~KArgumentSchema() = default;

KArgumentSchema::KElement& KArgumentSchema::Require(std::string Names)
{
    fNumberOfRequiredParameters = fParameterList.size() + 1;

    return Take(Names);
}

KArgumentSchema::KElement& KArgumentSchema::Take(std::string Names)
{
    if (Names.empty()) {
        //.. BUG: this is actually an error ...//
        return AddParameter(Names);
    }
    else if (Names[0] == '-') {
        return AddOption(Names);
    }
    else {
        return AddParameter(Names);
    }
}

KArgumentSchema::KElement& KArgumentSchema::TakeMultiple(std::string Names)
{
    fTakesMultipleParameters = true;
    return Take(Names);
}

KArgumentSchema::KElement& KArgumentSchema::AddParameter(std::string Name)
{
    fParameterList.push_back(KElement(Name).InTypeOf(""));
    fParameterNameList.push_back(Name);

    fNameLength = max(fNameLength, (unsigned int) Name.size());

    return fParameterList.back();
}

KArgumentSchema::KElement& KArgumentSchema::AddOption(std::string Names)
{
    unsigned int Index = fOptionList.size();
    fOptionList.emplace_back(Names);

    fNameLength = max(fNameLength, (unsigned int) Names.size());

    while (true) {
        string::size_type Start = Names.find_first_not_of(' ');
        if (Start == string::npos) {
            break;
        }
        string::size_type End = Names.find_first_of(',');
        string Name = Names.substr(Start, End - Start);
        string::size_type Cut = Name.find_first_of("= ");
        if (Cut != string::npos) {
            Name = Name.substr(0, Cut);
        }
        if (!Name.empty()) {
            fNameIndexTable[Name] = Index + 1;
            fOptionNameList.push_back(Name);
        }
        if (End == string::npos) {
            break;
        }
        Names = Names.substr(End + 1);
    }

    return fOptionList.back();
}

KArgumentSchema& KArgumentSchema::AllowExcess()
{
    fIsExcessAllowed = true;
    return *this;
}

KArgumentSchema& KArgumentSchema::AllowUnknown()
{
    fIsUnknownAllowed = true;
    return *this;
}

void KArgumentSchema::Print(std::ostream& os)
{
    os << "Parameters:" << endl;
    for (auto& i : fParameterList) {
        os << "  ";
        i.Print(os, fNameLength + 3);
    }

    os << "Options:" << endl;
    for (auto& i : fOptionList) {
        os << "  ";
        i.Print(os, fNameLength + 3);
    }
}

void KArgumentSchema::Validate(KArgumentList& ArgumentList)
{
    // verification: parameter //
    if (ArgumentList.Length() < fNumberOfRequiredParameters) {
        throw KException() << "too few parameters";
    }
    for (unsigned i = 0; i < ArgumentList.Length(); i++) {
        unsigned Index = i;
        if (i >= fParameterList.size()) {
            if (fTakesMultipleParameters) {
                Index = fParameterList.size() - 1;
            }
            else if (fIsExcessAllowed) {
                continue;
            }
            else {
                throw KException() << "too many parameters";
            }
        }
        fParameterList[Index].Validate(ArgumentList[i]);
    }

    // verification: option //
    for (unsigned i = 0; i < ArgumentList.OptionList().size(); i++) {
        string Name = ArgumentList.OptionList()[i];
        unsigned int Index = fNameIndexTable[Name];
        if (Index == 0) {
            if (fIsUnknownAllowed) {
                continue;
            }
            throw KException() << "undefined option: " << Name;
        }

        fOptionList[Index - 1].Validate(ArgumentList[Name], Name);
    }

    // default value filling: parameter //
    unsigned NumberOfParameters = fParameterList.size();
    if (fTakesMultipleParameters) {
        NumberOfParameters -= 1;
    }
    for (unsigned i = ArgumentList.Length(); i < NumberOfParameters; i++) {
        ArgumentList.SetParameter(i, fParameterList[i].DefaultValue());
    }

    // default value filling: option //
    // note one option can have multiple names //
    vector<KVariant> OptionValueList;
    for (auto& i : fOptionList) {
        // fill all elements with default value //
        OptionValueList.push_back(i.DefaultValue());
    }
    for (auto Name : fOptionNameList) {
        // overwrite with specified values //
        if (ArgumentList[Name].IsVoid()) {
            continue;
        }
        unsigned int Index = fNameIndexTable[Name] - 1;
        if (!OptionValueList[Index].IsVoid()) {
            OptionValueList[Index].Assign(ArgumentList[Name]);
        }
    }
    for (auto Name : fOptionNameList) {
        // fill the argument lists //
        unsigned int Index = fNameIndexTable[Name] - 1;
        if (!OptionValueList[Index].IsVoid()) {
            ArgumentList.SetOption(Name, OptionValueList[Index]);
        }
    }
}
