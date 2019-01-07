// KArgumentList.cpp //
// Author: Sanshiro Enomoto <sanshiro@uw.edu> //


#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <deque>
#include <map>
#include <algorithm>
#include <cctype>
#include "KVariant.h"
#include "KArgumentList.h"


using namespace std;
using namespace katrin;


KArgumentList::KArgumentList(int argc, char** argv)
{
    if (argc > 0) {
        fProgramPathName = argv[0];
        fCommandLine = argv[0];
        string::size_type slash = fProgramPathName.find_last_of('/');
        if (slash != string::npos) {
            fProgramName = fProgramPathName.substr(slash+1, string::npos);
        }
        else {
            fProgramName = fProgramPathName;
        }
    }
    for (int i = 1; i < argc; i++) {
	string Argument = argv[i];
        fCommandLine += " " + Argument;
	if ((Argument[0] != '-') || (Argument == "-") || (Argument == "--")) {
	    fParameterList.push_back(Argument);
	}
	else if ((Argument.size() > 1) && (isdigit(Argument[1]))) {
	    // negative number parameter //
	    fParameterList.push_back(Argument);
	}
	else {
	    string::size_type NameLength = Argument.find_first_of('=');
	    string Name = Argument.substr(0, NameLength);
	    string Value = "";

	    if (NameLength != string::npos) {
		Value = Argument.substr(NameLength + 1, Argument.size());
	    }
	    
	    if (Value.empty() && (i+1 < argc)) {
		if (NameLength != string::npos) {
		    // already contains the assign operator// 
		    i++;
		    Value = argv[i];
		}
		else if (argv[i+1][0] == '=') {
		    i++;
		    if (argv[i][1] != '\0') {
			Value = (argv[i] + 1);
		    }
		    else if (i+1 < argc) {
			i++;
			Value = argv[i];
		    }
		}
	    }

	    fOptionTable[Name] = Value;
	    fOptionNameList.push_back(Name);
	}
    }

    fArgvBuffer = 0;
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
    map<string, string>::const_iterator Option = fOptionTable.find(Name);
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
        fParameterList.push_back("");
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
    for (unsigned i = 0; i < fParameterList.size(); i++) {
	os << "    " << fParameterList[i] << endl;
    }
    
    os << "Options:" << endl;
    for (unsigned i = 0; i < fOptionNameList.size(); i++) {
        string Name = fOptionNameList[i];
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
        fArgvBuffer[i+1] = strdup(fParameterList[i].c_str());
    }

    argc = fParameterList.size() + 1;
    argv = fArgvBuffer;
}

KArgumentSchema::KElement::KElement(std::string Name)
{
    fName = Name;
    fIsDefaultValueEnabled = false;
}

KArgumentSchema::KElement::~KElement()
{
}

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
        if (! Value.empty()) {
            throw KException() << "argument does not take value: " << Name;
        }
    }
    else {
        try {
            KVariant TestValue = fPrototype;
            TestValue.Assign(Value);
        }
        catch (KException &e) {
            throw KException() << "invalid argument value: " << Name << "=" << Value;
        }
    }
}

KVariant KArgumentSchema::KElement::DefaultValue(void) const
{
    if (fIsDefaultValueEnabled) {
        return fPrototype;
    }
    else {
        return KVariant();
    }
}



KArgumentSchema::KArgumentSchema(void)
{
    fNameLength = 0;

    fIsExcessAllowed = false;
    fIsUnknownAllowed = false;
    fTakesMultipleParameters = false;
    fNumberOfRequiredParameters = 0;
}

KArgumentSchema::~KArgumentSchema()
{
}

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
    fOptionList.push_back(KElement(Names));
    
    fNameLength = max(fNameLength, (unsigned int) Names.size());

    while (1) {
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
        if (! Name.empty()) {
            fNameIndexTable[Name] = Index+1;
            fOptionNameList.push_back(Name);
        }
        if (End == string::npos) {
            break;
        }
        Names = Names.substr(End+1);
    }

    return fOptionList.back();
}

KArgumentSchema& KArgumentSchema::AllowExcess(void)
{
    fIsExcessAllowed = true;
    return *this;
}

KArgumentSchema& KArgumentSchema::AllowUnknown(void)
{
    fIsUnknownAllowed = true;
    return *this;
}

void KArgumentSchema::Print(std::ostream& os)
{
    os << "Parameters:" << endl;
    for (unsigned i = 0; i < fParameterList.size(); i++) {
        os << "  ";
        fParameterList[i].Print(os, fNameLength + 3);
    }

    os << "Options:" << endl;
    for (unsigned i = 0; i < fOptionList.size(); i++) {
        os << "  ";
        fOptionList[i].Print(os, fNameLength + 3);
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

        fOptionList[Index-1].Validate(ArgumentList[Name], Name);
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
    for (unsigned i = 0; i < fOptionList.size(); i++) {
        // fill all elements with default value //
        OptionValueList.push_back(fOptionList[i].DefaultValue());
    }
    for (unsigned i = 0; i < fOptionNameList.size(); i++) {
        // overwrite with specified values //
        string Name = fOptionNameList[i];
        if (ArgumentList[Name].IsVoid()) {
            continue;
        }
        unsigned int Index = fNameIndexTable[Name] - 1;
        if (! OptionValueList[Index].IsVoid()) {
            OptionValueList[Index].Assign(ArgumentList[Name]);
        }
    }
    for (unsigned i = 0; i < fOptionNameList.size(); i++) {
        // fill the argument lists //
        string Name = fOptionNameList[i];
        unsigned int Index = fNameIndexTable[Name] - 1;
        if (! OptionValueList[Index].IsVoid()) {
            ArgumentList.SetOption(Name, OptionValueList[Index]);
        }
    }
}
