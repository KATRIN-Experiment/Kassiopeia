// KArgumentList.h //
// Author: Sanshiro Enomoto <sanshiro@uw.edu> //


#ifndef KArgumentList_h__
#define KArgumentList_h__

#include <iostream>
#include <string>
#include <deque>
#include <map>
#include <vector>
#include <KVariant.h>

namespace katrin {

/**
 * \brief Program argument list, Tabree-style interface (associative array of variants)
 */
class KArgumentList {
  public:
    KArgumentList()
    { }
    KArgumentList(int argc, char** argv);
    virtual ~KArgumentList();
    inline KVariant operator[](unsigned int Index) const;
    inline KVariant operator[](const std::string& Name) const;
    inline size_t Length(void) const;
    inline KVariant Pop(void);
    inline std::string CommandLine(void) const;
    inline std::string ProgramName(void) const;
    inline std::string ProgramPathName(void) const;
    inline const std::deque<std::string>& ParameterList(void) const;
    inline const std::deque<std::string>& OptionList(void) const;
    inline const std::map<std::string, std::string>& OptionTable(void) const;
    virtual void Dump(std::ostream& os = std::cout) const;
  public:
    virtual KVariant GetParameter(unsigned int Index) const;
    virtual KVariant GetOption(const std::string& Name) const;
    virtual void SetParameter(unsigned int Index, const std::string& Value);
    virtual void SetOption(const std::string& Name, const std::string& Value);
    virtual void PullBack(int& argc, char**& argv) const;
  private:
    std::string fCommandLine, fProgramPathName, fProgramName;
    std::deque<std::string> fParameterList;
    std::deque<std::string> fOptionNameList;
    std::map<std::string, std::string> fOptionTable;
  private:
    mutable unsigned fArgvBufferSize;
    mutable char** fArgvBuffer;
};



/**
 * \brief Program argument list definition and validation
 */
class KArgumentSchema {
  public:
    class KElement {
      public:
        KElement(std::string Name);
        virtual ~KElement();
        virtual KElement& InTypeOf(const KVariant& Prototype);
        virtual KElement& WithDefault(const KVariant& Prototype);
        virtual KElement& WhichIs(const std::string& Description);
        virtual void Print(std::ostream& os, size_t NameWidth);
        virtual void Validate(const std::string& Value, std::string Name = "") throw(KException);
        virtual KVariant DefaultValue(void) const;
      protected:
        std::string fName;
        std::string fDescription;
        KVariant fPrototype;
        bool fIsDefaultValueEnabled;
    };
  public:
    KArgumentSchema(void);
    virtual ~KArgumentSchema();
    virtual KElement& Require(std::string Names);
    virtual KElement& Take(std::string Names);
    virtual KElement& TakeMultiple(std::string Names);
    virtual KArgumentSchema& AllowExcess(void);
    virtual KArgumentSchema& AllowUnknown(void);
  public:
    virtual void Print(std::ostream& os = std::cout);
    virtual void Validate(KArgumentList& ArgumentList) throw(KException);
  protected:
    virtual KElement& AddParameter(std::string Names);
    virtual KElement& AddOption(std::string Names);
  protected:
    bool fIsExcessAllowed, fIsUnknownAllowed, fTakesMultipleParameters;
    unsigned int fNumberOfRequiredParameters;
    std::vector<KElement> fParameterList;
    std::vector<KElement> fOptionList;
    std::vector<std::string> fParameterNameList;
    std::vector<std::string> fOptionNameList;
    std::map<std::string, unsigned int> fNameIndexTable;
  private:
    unsigned int fNameLength;
};



inline KVariant KArgumentList::operator[](unsigned int Index) const
{
    return this->GetParameter(Index);
}

inline KVariant KArgumentList::operator[](const std::string& Name) const
{
    return this->GetOption(Name);
}

inline size_t KArgumentList::Length(void) const
{
    return fParameterList.size();
}

inline std::string KArgumentList::ProgramName(void) const
{
    return fProgramName;
}

inline std::string KArgumentList::ProgramPathName(void) const
{
    return fProgramPathName;
}

inline std::string KArgumentList::CommandLine(void) const
{
    return fCommandLine;
}

inline const std::deque<std::string>& KArgumentList::ParameterList(void) const
{
    return fParameterList;
}

inline const std::deque<std::string>& KArgumentList::OptionList(void) const
{
    return fOptionNameList;
}

inline const std::map<std::string, std::string>& KArgumentList::OptionTable(void) const
{
    return fOptionTable;
}

inline KVariant KArgumentList::Pop(void)
{
    if (fParameterList.empty()) {
        return KVariant();
    }

    KVariant Parameter = fParameterList.front();
    fParameterList.pop_front();

    return Parameter;
}



}
#endif
