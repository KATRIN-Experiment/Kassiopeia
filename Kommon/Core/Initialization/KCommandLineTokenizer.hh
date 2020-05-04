#ifndef Kommon_KCommandLineTokenizer_hh_
#define Kommon_KCommandLineTokenizer_hh_

#include <map>
#include <string>
#include <vector>

namespace katrin
{

class KCommandLineTokenizer
{
  public:
    KCommandLineTokenizer();
    virtual ~KCommandLineTokenizer();

    //**********
    //processing
    //**********

  public:
    void ProcessCommandLine(int anArgc = 0, char** anArgv = nullptr);
    void ProcessCommandLine(std::vector<std::string> anArgList);

    const std::vector<std::string>& GetFiles();
    const std::map<std::string, std::string>& GetVariables();

  protected:
    void ReadEnvironmentVars();

  private:
    std::vector<std::string> fFiles;
    std::map<std::string, std::string> fVariables;
};

inline const std::vector<std::string>& KCommandLineTokenizer::GetFiles()
{
    return fFiles;
}
inline const std::map<std::string, std::string>& KCommandLineTokenizer::GetVariables()
{
    return fVariables;
}

}  // namespace katrin

#endif
