#ifndef KEMFILEINTERFACE_DEF
#define KEMFILEINTERFACE_DEF

#include <set>

#include "KEMFile.hh"

#include "KSAInputNode.hh"
#include "KSAOutputNode.hh"

using std::string;
using std::vector;
using std::set;

namespace KEMField
{

  /**
   * @class KEMFileInterface
   *
   * @brief A class for reading and writing KEMField files.
   *
   * KEMFileInterface is a class for reading and writing KEMField files.
   *
   * @author T.J. Corona
   */

  class KEMFileInterface : public KEMFile
  {
  public:
    static KEMFileInterface* GetInstance();

    template <class Readable>
    void FindByName(Readable& readable,string name, bool& result=KEMFileInterface::fNullResult);

    template <class Readable>
    void FindByHash(Readable& readable,string hash, bool& result=KEMFileInterface::fNullResult);

    template <class Readable>
    void FindByLabel(Readable& readable,string label,unsigned int index=0, bool& result=KEMFileInterface::fNullResult);

    template <class Readable>
    void FindByLabels(Readable& readable,vector<string> labels,unsigned int index=0,bool& result=KEMFileInterface::fNullResult);

    void ReadKSAFile(KSAInputNode* node, string file_name, bool& result=KEMFileInterface::fNullResult);
    void SaveKSAFile(KSAOutputNode* node, string file_name, bool& result=KEMFileInterface::fNullResult, bool forceOverwrite = false);

    unsigned int NumberWithLabel(string label) const;
    unsigned int NumberWithLabels(vector<string> labels) const;

    set<string> FileList(string directory="") const;
    set<string> CompleteFileList(string directory="") const;

    bool DirectoryExists(string directory);
    bool CreateDirectory(string directory);
    bool RemoveDirectory(string directory);

    void ActiveDirectory(string directory);
    string ActiveDirectory() const { return fActiveDirectory; }

  private:
    KEMFileInterface();
    virtual ~KEMFileInterface() {}

    static KEMFileInterface* fEMFileInterface;

    string fActiveDirectory;

    static bool fNullResult;
  };

  template <class Readable>
  void KEMFileInterface::FindByName(Readable& readable,string name,bool& result)
  {
    result = true;
    set<string> fileList = FileList();

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
    {
      if (HasElement(*it,name))
	return Read(*it,readable,name);
    }
    result = false;
  }

  template <class Readable>
  void KEMFileInterface::FindByHash(Readable& readable,string hash,bool& result)
  {
    result = true;
    set<string> fileList = FileList();

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
    {
      if (HasElement(*it,hash))
	return ReadHashed(*it,readable,hash);
    }
    result = false;
  }

  template <class Readable>
  void KEMFileInterface::FindByLabel(Readable& readable,string label,unsigned int index,bool& result)
  {
    result = true;
    set<string> fileList = FileList();

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
    {
      unsigned int nLabeled = NumberOfLabeled(*it,label);
      if (index < nLabeled)
      {
	return ReadLabeled(*it,readable,label,index);
      }
      else
	index -= nLabeled;
    }
    result = false;
  }

  template <class Readable>
  void KEMFileInterface::FindByLabels(Readable& readable,vector<string> labels,unsigned int index,bool& result)
  {
    result = true;
    set<string> fileList = FileList();

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
    {
      unsigned int nLabeled = NumberOfLabeled(*it,labels);
      if (index < nLabeled)
      {
	return ReadLabeled(*it,readable,labels,index);
      }
      else
	index -= nLabeled;
    }
    result = false;
  }
}

#endif /* KEMFILEINTERFACE_DEF */
