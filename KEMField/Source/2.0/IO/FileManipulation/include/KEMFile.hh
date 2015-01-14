#ifndef KEMFILE_DEF
#define KEMFILE_DEF

#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstdarg>
#include <algorithm>

#include "KEMCout.hh"

#include "KMD5HashGenerator.hh"
#include "KBinaryDataStreamer.hh"
#include "KStreamedSizeOf.hh"

using std::string;
using std::vector;

namespace KEMField
{

  /**
   * @class KEMFile
   *
   * @brief A class for reading and writing KEMField files.
   *
   * KEMFile is a class for reading and writing files, allowing for random
   * access using a key system.
   *
   * @author T.J. Corona
   */

  class KEMFile
  {
  public:
    KEMFile();
    KEMFile(string fileName);
    virtual ~KEMFile();

    void ActiveFile(string fileName) { fFileName = fileName; }
    std::string GetActiveFileName() const { return fFileName; }

    template <class Writable>
    void Write(string,const Writable&,string,vector<string>&);

    template <class Writable>
    void Write(string,const Writable&,string);

    template <class Writable>
    void Write(string,const Writable&,string,string);

    template <class Writable>
    void Write(const Writable& w,string name) { Write(fFileName,w,name); }

    template <class Writable>
    void Write(const Writable& w,string name,string label) { Write(fFileName,w,name,label); }

    template <class Writable>
    void Write(const Writable& w,string name,vector<string> labels) { Write(fFileName,w,name,labels); }

    template <class Readable>
    void Read(string,Readable&,string);

    template <class Readable>
    void Read(Readable& r,string s) { Read(fFileName,r,s); }

    template <class Readable>
    void ReadHashed(string,Readable&,string);

    template <class Readable>
    void ReadLabeled(string,Readable&,string,unsigned int index = 0);

    template <class Readable>
    void ReadLabeled(string,Readable&,vector<string>,unsigned int index = 0);

    template <class Writable>
    void Overwrite(string,const Writable&,string);

    template <class Writable>
    void Overwrite(const Writable& w,string s) { Overwrite(fFileName,w,s);}

    void Inspect(string) const;

    bool HasElement(string,string) const;
    bool HasLabeled(string,vector<string>) const;
    unsigned int NumberOfLabeled(string,string) const;
    unsigned int NumberOfLabeled(string,vector<string>) const;

    vector<string> LabelsForElement(string,string) const;
    bool ElementHasLabel(string,string,string) const;

    static bool FileExists(string);

    string GetFileSuffix() const { return fStreamer.GetFileSuffix(); }

  protected:

    string fFileName;

    KMD5HashGenerator fHashGenerator;

    mutable KBinaryDataStreamer fStreamer;

    struct Key
    {
      Key() {}
      ~Key() {}

      static string Name() { return "Key"; }

      void clear() { fObjectName.clear(); fClassName.clear(); fObjectHash.clear(); fLabels.clear(); fObjectLocation = fObjectSize = 0; }

      size_t NextKey() const { return fObjectLocation + fObjectSize; }

      template <typename Stream>
      friend Stream& operator>>(Stream& s,Key& k)
      {
	s.PreStreamInAction(k);
	s >> k.fObjectName;
	s >> k.fClassName;
	s >> k.fObjectHash;
	unsigned int size;
	s >> size;
	k.fLabels.resize(size);
	for (unsigned int i=0;i<size;i++)
	  s >> k.fLabels[i];
	s >> k.fObjectLocation;
	s >> k.fObjectSize;
	s.PostStreamInAction(k);
	return s;
      }

      template <typename Stream>
      friend Stream& operator<<(Stream& s,const Key& k)
      {
	s.PreStreamOutAction(k);
	s << k.fObjectName;
	s << k.fClassName;
	s << k.fObjectHash;
	s << (unsigned int)(k.fLabels.size());
	for (unsigned int i=0;i<k.fLabels.size();i++)
	  s << k.fLabels.at(i);
	s << k.fObjectLocation;
	s << k.fObjectSize;
	s.PostStreamOutAction(k);
	return s;
      }

      string fObjectName;
      string fClassName;
      string fObjectHash;
      vector<string> fLabels;
      size_t fObjectLocation;
      size_t fObjectSize;
    };

    Key KeyForElement(string,string);
    Key KeyForHashed(string,string);
    Key KeyForLabeled(string,string,unsigned int index=0);

  };

  template <class Writable>
  void KEMFile::Write(string fileName,const Writable& writable,string name)
  {
    vector<string> labels(0);
    Write<Writable>(fileName,writable,name,labels);
  }

  template <class Writable>
  void KEMFile::Write(string fileName,const Writable& writable,string name,string label)
  {
    vector<string> labels;
    labels.push_back(label);
    Write<Writable>(fileName,writable,name,labels);
  }

  template <class Writable>
  void KEMFile::Write(string fileName,const Writable& writable,string name, vector<string>& labels)
  {
    if (!FileExists(fileName))
    {
      fStreamer.open(fileName,"overwrite");
      fStreamer.close();
    }
    else
    {
      fStreamer.open(fileName,"read");

      // Pull and check the keys sequentially
      Key key;

      size_t readPoint = 0;
      fStreamer.Stream().seekg(0, fStreamer.Stream().end);
      size_t end = fStreamer.Stream().tellg();
      fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

      bool duplicateName = false;

      while (readPoint < end)
      {
	fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
	fStreamer >> key;
	if (key.fObjectName == name)
	  duplicateName = true;
	readPoint = key.NextKey();
      }

      fStreamer.close();

      if (duplicateName)
      {
	KEMField::cout<<"Element <"<<name<<"> already exists in file "<<fileName<<"."<<KEMField::endl;
	return;
      }
    }

    fStreamer.open(fileName,"update");

    // First, create a key for our object
    Key key;
    key.fObjectName = name;
    key.fClassName = Writable::Name();

    key.fObjectHash = fHashGenerator.GenerateHash(writable);

    for (vector<string>::iterator it = labels.begin();it!=labels.end();++it)
      key.fLabels.push_back(*it);

    // Write the incomplete key into the stream
    fStreamer.Stream().seekp(0, fStreamer.Stream().end);
    size_t keyLocation = fStreamer.Stream().tellp();
    fStreamer << key;

    // Write the object into the stream
    size_t objectLocation = fStreamer.Stream().tellp();
    fStreamer << writable;
    size_t objectEnd = fStreamer.Stream().tellp();

    // Complete the key information, and overwrite the key
    key.fObjectLocation = objectLocation;
    key.fObjectSize = objectEnd - objectLocation;

    fStreamer.close();
    fStreamer.open(fileName,"modify");

    fStreamer.Stream().seekp(keyLocation, fStreamer.Stream().beg);
    fStreamer << key;

    fStreamer.close();
  }

  template <class Writable>
  void KEMFile::Overwrite(string fileName,const Writable& writable,string name)
  {
    if (!FileExists(fileName))
    {
      KEMField::cout<<"Cannot open file /'"<<fileName<<"/'."<<KEMField::endl;
      return;
    }

    // First, we find the key associated with our object
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool elementFound = false;

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      if (key.fObjectName == name)
      {
	if (key.fClassName != Writable::Name())
	{
	  KEMField::cout<<"Element <"<<name<<"> is stored as a "<<key.fClassName<<KEMField::endl;
	  break;
	}

	elementFound = true;
	break;
      }
      else
	readPoint = key.NextKey();
    }

    fStreamer.close();

    if (!elementFound)
    {
      if (readPoint >= end)
	KEMField::cout<<"Element <"<<name<<"> could not be located in file \'"<<fileName<<"\'."<<KEMField::endl;
      return;
    }

    // check to make sure the objects are the same size
    KStreamedSizeOf sizeOf;
    if (sizeOf(writable) != key.fObjectSize)
    {
      KEMField::cout<<"Cannot overwrite element <"<<name<<"> with an instance of "<<key.fClassName<<" of a different size."<<KEMField::endl;
    }

    // overwrite the modified element
    fStreamer.open(fileName,"modify");

    fStreamer.Stream().seekp(key.fObjectLocation, fStreamer.Stream().beg);
    fStreamer << writable;

    fStreamer.close();
  }

  template <class Readable>
  void KEMFile::ReadHashed(string fileName,Readable& readable,string hash)
  {
    if (!FileExists(fileName))
    {
      KEMField::cout<<"Cannot open file /'"<<fileName<<"/'."<<KEMField::endl;
      return;
    }

    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);

      fStreamer >> key;

      if (key.fObjectHash == hash)
      {
	if (key.fClassName != Readable::Name())
	{
	  KEMField::cout<<"Element with hash <"<<hash<<"> is stored as a "<<key.fClassName<<KEMField::endl;
	}
	else
	{
	  fStreamer.Stream().seekg(key.fObjectLocation,
				   fStreamer.Stream().beg);
	  fStreamer >> readable;
	}
	break;
      }
      readPoint = key.NextKey();
    }
    fStreamer.close();
  }

  template <class Readable>
  void KEMFile::ReadLabeled(string fileName,Readable& readable,string label,unsigned int index)
  {
    if (!FileExists(fileName))
    {
      KEMField::cout<<"Cannot open file /'"<<fileName<<"/'."<<KEMField::endl;
      return;
    }

    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    unsigned int index_ = 0;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool found = false;

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);

      fStreamer >> key;
      for (vector<string>::iterator it = key.fLabels.begin();it!=key.fLabels.end();++it)
      {
	if (*it == label)
	{
	  // first, check if this is the correct index
	  if (index != index_)
	  {
	    index_++;
	    break;
	  }
	  else
	  {
	    if (key.fClassName != Readable::Name())

	      KEMField::cout<<"Label <"<<label<<"> is assigned to a "
		       <<key.fClassName<<KEMField::endl;
	    else
	    {
	      fStreamer.Stream().seekg(key.fObjectLocation,
				       fStreamer.Stream().beg);
	      fStreamer >> readable;
	    }
	      found = true;
	      break;
	  }
	}
      }
      if (found) break;
      readPoint = key.NextKey();
    }
    fStreamer.close();
  }

  template <class Readable>
  void KEMFile::ReadLabeled(string fileName,Readable& readable,vector<string> labels, unsigned int index)
  {
    if (!FileExists(fileName))
    {
      KEMField::cout<<"Cannot open file /'"<<fileName<<"/'."<<KEMField::endl;
      return;
    }

    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    unsigned int index_ = 0;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool found = false;

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      found = true;
      for (std::vector<std::string>::iterator it = labels.begin();it!=labels.end();++it)
      {
	std::vector<std::string>::iterator it2 = std::find(key.fLabels.begin(),key.fLabels.end(),*it);
	if (it2 == key.fLabels.end())
	{
	  found = false;
	  break;
	}
      }
      if (found)
      {
	// first, check if this is the correct index
	if (index != index_)
	{
	  index_++;
	}
	else
	{
	  if (key.fClassName != Readable::Name())
	  {
	    KEMField::cout<<"Label set is assigned to a "<<key.fClassName<<KEMField::endl;
	  }
	  else
	  {
	    fStreamer.Stream().seekg(key.fObjectLocation,
				     fStreamer.Stream().beg);
	    fStreamer >> readable;
	  }
	  break;
	}
      }
      readPoint = key.NextKey();
    }
    fStreamer.close();
  }

  template <class Readable>
  void KEMFile::Read(string fileName,Readable& readable,string name)
  {
    if (!FileExists(fileName))
    {
      KEMField::cout<<"Cannot open file /'"<<fileName<<"/'."<<KEMField::endl;
      return;
    }

    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);

      fStreamer >> key;
      if (key.fObjectName == name)
      {
	if (key.fClassName != Readable::Name())
	{
	  KEMField::cout<<"Element <"<<name<<"> is stored as a "<<key.fClassName<<KEMField::endl;
	  break;
	}

	fStreamer.Stream().seekg(key.fObjectLocation,fStreamer.Stream().beg);
	fStreamer >> readable;
	break;
      }
      else
	readPoint = key.NextKey();
    }

    fStreamer.close();
  }
}

#endif /* KEMFILE_DEF */
