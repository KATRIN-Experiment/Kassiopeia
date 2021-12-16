#ifndef KFILE_H_
#define KFILE_H_

#include <string>
#include <vector>

#define STRING(anArgument)    #anArgument
#define AS_STRING(anArgument) STRING(anArgument)

#ifndef CONFIG_INSTALL_DIR
//static_assert( false, "CONFIG_INSTALL_DIR was not defined." );
#define CONFIG_DEFAULT_DIR "."
#else
#define CONFIG_DEFAULT_DIR AS_STRING(CONFIG_INSTALL_DIR)
#endif

#ifndef DATA_INSTALL_DIR
//static_assert( false, "DATA_INSTALL_DIR was not defined.");
#define DATA_DEFAULT_DIR "."
#else
#define DATA_DEFAULT_DIR AS_STRING(DATA_INSTALL_DIR)
#endif

#ifndef SCRATCH_INSTALL_DIR
//static_assert(false, "SCRATCH_INSTALL_DIR was not defined.");
#define SCRATCH_DEFAULT_DIR "."
#else
#define SCRATCH_DEFAULT_DIR AS_STRING(SCRATCH_INSTALL_DIR)
#endif

#ifndef OUTPUT_INSTALL_DIR
//static_assert(false, "OUTPUT_INSTALL_DIR was not defined.");
#define OUTPUT_DEFAULT_DIR "."
#else
#define OUTPUT_DEFAULT_DIR AS_STRING(OUTPUT_INSTALL_DIR)
#endif

#ifndef LOG_INSTALL_DIR
//static_assert(false, "LOG_INSTALL_DIR was not defined.");
#define LOG_DEFAULT_DIR "."
#else
#define LOG_DEFAULT_DIR AS_STRING(LOG_INSTALL_DIR)
#endif

namespace katrin
{

class KFile
{
  public:
    KFile();
    virtual ~KFile();

  public:
    void AddToPaths(const std::string& aPath);
    void SetDefaultPath(const std::string& aPath);
    void AddToBases(const std::string& aBase);
    void SetDefaultBase(const std::string& aBase);
    void AddToNames(const std::string& aName);
    const std::string& GetPath() const;
    const std::string& GetBase() const;
    const std::string& GetName() const;
    std::string GetAbsoluteName() const;
    const std::string& GetDefaultPath() const;
    const std::string& GetDefaultBase() const;
    bool IsUsingDefaultBase() const;
    bool IsUsingDefaultPath() const;

  protected:
    std::vector<std::string> fPaths;
    std::string fDefaultPath;
    std::vector<std::string> fBases;
    std::string fDefaultBase;
    std::vector<std::string> fNames;
    std::string fResolvedPath;
    std::string fResolvedBase;
    std::string fResolvedName;
    bool fUsingDefaultBase;
    bool fUsingDefaultPath;

  public:
    typedef enum  // NOLINT(modernize-use-using)
    {
        eRead,
        eWrite,
        eAppend
    } Mode;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eOpen,
        eClosed
    } State;

    bool Open(Mode aMode = eRead);
    bool IsOpen();

    bool Close();
    bool IsClosed();

    static bool Test(const std::string& aName);

  protected:
    virtual bool OpenFileSubclass(const std::string& aName, const Mode& aMode) = 0;
    virtual bool CloseFileSubclass() = 0;
    State fState;

  private:
    void SetResolvedAttributes(const std::string& resolvedName);

  protected:
    static const std::string fDirectoryMark;
    static const std::string fExtensionMark;
};

}  // namespace katrin

#endif
