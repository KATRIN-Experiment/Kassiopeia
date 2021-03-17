#ifndef Kommon_KIncludeProcessor_hh_
#define Kommon_KIncludeProcessor_hh_

#include "KProcessor.hh"

#include <vector>

namespace katrin
{

class KIncludeProcessor : public KProcessor
{
  public:
    KIncludeProcessor();
    KIncludeProcessor& operator=(const KIncludeProcessor& other) = delete;
    ~KIncludeProcessor() override;

    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;

    void SetPath(const std::string& path);
    void AddDefaultPath(const std::string& path);

  private:
    void Reset();

    typedef enum  // NOLINT(modernize-use-using)
    {
        eElementInactive,
        eActive,
        eElementComplete
    } ElementState;
    ElementState fElementState;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eAttributeInactive,
        eName,
        ePath,
        eBase,
        eOptionalFlag,
        eAttributeComplete
    } AttributeState;
    AttributeState fAttributeState;

    bool fOptionalFlag;

    std::vector<std::string> fNames;
    std::vector<std::string> fPaths;
    std::vector<std::string> fBases;

    std::string fDefaultPath;
    std::vector<std::string> fAdditionalDefaultPaths;

    std::map<std::string, std::string> fIncludedPaths;

    using TreeNode = struct TreeNode_t
    {
        std::string fName;
        TreeNode_t* fParent;
        std::list<TreeNode_t*> fChildren;
    };
    TreeNode* fIncludeTree;

    static void PrintTreeNode(TreeNode* node, int level = 0, bool deleteAll = false);
};

}  // namespace katrin

#endif
