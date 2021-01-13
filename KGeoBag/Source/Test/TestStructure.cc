#include "KGCore.hh"
#include "KGCoreMessage.hh"

//#include "KGCylinder.hh"

namespace KGeoBag
{

class TestSurface
{
  public:
    TestSurface() : fArea(0.) {}
    virtual ~TestSurface() = default;

    double fArea;
};

class TestSpace
{
  public:
    TestSpace() : fVolume(0.) {}
    virtual ~TestSpace() = default;

    double fVolume;
};

class Test
{
  public:
    typedef TestSurface Surface;
    using Space = TestSpace;
};

using KGTestSpace = KGExtendedSpace<Test>;
using KGTestSurface = KGExtendedSurface<Test>;

}  // namespace KGeoBag

using namespace KGeoBag;

int main(int anArgc, char** anArgv)
{

    if (anArgc < 2) {
        coremsg(eWarning) << "usage:" << ret;
        coremsg(eWarning) << "  TestStructure <path>" << eom;
        return -1;
    }

    coremsg(eNormal) << "building geometry..." << eom;

    //    KGCylinder* tCylinder = new KGCylinder();
    //    tCylinder->SetR( 0.5 );
    //    tCylinder->SetZ1( -0.1 );
    //    tCylinder->SetZ2( 0.25 );

    KGVolume* tCylinder = nullptr;

    auto* tRoot = new KGSpace(tCylinder);

    tRoot->AsExtension<Test>()->fVolume = 3.2;
    tRoot->SetName("root");
    tRoot->AddTag("all");

    auto* tLeft = new KGSpace(tCylinder);
    tLeft->AsExtension<Test>()->fVolume = 3.2;
    tLeft->SetName("left");
    tLeft->AddTag("all");
    tLeft->AddTag("active");
    tRoot->AddChildSpace(tLeft);

    auto* tLeftOne = new KGSpace(tCylinder);
    tLeftOne->SetName("left_one");
    tLeftOne->AddTag("all");
    tLeftOne->AddTag("active");
    tLeft->AddChildSpace(tLeftOne);

    auto* tLeftTwo = new KGSpace(tCylinder);
    tLeftTwo->SetName("left_two");
    tLeftTwo->AddTag("all");
    tLeftTwo->AddTag("active");
    tLeft->AddChildSpace(tLeftTwo);

    auto* tRight = new KGSpace(tCylinder);
    tRight->SetName("right");
    tRight->AddTag("all");
    tRight->AddTag("inactive");
    tRoot->AddChildSpace(tRight);

    auto* tRightOne = new KGSpace(tCylinder);
    tRightOne->AsExtension<Test>()->fVolume = 3.2;
    tRightOne->SetName("right_one");
    tRightOne->AddTag("all");
    tRightOne->AddTag("inactive");
    tRight->AddChildSpace(tRightOne);

    auto* tRightTwo = new KGSpace(tCylinder);
    tRightTwo->AsExtension<Test>()->fVolume = 3.2;
    tRightTwo->SetName("right_two");
    tRightTwo->AddTag("all");
    tRightTwo->AddTag("inactive");
    tRight->AddChildSpace(tRightTwo);

    coremsg(eNormal) << "...done" << eom;

    //    KGInterface* tInterface = KGInterface::GetInstance();
    //
    //    string tPath( anArgv[ 1 ] );
    //    vector< KGSpace* > tSet = tInterface->RetrieveSpaces( tPath );
    //    vector< KGTestSpace* > tTestSet = tInterface->As< Test >()->RetrieveSpaces( tPath );
    //
    //    coremsg << "retrieved spaces:" << ret;
    //    for( vector< KGSpace* >::iterator tSetIt = tSet.begin(); tSetIt != tSet.end(); tSetIt++ )
    //    {
    //        coremsg << "  <" << (*tSetIt)->GetName() << ">" << ret;
    //    }
    //    coremsg( eNormal ) << eom;
    //
    //    coremsg << "retrieved test spaces:" << ret;
    //    for( vector< KGTestSpace* >::iterator tTestSetIt = tTestSet.begin(); tTestSetIt != tTestSet.end(); tTestSetIt++ )
    //    {
    //        coremsg << "  <" << (*tTestSetIt)->GetName() << ">" << ret;
    //    }
    //    coremsg( eNormal ) << eom;

    return 0;
}
