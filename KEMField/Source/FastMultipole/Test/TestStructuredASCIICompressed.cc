#include "KSATestA.hh"
#include "KSATestB.hh"
#include "KSATestC.hh"
#include "KSATestD.hh"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

//#include "KSAObjectOutputNode.hh"
#include "KSAFileReader.hh"
#include "KSAFileWriter.hh"
#include "KSAFixedSizeInputOutputObject.hh"
#include "KSAInputCollector.hh"
#include "KSAOutputCollector.hh"
#include "KSAStructuredASCIIHeaders.hh"


#ifndef DEFAULT_DATA_DIR
#define DEFAULT_DATA_DIR "."
#endif /* !DEFAULT_DATA_DIR */

using namespace KEMField;

double UniformRandom(double lower_limit, double upper_limit)
{
    double r = 0;
    //we don't need high quality random numbers here, so we use rand()
    double m = RAND_MAX;
    m += 1;  // do not want the range to be inclusive of the upper limit
    double r1 = rand();
    r = r1 / m;
    return lower_limit + (upper_limit - lower_limit) * r;
}

int main()
{
    //the the specialization:

    std::cout << "is KSATestB derived from fixed size obj?  = " << std::endl;
    unsigned int test = KSAIsDerivedFrom<KSATestB, KSAFixedSizeInputOutputObject>::Is;
    std::cout << "test = " << test << std::endl;

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////


    std::stringstream s;
    s << DEFAULT_DATA_DIR << "/testKSA_orig.zksa";
    std::string outfile = s.str();
    s.clear();
    s.str("");
    s << DEFAULT_DATA_DIR << "/testKSA_copy.zksa";
    std::string outfile2 = s.str();

    KSAFileWriter writer;

    KSAOutputCollector collector;

    collector.SetUseTabbingTrue();

    writer.SetFileName(outfile);
    //    writer.SetModeRecreate();
    //    writer.IncludeXMLGuards();


    KSAOutputNode* root = new KSAOutputNode(std::string("root"));
    KSAObjectOutputNode<std::vector<KSATestC>>* test_c_vec_node =
        new KSAObjectOutputNode<std::vector<KSATestC>>("TestCVector");

    if (writer.Open()) {
        int n = 1;   //number of objects
        int nv = 1;  //number of doubles in vector

        std::vector<KSATestC>* C_vec = new std::vector<KSATestC>();

        KSATestC C_obj;
        KSATestB B_obj;
        KSATestD D_obj;
        double temp[3] = {1., 2., 3.};

        std::vector<KSATestB*> BVec;

        for (int i = 0; i < n; i++) {

            std::cout << "i = " << i << std::endl;
            C_obj.ClearData();
            C_obj.ClearBVector();

            for (int j = 0; j < nv; j++) {
                C_obj.AddData(UniformRandom(0, 1) * 1e-15);
            }


            B_obj.SetX(UniformRandom(0, 1));
            B_obj.SetY(UniformRandom(0, 1));
            D_obj.SetX(UniformRandom(0, 1));
            D_obj.SetY(UniformRandom(0, 1));
            D_obj.SetD(UniformRandom(0, 1));
            temp[0] = UniformRandom(0, 1);
            temp[1] = UniformRandom(0, 1);
            temp[2] = UniformRandom(0, 1);

            B_obj.SetArray(temp);
            D_obj.SetArray(temp);

            C_obj.SetB(B_obj);

            BVec.clear();
            B_obj.SetX(UniformRandom(0, 1));
            BVec.push_back(new KSATestB(B_obj));
            D_obj.SetD(UniformRandom(0, 1));
            BVec.push_back(new KSATestD(D_obj));

            C_obj.AddBVector(&BVec);

            BVec.clear();
            B_obj.SetY(UniformRandom(0, 1));
            BVec.push_back(new KSATestB(B_obj));
            B_obj.SetY(UniformRandom(0, 1));
            BVec.push_back(new KSATestD(D_obj));
            B_obj.SetY(UniformRandom(0, 1));
            BVec.push_back(new KSATestB(B_obj));

            C_obj.AddBVector(&BVec);

            C_obj.SetCData(UniformRandom(0, 1));

            C_vec->push_back(C_obj);
        }


        std::cout << "attaching object to node" << std::endl;
        test_c_vec_node->AttachObjectToNode(C_vec);

        root->AddChild(test_c_vec_node);

        std::cout << "setting file writer" << std::endl;
        collector.SetFileWriter(&writer);

        std::cout << "collecting output" << std::endl;
        collector.CollectOutput(root);

        std::cout << "done collecting output" << std::endl;

        writer.Close();

        delete C_vec;
    }
    else {
        std::cout << "Could not open file" << std::endl;
    }


    delete root;

    std::cout << "closing file" << std::endl;

    KSAFileReader reader;
    reader.SetFileName(outfile);

    KSAInputCollector* in_collector = new KSAInputCollector();
    in_collector->SetFileReader(&reader);


    KSAInputNode* input_root = new KSAInputNode(std::string("root"));
    KSAObjectInputNode<std::vector<KSATestC>>* input_c_vec =
        new KSAObjectInputNode<std::vector<KSATestC>>(std::string("TestCVector"));
    input_root->AddChild(input_c_vec);

    std::cout << "reading file" << std::endl;
    if (reader.Open()) {
        in_collector->ForwardInput(input_root);
    }
    else {
        std::cout << "Could not open file" << std::endl;
    }

    std::cout << "vector size = " << input_c_vec->GetObject()->size() << std::endl;

    KSAOutputCollector collector2;
    collector2.SetUseTabbingTrue();
    KSAFileWriter writer2;
    writer2.SetFileName(outfile2);
    //    writer2.SetModeRecreate();
    //    writer2.IncludeXMLGuards();

    KSAOutputNode* root2 = new KSAOutputNode(std::string("root"));
    KSAObjectOutputNode<std::vector<KSATestC>>* copy_c_vec =
        new KSAObjectOutputNode<std::vector<KSATestC>>("TestCVector");
    copy_c_vec->AttachObjectToNode(input_c_vec->GetObject());
    root2->AddChild(copy_c_vec);

    if (writer2.Open()) {
        collector2.SetFileWriter(&writer2);
        collector2.CollectOutput(root2);

        writer2.Close();
    }
    else {
        std::cout << "Could not open file" << std::endl;
    }
    std::cout << "done" << std::endl;

    delete root2;

    return 0;
}
