#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <limits>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <vector>


#include "TestA.hh"
#include "TestB.hh"
#include "TestC.hh"
#include "TestD.hh"

//#include "KSAObjectOutputNode.hh"
#include "KSAFixedSizeInputOutputObject.hh"

#include "KSAStructuredASCIIHeaders.hh"

#include "KSAFileWriter.hh"
#include "KSAFileReader.hh"

#include "KSAOutputCollector.hh"
#include "KSAInputCollector.hh"

#include <gsl/gsl_rng.h>


#ifndef DEFAULT_DATA_DIR
#define DEFAULT_DATA_DIR "~/."
#endif /* !DEFAULT_DATA_DIR */

using namespace KEMField;

int main(int argc, char **argv)
{


    //the the specialization:

    std::cout<<"is TestB derived from fixed size obj?  = " << std::endl;
    unsigned int test = KSAIsDerivedFrom< TestB, KSAFixedSizeInputOutputObject >::Is;
    std::cout<<"test = "<<test<<std::endl;

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////


    std::stringstream s;
    s << DEFAULT_DATA_DIR << "/testKSA_orig.xml";
    std::string outfile = s.str();
    s.clear();s.str("");
    s << DEFAULT_DATA_DIR << "/testKSA_copy.xml";
    std::string outfile2 = s.str();

    KSAFileWriter writer;

    KSAOutputCollector collector;

    collector.SetUseTabbingTrue();

    writer.SetFileName(outfile);
//    writer.SetModeRecreate();
//    writer.IncludeXMLGuards();


    KSAOutputNode* root = new KSAOutputNode(std::string("root"));
    KSAObjectOutputNode< std::vector< TestC >  >* test_c_vec_node = new KSAObjectOutputNode< std::vector< TestC >  >("TestCVector");

    if( writer.Open() )
    {
	const gsl_rng_type * T;
	gsl_rng * r;
	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);


        int n = 10000; //number of objects
        int nv = 1000; //number of doubles in vector

        std::vector<TestC>* C_vec = new std::vector<TestC>();

        TestC C_obj;
        TestB B_obj;
        TestD D_obj;
        double temp[3] = {1.,2.,3.};

        std::vector<TestB*> BVec;

        for(int i=0; i<n; i++)
        {
            C_obj.ClearData();
            C_obj.ClearBVector();

            for(int j=0; j<nv; j++)
            {
                C_obj.AddData(gsl_rng_uniform(r)*1e-15);
            }


            B_obj.SetX(gsl_rng_uniform(r));
            B_obj.SetY(gsl_rng_uniform(r));
            D_obj.SetX(gsl_rng_uniform(r));
            D_obj.SetY(gsl_rng_uniform(r));
            D_obj.SetD(gsl_rng_uniform(r));
            temp[0] = gsl_rng_uniform(r);
            temp[1] = gsl_rng_uniform(r);
            temp[2] = gsl_rng_uniform(r);

            B_obj.SetArray(temp);
            D_obj.SetArray(temp);

            C_obj.SetB(B_obj);

            BVec.clear();
            B_obj.SetX(gsl_rng_uniform(r));
            BVec.push_back(new TestB(B_obj));
            D_obj.SetD(gsl_rng_uniform(r));
            BVec.push_back(new TestD(D_obj));

            C_obj.AddBVector(&BVec);

            BVec.clear();
            B_obj.SetY(gsl_rng_uniform(r));
            BVec.push_back(new TestB(B_obj));
            B_obj.SetY(gsl_rng_uniform(r));
            BVec.push_back(new TestD(D_obj));
            B_obj.SetY(gsl_rng_uniform(r));
            BVec.push_back(new TestB(B_obj));

            C_obj.AddBVector(&BVec);

            C_obj.SetCData(gsl_rng_uniform(r));

            C_vec->push_back(C_obj);
        }


        test_c_vec_node->AttachObjectToNode(C_vec);

        root->AddChild(test_c_vec_node);

        collector.SetFileWriter(&writer);
        collector.CollectOutput(root);

        writer.Close();

        delete C_vec;

    }
    else
    {
        std::cout<<"Could not open file"<<std::endl;
    }


    delete root;

    std::cout<<"closing file"<<std::endl;

    KSAFileReader reader;
    reader.SetFileName(outfile);

    KSAInputCollector* in_collector = new KSAInputCollector();
    in_collector->SetFileReader(&reader);


    KSAInputNode* input_root = new KSAInputNode(std::string("root"));
    KSAObjectInputNode< std::vector< TestC > >* input_c_vec = new KSAObjectInputNode< std::vector< TestC > >(std::string("TestCVector"));
    input_root->AddChild(input_c_vec);

    std::cout<<"reading file"<<std::endl;
    if( reader.Open() )
    {
        in_collector->ForwardInput(input_root);
    }
    else
    {
        std::cout<<"Could not open file"<<std::endl;
    }

    std::cout<<"vector size = "<<input_c_vec->GetObject()->size()<<std::endl;

    KSAOutputCollector collector2;
    collector2.SetUseTabbingTrue();
    KSAFileWriter writer2;
    writer2.SetFileName(outfile2);
//    writer2.SetModeRecreate();
//    writer2.IncludeXMLGuards();

    KSAOutputNode* root2 = new KSAOutputNode(std::string("root"));
    KSAObjectOutputNode< std::vector< TestC >  >* copy_c_vec = new KSAObjectOutputNode< std::vector< TestC >  >("TestCVector");
    copy_c_vec->AttachObjectToNode(input_c_vec->GetObject());
    root2->AddChild(copy_c_vec);

    if( writer2.Open() )
    {
        collector2.SetFileWriter(&writer2);
        collector2.CollectOutput(root2);

        writer2.Close();

    }
    else
    {
        std::cout<<"Could not open file"<<std::endl;
    }
    std::cout<<"done"<<std::endl;

    delete root2;

    return 0;


}
