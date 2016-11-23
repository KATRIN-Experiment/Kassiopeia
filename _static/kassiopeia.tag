<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile>
  <compound kind="file">
    <name>ElasticFerencCalculator.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Interactions/Include/</path>
    <filename>ElasticFerencCalculator_8h</filename>
    <class kind="class">Kassiopeia::ElasticFerencCalculator</class>
  </compound>
  <compound kind="file">
    <name>InelasticFerencCalculator.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Interactions/Include/</path>
    <filename>InelasticFerencCalculator_8h</filename>
    <class kind="class">Kassiopeia::InelasticFerencCalculator</class>
  </compound>
  <compound kind="file">
    <name>KSCyclicIterator.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Utility/Include/</path>
    <filename>KSCyclicIterator_8h</filename>
    <class kind="class">Kassiopeia::KSCyclicIterator</class>
  </compound>
  <compound kind="file">
    <name>KSGenConversion.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Generators/Include/</path>
    <filename>KSGenConversion_8h</filename>
    <class kind="class">Kassiopeia::KSGenConversion</class>
  </compound>
  <compound kind="file">
    <name>KSGenRelaxation.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Generators/Include/</path>
    <filename>KSGenRelaxation_8h</filename>
    <class kind="class">Kassiopeia::KSGenRelaxation</class>
    <class kind="struct">Kassiopeia::KSGenRelaxation::line_struct</class>
  </compound>
  <compound kind="file">
    <name>KSGenShakeOff.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Generators/Include/</path>
    <filename>KSGenShakeOff_8h</filename>
    <class kind="class">Kassiopeia::KSGenShakeOff</class>
  </compound>
  <compound kind="file">
    <name>KSIntCalculatorHydrogen.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Interactions/Include/</path>
    <filename>KSIntCalculatorHydrogen_8h</filename>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenDissoziation10</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenDissoziation15</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenElastic</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenElasticBase</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenExcitationB</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenExcitationBase</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenExcitationC</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenExcitationElectronic</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenIonisation</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenRot02</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenRot13</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenRot20</class>
    <class kind="class">Kassiopeia::KSIntCalculatorHydrogenVib</class>
  </compound>
  <compound kind="file">
    <name>KSIntCalculatorTritium.cxx</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Interactions/Source/</path>
    <filename>KSIntCalculatorTritium_8cxx</filename>
    <includes id="KSIntCalculatorTritium_8h" name="KSIntCalculatorTritium.h" local="yes" imported="no">KSIntCalculatorTritium.h</includes>
  </compound>
  <compound kind="file">
    <name>KSIntCalculatorTritium.h</name>
    <path>/home/john/work/projects/kasper/Kassiopeia/Interactions/Include/</path>
    <filename>KSIntCalculatorTritium_8h</filename>
    <includes id="KSIntCalculatorHydrogen_8h" name="KSIntCalculatorHydrogen.h" local="yes" imported="no">KSIntCalculatorHydrogen.h</includes>
    <class kind="class">Kassiopeia::KSIntCalculatorTritiumElastic</class>
    <class kind="class">Kassiopeia::KSIntCalculatorTritiumElasticBase</class>
    <class kind="class">Kassiopeia::KSIntCalculatorTritiumRot02</class>
    <class kind="class">Kassiopeia::KSIntCalculatorTritiumRot13</class>
    <class kind="class">Kassiopeia::KSIntCalculatorTritiumRot20</class>
    <class kind="class">Kassiopeia::KSIntCalculatorTritiumVib</class>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::ElasticFerencCalculator</name>
    <filename>classKassiopeia_1_1ElasticFerencCalculator.html</filename>
    <member kind="function">
      <type></type>
      <name>ElasticFerencCalculator</name>
      <anchorfile>classKassiopeia_1_1ElasticFerencCalculator.html</anchorfile>
      <anchor>a5d9136ae33e27078c06faf7eeadf1d7c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~ElasticFerencCalculator</name>
      <anchorfile>classKassiopeia_1_1ElasticFerencCalculator.html</anchorfile>
      <anchor>aa9e6cc7b22831973af40e8fe28bc2c5a</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>randomel</name>
      <anchorfile>classKassiopeia_1_1ElasticFerencCalculator.html</anchorfile>
      <anchor>ac18d8d13a89fd126003cb2d5a8d6b60f</anchor>
      <arglist>(double anE, double &amp;Eloss, double &amp;Theta)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>sigmaeltot</name>
      <anchorfile>classKassiopeia_1_1ElasticFerencCalculator.html</anchorfile>
      <anchor>adf79a3a8afd498d0636fbd1c87e92150</anchor>
      <arglist>(double anE)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>DiffXSecEl</name>
      <anchorfile>classKassiopeia_1_1ElasticFerencCalculator.html</anchorfile>
      <anchor>ad05e81dec7748dd1e21b481b1e02cf52</anchor>
      <arglist>(double anE, double cosTheta)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::FBBRionization</name>
    <filename>classKassiopeia_1_1FBBRionization.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::InelasticFerencCalculator</name>
    <filename>classKassiopeia_1_1InelasticFerencCalculator.html</filename>
    <member kind="function" virtualness="virtual">
      <type>virtual</type>
      <name>~InelasticFerencCalculator</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a0b08f733d251cdda3331c88b9b52b567</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual double</type>
      <name>GetIonizationEnergy</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>aa4680ea4337968442d9abee4dbb8c12e</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>randomexc</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>ac6b5d0f1bc1f163c4bd16900ce5e7220</anchor>
      <arglist>(double anE, double &amp;Eloss, double &amp;Theta)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>randomion</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a6001213bf0cef5c3915ee4eb376d872c</anchor>
      <arglist>(double anE, double &amp;Eloss, double &amp;Theta)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>setmolecule</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a54d63424b47750929dba6d9819d0b53b</anchor>
      <arglist>(const std::string &amp;aMolecule)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual double</type>
      <name>sigmaexc</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a134bb3e664ebd9635652559c8af0c4a8</anchor>
      <arglist>(double anE)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual double</type>
      <name>sigmaion</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>add4550913d35c35253b846056bf911a0</anchor>
      <arglist>(double anE)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>DiffXSecExc</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a3441837cd9691ce0d4d814aa4408bcf0</anchor>
      <arglist>(double anE, double cosTheta)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>DiffXSecInel</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a29396c5c93f265569754ee11fdf98c43</anchor>
      <arglist>(double anE, double cosTheta)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>void</type>
      <name>gensecelen</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a4f4452a317a4503a1310bf414b6a5732</anchor>
      <arglist>(double E, double &amp;W)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>sigmaBC</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>aeadba7a05256f710bbfed2a70029c761</anchor>
      <arglist>(double anE)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>sigmadiss10</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>acea7238bb5f8160e9d0d3bcfc798823a</anchor>
      <arglist>(double anE)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>sigmadiss15</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>a7b810bef6e6c63efbc961f77853fb59b</anchor>
      <arglist>(double anE)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>sigmainel</name>
      <anchorfile>classKassiopeia_1_1InelasticFerencCalculator.html</anchorfile>
      <anchor>ac5c0d0b6c2e3717ae5f12a1f7845ac4b</anchor>
      <arglist>(double anE)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSElasticElsepa</name>
    <filename>classKassiopeia_1_1KESSElasticElsepa.html</filename>
    <base>KSComponentTemplate&lt; KESSElasticElsepa, KSIntCalculator &gt;</base>
    <base>Kassiopeia::KESSScatteringCalculator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSInelasticBetheFano</name>
    <filename>classKassiopeia_1_1KESSInelasticBetheFano.html</filename>
    <base>KSComponentTemplate&lt; KESSInelasticBetheFano, KSIntCalculator &gt;</base>
    <base>Kassiopeia::KESSScatteringCalculator</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>ExecuteInteraction</name>
      <anchorfile>classKassiopeia_1_1KESSInelasticBetheFano.html</anchorfile>
      <anchor>a1086a6a03776a027502708ddad3cdbca</anchor>
      <arglist>(const KSParticle &amp;anInitialParticle, KSParticle &amp;aFinalParticle, KSParticleQueue &amp;aQueue)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSInelasticPenn</name>
    <filename>classKassiopeia_1_1KESSInelasticPenn.html</filename>
    <base>KSComponentTemplate&lt; KESSInelasticPenn, KSIntCalculator &gt;</base>
    <base>Kassiopeia::KESSScatteringCalculator</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>ExecuteInteraction</name>
      <anchorfile>classKassiopeia_1_1KESSInelasticPenn.html</anchorfile>
      <anchor>a1c98756a3e58ace04fc419e9e08e8708</anchor>
      <arglist>(const KSParticle &amp;anInitialParticle, KSParticle &amp;aFinalParticle, KSParticleQueue &amp;aQueue)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSPhotoAbsorbtion</name>
    <filename>classKassiopeia_1_1KESSPhotoAbsorbtion.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSRelaxation</name>
    <filename>classKassiopeia_1_1KESSRelaxation.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSScatteringCalculator</name>
    <filename>classKassiopeia_1_1KESSScatteringCalculator.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KESSSurfaceInteraction</name>
    <filename>classKassiopeia_1_1KESSSurfaceInteraction.html</filename>
    <base>KSComponentTemplate&lt; KESSSurfaceInteraction, KSSurfaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>KGMeshElementCollector</name>
    <filename>classKGMeshElementCollector.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSAddExpression</name>
    <filename>classKassiopeia_1_1KSAddExpression.html</filename>
    <templarg>XLeft</templarg>
    <templarg>XRight</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSArray</name>
    <filename>classKassiopeia_1_1KSArray.html</filename>
    <templarg>XDimension</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommand</name>
    <filename>classKassiopeia_1_1KSCommand.html</filename>
    <base>Kassiopeia::KSObject</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandFactory</name>
    <filename>classKassiopeia_1_1KSCommandFactory.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandGroup</name>
    <filename>classKassiopeia_1_1KSCommandGroup.html</filename>
    <base>Kassiopeia::KSCommand</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandMemberAdd</name>
    <filename>classKassiopeia_1_1KSCommandMemberAdd.html</filename>
    <templarg>XParentType</templarg>
    <templarg>XChildType</templarg>
    <base>Kassiopeia::KSCommand</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandMemberAddFactory</name>
    <filename>classKassiopeia_1_1KSCommandMemberAddFactory.html</filename>
    <templarg>XParentType</templarg>
    <templarg>XChildType</templarg>
    <base>Kassiopeia::KSCommandFactory</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSCommandMemberData</name>
    <filename>classkatrin_1_1KSCommandMemberData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandMemberParameter</name>
    <filename>classKassiopeia_1_1KSCommandMemberParameter.html</filename>
    <templarg>XParentType</templarg>
    <templarg>XChildType</templarg>
    <base>Kassiopeia::KSCommand</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandMemberParameterFactory</name>
    <filename>classKassiopeia_1_1KSCommandMemberParameterFactory.html</filename>
    <templarg>XParentType</templarg>
    <templarg>XChildType</templarg>
    <base>Kassiopeia::KSCommandFactory</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandMemberRemove</name>
    <filename>classKassiopeia_1_1KSCommandMemberRemove.html</filename>
    <templarg>XParentType</templarg>
    <templarg>XChildType</templarg>
    <base>Kassiopeia::KSCommand</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCommandMemberRemoveFactory</name>
    <filename>classKassiopeia_1_1KSCommandMemberRemoveFactory.html</filename>
    <templarg>XParentType</templarg>
    <templarg>XChildType</templarg>
    <base>Kassiopeia::KSCommandFactory</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSCommandMemberSimpleData</name>
    <filename>classkatrin_1_1KSCommandMemberSimpleData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponent</name>
    <filename>classKassiopeia_1_1KSComponent.html</filename>
    <base>Kassiopeia::KSObject</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentDelta</name>
    <filename>classKassiopeia_1_1KSComponentDelta.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentDeltaData</name>
    <filename>classkatrin_1_1KSComponentDeltaData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentFactory</name>
    <filename>classKassiopeia_1_1KSComponentFactory.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentGroup</name>
    <filename>classKassiopeia_1_1KSComponentGroup.html</filename>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentIntegral</name>
    <filename>classKassiopeia_1_1KSComponentIntegral.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentIntegralData</name>
    <filename>classkatrin_1_1KSComponentIntegralData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMath</name>
    <filename>classKassiopeia_1_1KSComponentMath.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentMathData</name>
    <filename>classkatrin_1_1KSComponentMathData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMaximum</name>
    <filename>classKassiopeia_1_1KSComponentMaximum.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMaximumAt</name>
    <filename>classKassiopeia_1_1KSComponentMaximumAt.html</filename>
    <templarg>XValueType</templarg>
    <templarg>XValueTypeSource</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentMaximumAtData</name>
    <filename>classkatrin_1_1KSComponentMaximumAtData.html</filename>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentMaximumData</name>
    <filename>classkatrin_1_1KSComponentMaximumData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMember</name>
    <filename>classKassiopeia_1_1KSComponentMember.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMember&lt; const XValueType &amp;(XParentType::*)(void) const  &gt;</name>
    <filename>classKassiopeia_1_1KSComponentMember_3_01const_01XValueType_01_6_07XParentType_1_1_5_08_07void_08_01const_01_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMember&lt; XValueType(XParentType::*)(void) const  &gt;</name>
    <filename>classKassiopeia_1_1KSComponentMember_3_01XValueType_07XParentType_1_1_5_08_07void_08_01const_01_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentMemberData</name>
    <filename>classkatrin_1_1KSComponentMemberData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMemberFactory</name>
    <filename>classKassiopeia_1_1KSComponentMemberFactory.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMemberFactory&lt; const XValueType &amp;(XParentType::*)() const  &gt;</name>
    <filename>classKassiopeia_1_1KSComponentMemberFactory_3_01const_01XValueType_01_6_07XParentType_1_1_5_08_07_08_01const_01_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <base>Kassiopeia::KSComponentFactory</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMemberFactory&lt; XValueType(XParentType::*)() const  &gt;</name>
    <filename>classKassiopeia_1_1KSComponentMemberFactory_3_01XValueType_07XParentType_1_1_5_08_07_08_01const_01_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <base>Kassiopeia::KSComponentFactory</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMinimum</name>
    <filename>classKassiopeia_1_1KSComponentMinimum.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentMinimumAt</name>
    <filename>classKassiopeia_1_1KSComponentMinimumAt.html</filename>
    <templarg>XValueType</templarg>
    <templarg>XValueTypeSource</templarg>
    <base>Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentMinimumAtData</name>
    <filename>classkatrin_1_1KSComponentMinimumAtData.html</filename>
  </compound>
  <compound kind="class">
    <name>katrin::KSComponentMinimumData</name>
    <filename>classkatrin_1_1KSComponentMinimumData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentTemplate</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KESSElasticElsepa, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KESSInelasticBetheFano, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KESSInelasticPenn, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KESSSurfaceInteraction, KSSurfaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSElectricField &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSEvent &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenDirectionSphericalComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenDirectionSurfaceComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenEnergyBetaDecay, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenEnergyComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenEnergyKryptonEvent, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenEnergyLeadEvent, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenEnergyRadonEvent, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenEnergyRydberg, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenerator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenGeneratorComposite, KSGenerator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenGeneratorSimulation, KSGenerator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenLComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenLStatistical, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenLUniformMaxN, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenMomentumRectangularComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenNComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionCylindricalComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionFluxTube, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionFrustrumComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionHomogeneousFluxTube, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionMask, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionMeshSurfaceRandom, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionRectangularComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionSpaceRandom, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionSphericalComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionSurfaceAdjustmentStep, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenPositionSurfaceRandom, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenSpecial &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenSpinComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenTimeComposite, KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueAngleCosine, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueAngleSpherical, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueBoltzmann, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueFix, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueFormula, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueGauss, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueGeneralizedGauss, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueHistogram, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueList, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValuePareto, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueRadiusCylindrical, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueRadiusFraction, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueRadiusSpherical, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueSet, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueUniform, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGenValueZFrustrum, KSGenValue &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGeoSide, KSSide &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGeoSpace, KSSpace &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSGeoSurface, KSSurface &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorArgon, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorArgonDoubleIonisation, KSIntCalculatorArgon &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorArgonElastic, KSIntCalculatorArgon &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorArgonExcitation, KSIntCalculatorArgon &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorArgonSingleIonisation, KSIntCalculatorArgon &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorConstant, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenDissoziation10, KSIntCalculatorHydrogenExcitationBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenDissoziation15, KSIntCalculatorHydrogenExcitationBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenElastic, KSIntCalculatorHydrogenElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenElasticBase, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationB, KSIntCalculatorHydrogenExcitationBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationBase, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationBC, KSIntCalculatorHydrogenExcitationBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationC, KSIntCalculatorHydrogenExcitationBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationElectronic, KSIntCalculatorHydrogenExcitationBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenIonisation, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenIonisationOld, KSIntCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot02, KSIntCalculatorHydrogenElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot13, KSIntCalculatorHydrogenElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot20, KSIntCalculatorHydrogenElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorHydrogenVib, KSIntCalculatorHydrogenElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorTritiumElastic, KSIntCalculatorTritiumElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorTritiumElasticBase, KSIntCalculatorHydrogenElasticBase &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorTritiumRot02, KSIntCalculatorHydrogenRot02 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorTritiumRot13, KSIntCalculatorHydrogenRot13 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorTritiumRot20, KSIntCalculatorHydrogenRot20 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntCalculatorTritiumVib, KSIntCalculatorHydrogenVib &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecay, KSSpaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorDeathConstRate, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorFerencBBRTransition, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorFerencIonisation, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorFerencSpontaneous, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovDeExcitation, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovExcitation, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovIonisation, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovSpontaneous, KSIntDecayCalculator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDensity &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntDensityConstant, KSIntDensity &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntScattering, KSSpaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntSpinFlip, KSSpaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntSurfaceDiffuse, KSSurfaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntSurfaceMultiplication, KSSurfaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSIntSurfaceSpecular, KSSurfaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSMagneticField &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSModDynamicEnhancement, KSStepModifier &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSNavMeshedSpace, KSSpaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSNavSpace, KSSpaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSNavSurface, KSSurfaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRoot &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootElectricField, KSElectricField &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootGenerator, KSGenerator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootMagneticField, KSMagneticField &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootSpace, KSSpace &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootSpaceInteraction, KSSpaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootSpaceNavigator, KSSpaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootStepModifier, KSStepModifier &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootSurfaceInteraction, KSSurfaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootSurfaceNavigator, KSSurfaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootTerminator, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootTrajectory, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRootWriter, KSWriter &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSRun &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSide &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSimulation &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSpace &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSpaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSpaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSStep &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSStepModifier &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSurface &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSurfaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSSurfaceNavigator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermDeath, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxEnergy, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxLength, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxLongEnergy, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxR, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxSteps, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxTime, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMaxZ, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMinDistance, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMinEnergy, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMinLongEnergy, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMinR, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermMinZ, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermOutput&lt; XValueType &gt;, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermSecondaries, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermStepsize, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTermTrapped, KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrack &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlBChange &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlCyclotron &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlEnergy &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlLength &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlMagneticMoment &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlMDot &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlMomentumNumericalError &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlPositionNumericalError &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlSpinPrecession &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajControlTime &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRK54 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRK65 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRK8 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRK86 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRK87 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRKDP54 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorRKDP853 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajIntegratorSym4 &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajInterpolatorContinuousRungeKutta &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajInterpolatorFast &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajInterpolatorHermite &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTermConstantForcePropagation &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTermDrift &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTermGravity &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTermGyration &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTermPropagation &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTermSynchrotron &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryAdiabatic, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryAdiabaticSpin, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryElectric, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryExact, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryExactSpin, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryExactTrapped, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryLinear, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSTrajTrajectoryMagnetic, KSTrajectory &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteASCII, KSWriter &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriter &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteROOT, KSWriter &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteROOTCondition &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteROOTConditionOutput&lt; XValueType &gt;, KSWriteROOTCondition &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteROOTConditionStep&lt; XValueType &gt;, KSWriteROOTCondition &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteROOTConditionTerminator, KSWriteROOTCondition &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>KSComponentTemplate&lt; KSWriteVTK, KSWriter &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentTemplate&lt; XThisType, void, void, void &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate_3_01XThisType_00_01void_00_01void_00_01void_01_4.html</filename>
    <templarg></templarg>
    <base virtualness="virtual">Kassiopeia::KSComponent</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentTemplate&lt; XThisType, XFirstParentType, void, void &gt;</name>
    <filename>classKassiopeia_1_1KSComponentTemplate_3_01XThisType_00_01XFirstParentType_00_01void_00_01void_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <base virtualness="virtual">Kassiopeia::KSComponent</base>
    <base>XFirstParentType</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentValue</name>
    <filename>classKassiopeia_1_1KSComponentValue.html</filename>
    <templarg>XValueType</templarg>
  </compound>
  <compound kind="class">
    <name>KSComponentValue&lt; XValueTypeSource &gt;</name>
    <filename>classKassiopeia_1_1KSComponentValue.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentValueDelta</name>
    <filename>classKassiopeia_1_1KSComponentValueDelta.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponentValue</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentValueIntegral</name>
    <filename>classKassiopeia_1_1KSComponentValueIntegral.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponentValue</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentValueMaximum</name>
    <filename>classKassiopeia_1_1KSComponentValueMaximum.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponentValue</base>
  </compound>
  <compound kind="class">
    <name>KSComponentValueMaximum&lt; XValueTypeSource &gt;</name>
    <filename>classKassiopeia_1_1KSComponentValueMaximum.html</filename>
    <base>KSComponentValue&lt; XValueTypeSource &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSComponentValueMinimum</name>
    <filename>classKassiopeia_1_1KSComponentValueMinimum.html</filename>
    <templarg>XValueType</templarg>
    <base>Kassiopeia::KSComponentValue</base>
  </compound>
  <compound kind="class">
    <name>KSComponentValueMinimum&lt; XValueTypeSource &gt;</name>
    <filename>classKassiopeia_1_1KSComponentValueMinimum.html</filename>
    <base>KSComponentValue&lt; XValueTypeSource &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCondition</name>
    <filename>classKassiopeia_1_1KSCondition.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSCyclicIterator</name>
    <filename>classKassiopeia_1_1KSCyclicIterator.html</filename>
    <templarg>XType</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSDictionary</name>
    <filename>classKassiopeia_1_1KSDictionary.html</filename>
    <templarg>XType</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSDivideExpression</name>
    <filename>classKassiopeia_1_1KSDivideExpression.html</filename>
    <templarg>XType</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSElectricField</name>
    <filename>classKassiopeia_1_1KSElectricField.html</filename>
    <base>KSComponentTemplate&lt; KSElectricField &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSElectricKEMField</name>
    <filename>classKassiopeia_1_1KSElectricKEMField.html</filename>
    <base>Kassiopeia::KSElectricField</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSEvent</name>
    <filename>classKassiopeia_1_1KSEvent.html</filename>
    <base>KSComponentTemplate&lt; KSEvent &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSFieldMagneticSuperPositionData</name>
    <filename>classkatrin_1_1KSFieldMagneticSuperPositionData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenConversion</name>
    <filename>classKassiopeia_1_1KSGenConversion.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenCreator</name>
    <filename>classKassiopeia_1_1KSGenCreator.html</filename>
    <base>KSComponentTemplate&lt; KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenDirectionSphericalComposite</name>
    <filename>classKassiopeia_1_1KSGenDirectionSphericalComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenDirectionSphericalComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenDirectionSurfaceComposite</name>
    <filename>classKassiopeia_1_1KSGenDirectionSurfaceComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenDirectionSurfaceComposite, KSGenCreator &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>AddSurface</name>
      <anchorfile>classKassiopeia_1_1KSGenDirectionSurfaceComposite.html</anchorfile>
      <anchor>a68dd0532e10dcb255e350d08fad2cfb1</anchor>
      <arglist>(KGeoBag::KGSurface *aSurface)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenDirectionSurfaceComposite.html</anchorfile>
      <anchor>a4460885fcd6d7b5b290457ec5a160a16</anchor>
      <arglist>(KSParticleQueue *aParticleList)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>RemoveSurface</name>
      <anchorfile>classKassiopeia_1_1KSGenDirectionSurfaceComposite.html</anchorfile>
      <anchor>a297cff2fa633f64bf4c74f66a70aed7f</anchor>
      <arglist>(KGeoBag::KGSurface *aSurface)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenEnergyBetaDecay</name>
    <filename>classKassiopeia_1_1KSGenEnergyBetaDecay.html</filename>
    <base>KSComponentTemplate&lt; KSGenEnergyBetaDecay, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenEnergyComposite</name>
    <filename>classKassiopeia_1_1KSGenEnergyComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenEnergyComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenEnergyKryptonEvent</name>
    <filename>classKassiopeia_1_1KSGenEnergyKryptonEvent.html</filename>
    <base>KSComponentTemplate&lt; KSGenEnergyKryptonEvent, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenEnergyLeadEvent</name>
    <filename>classKassiopeia_1_1KSGenEnergyLeadEvent.html</filename>
    <base>KSComponentTemplate&lt; KSGenEnergyLeadEvent, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenEnergyRadonEvent</name>
    <filename>classKassiopeia_1_1KSGenEnergyRadonEvent.html</filename>
    <base>KSComponentTemplate&lt; KSGenEnergyRadonEvent, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenEnergyRydberg</name>
    <filename>classKassiopeia_1_1KSGenEnergyRydberg.html</filename>
    <base>KSComponentTemplate&lt; KSGenEnergyRydberg, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenerator</name>
    <filename>classKassiopeia_1_1KSGenerator.html</filename>
    <base>KSComponentTemplate&lt; KSGenerator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenGeneratorComposite</name>
    <filename>classKassiopeia_1_1KSGenGeneratorComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenGeneratorComposite, KSGenerator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenGeneratorSimulation</name>
    <filename>classKassiopeia_1_1KSGenGeneratorSimulation.html</filename>
    <base>KSComponentTemplate&lt; KSGenGeneratorSimulation, KSGenerator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenLComposite</name>
    <filename>classKassiopeia_1_1KSGenLComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenLComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenLStatistical</name>
    <filename>classKassiopeia_1_1KSGenLStatistical.html</filename>
    <base>KSComponentTemplate&lt; KSGenLStatistical, KSGenCreator &gt;</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenLStatistical.html</anchorfile>
      <anchor>a4ecfe0e4e8927c0bcc72bde910e78985</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenLUniformMaxN</name>
    <filename>classKassiopeia_1_1KSGenLUniformMaxN.html</filename>
    <base>KSComponentTemplate&lt; KSGenLUniformMaxN, KSGenCreator &gt;</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenLUniformMaxN.html</anchorfile>
      <anchor>a4b5d3a682d17eca7460024c166940de7</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenMomentumRectangularComposite</name>
    <filename>classKassiopeia_1_1KSGenMomentumRectangularComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenMomentumRectangularComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenNComposite</name>
    <filename>classKassiopeia_1_1KSGenNComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenNComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionCylindricalComposite</name>
    <filename>classKassiopeia_1_1KSGenPositionCylindricalComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionCylindricalComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionFluxTube</name>
    <filename>classKassiopeia_1_1KSGenPositionFluxTube.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionFluxTube, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionFrustrumComposite</name>
    <filename>classKassiopeia_1_1KSGenPositionFrustrumComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionFrustrumComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionHomogeneousFluxTube</name>
    <filename>classKassiopeia_1_1KSGenPositionHomogeneousFluxTube.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionHomogeneousFluxTube, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionMask</name>
    <filename>classKassiopeia_1_1KSGenPositionMask.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionMask, KSGenCreator &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>AddAllowedSpace</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMask.html</anchorfile>
      <anchor>a59f775eaa01eeb4acbbe8b50eeb82bf1</anchor>
      <arglist>(const KGeoBag::KGSpace *aSpace)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>AddForbiddenSpace</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMask.html</anchorfile>
      <anchor>a746b36d5caf43a86e6443280e25bd2f4</anchor>
      <arglist>(const KGeoBag::KGSpace *aSpace)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMask.html</anchorfile>
      <anchor>a2b71bc74b0d661cc4a04a37df7d5ba4f</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>SetGenerator</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMask.html</anchorfile>
      <anchor>a6ecbb8d0256231e0682fe01a5768c4fb</anchor>
      <arglist>(KSGenCreator *aGenerator)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>SetMaxRetries</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMask.html</anchorfile>
      <anchor>a5ec5342de8aad52e4642161b0f12adc7</anchor>
      <arglist>(unsigned int &amp;aNumber)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionMeshSurfaceRandom</name>
    <filename>classKassiopeia_1_1KSGenPositionMeshSurfaceRandom.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionMeshSurfaceRandom, KSGenCreator &gt;</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMeshSurfaceRandom.html</anchorfile>
      <anchor>a72f178ced2ff8431332a3712b42f6cad</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>VisitExtendedSurface</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMeshSurfaceRandom.html</anchorfile>
      <anchor>a840b62ac3ab25470503f66f5c3b878b6</anchor>
      <arglist>(KGeoBag::KGExtendedSurface&lt; KGeoBag::KGMesh &gt; *aSurface)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>VisitSurface</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionMeshSurfaceRandom.html</anchorfile>
      <anchor>a6902381531a29458e82b807506b51fe7</anchor>
      <arglist>(KGeoBag::KGSurface *aSurface)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionRectangularComposite</name>
    <filename>classKassiopeia_1_1KSGenPositionRectangularComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionRectangularComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionSpaceRandom</name>
    <filename>classKassiopeia_1_1KSGenPositionSpaceRandom.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionSpaceRandom, KSGenCreator &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>AddSpace</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSpaceRandom.html</anchorfile>
      <anchor>a6385a823b5a693bc29d1ae2d90b9385b</anchor>
      <arglist>(KGeoBag::KGSpace *aSpace)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSpaceRandom.html</anchorfile>
      <anchor>a63ccc82e6a907b75821ab4fb9693a1f9</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>RemoveSpace</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSpaceRandom.html</anchorfile>
      <anchor>af1a7b245c57c38ca45e55b6dff035b5f</anchor>
      <arglist>(KGeoBag::KGSpace *aSpace)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionSphericalComposite</name>
    <filename>classKassiopeia_1_1KSGenPositionSphericalComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionSphericalComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionSurfaceAdjustmentStep</name>
    <filename>classKassiopeia_1_1KSGenPositionSurfaceAdjustmentStep.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionSurfaceAdjustmentStep, KSGenCreator &gt;</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSurfaceAdjustmentStep.html</anchorfile>
      <anchor>a9eababa2e6f0c254dfeae6f0adb2e275</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenPositionSurfaceRandom</name>
    <filename>classKassiopeia_1_1KSGenPositionSurfaceRandom.html</filename>
    <base>KSComponentTemplate&lt; KSGenPositionSurfaceRandom, KSGenCreator &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>AddSurface</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSurfaceRandom.html</anchorfile>
      <anchor>a7eb2a2cd3643721da6a6d4744a0f742e</anchor>
      <arglist>(KGeoBag::KGSurface *aSurface)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>Dice</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSurfaceRandom.html</anchorfile>
      <anchor>a4f50df7b8971bc3c2cb6c220ad2bca02</anchor>
      <arglist>(KSParticleQueue *aPrimaries)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>RemoveSurface</name>
      <anchorfile>classKassiopeia_1_1KSGenPositionSurfaceRandom.html</anchorfile>
      <anchor>a95f004b46a4b0f198f0a503feef5f00f</anchor>
      <arglist>(KGeoBag::KGSurface *aSurface)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenRelaxation</name>
    <filename>classKassiopeia_1_1KSGenRelaxation.html</filename>
    <class kind="struct">Kassiopeia::KSGenRelaxation::line_struct</class>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenShakeOff</name>
    <filename>classKassiopeia_1_1KSGenShakeOff.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenSpecial</name>
    <filename>classKassiopeia_1_1KSGenSpecial.html</filename>
    <base>KSComponentTemplate&lt; KSGenSpecial &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenSpinComposite</name>
    <filename>classKassiopeia_1_1KSGenSpinComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenSpinComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenTimeComposite</name>
    <filename>classKassiopeia_1_1KSGenTimeComposite.html</filename>
    <base>KSComponentTemplate&lt; KSGenTimeComposite, KSGenCreator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValue</name>
    <filename>classKassiopeia_1_1KSGenValue.html</filename>
    <base>KSComponentTemplate&lt; KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueAngleCosine</name>
    <filename>classKassiopeia_1_1KSGenValueAngleCosine.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueAngleCosine, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueAngleSpherical</name>
    <filename>classKassiopeia_1_1KSGenValueAngleSpherical.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueAngleSpherical, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueBoltzmann</name>
    <filename>classKassiopeia_1_1KSGenValueBoltzmann.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueBoltzmann, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueFix</name>
    <filename>classKassiopeia_1_1KSGenValueFix.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueFix, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueFormula</name>
    <filename>classKassiopeia_1_1KSGenValueFormula.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueFormula, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueGauss</name>
    <filename>classKassiopeia_1_1KSGenValueGauss.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueGauss, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueGeneralizedGauss</name>
    <filename>classKassiopeia_1_1KSGenValueGeneralizedGauss.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueGeneralizedGauss, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueHistogram</name>
    <filename>classKassiopeia_1_1KSGenValueHistogram.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueHistogram, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueList</name>
    <filename>classKassiopeia_1_1KSGenValueList.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueList, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValuePareto</name>
    <filename>classKassiopeia_1_1KSGenValuePareto.html</filename>
    <base>KSComponentTemplate&lt; KSGenValuePareto, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueRadiusCylindrical</name>
    <filename>classKassiopeia_1_1KSGenValueRadiusCylindrical.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueRadiusCylindrical, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueRadiusFraction</name>
    <filename>classKassiopeia_1_1KSGenValueRadiusFraction.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueRadiusFraction, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueRadiusSpherical</name>
    <filename>classKassiopeia_1_1KSGenValueRadiusSpherical.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueRadiusSpherical, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueSet</name>
    <filename>classKassiopeia_1_1KSGenValueSet.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueSet, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueUniform</name>
    <filename>classKassiopeia_1_1KSGenValueUniform.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueUniform, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGenValueZFrustrum</name>
    <filename>classKassiopeia_1_1KSGenValueZFrustrum.html</filename>
    <base>KSComponentTemplate&lt; KSGenValueZFrustrum, KSGenValue &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGeoSide</name>
    <filename>classKassiopeia_1_1KSGeoSide.html</filename>
    <base>KSComponentTemplate&lt; KSGeoSide, KSSide &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGeoSpace</name>
    <filename>classKassiopeia_1_1KSGeoSpace.html</filename>
    <base>KSComponentTemplate&lt; KSGeoSpace, KSSpace &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSGeoSurface</name>
    <filename>classKassiopeia_1_1KSGeoSurface.html</filename>
    <base>KSComponentTemplate&lt; KSGeoSurface, KSSurface &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculator</name>
    <filename>classKassiopeia_1_1KSIntCalculator.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgon</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgon.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgon, KSIntCalculator &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgon, KSIntCalculator &gt;</base>
    <member kind="function" virtualness="virtual">
      <type>virtual double</type>
      <name>GetDifferentialCrossSectionAt</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a539ec07093f6b2b89439fe9a5ea2310b</anchor>
      <arglist>(const double &amp;anEnergy, const double &amp;anAngle) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>ComputeSupportingPoints</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>afb737b59e08382c046ed4f1910a1e3ac</anchor>
      <arglist>(unsigned int numOfParameters)</arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>GetCrossSectionAt</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a8ebb4e2d3587a29f8fa9ad5a0b2d137b</anchor>
      <arglist>(const double &amp;anEnergy) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="pure">
      <type>virtual double</type>
      <name>GetEnergyLoss</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a5e99fe4a196e193e6f44afef9687adc9</anchor>
      <arglist>(const double &amp;anEnergy, const double &amp;theta) const =0</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetInterpolation</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>aba53b4cb48b195e0951b40989bca74be</anchor>
      <arglist>(const double &amp;anEnergy, std::map&lt; double, double &gt;::iterator &amp;point) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetInterpolationForTotalCrossSection</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a82e483d3fe4e6f610f7f61e074533a5d</anchor>
      <arglist>(const double &amp;anEnergy, std::map&lt; double, double &gt;::iterator &amp;point) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetLowerExtrapolation</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a3e6d700e9d76dcac5e421a28d5d93057</anchor>
      <arglist>(const double &amp;anEnergy, std::map&lt; double, double &gt;::iterator &amp;point) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetLowerExtrapolationForTotalCrossSection</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a9f83c57f28a7884dfeafd7b3f870c53a</anchor>
      <arglist>(const double &amp;anEnergy, std::map&lt; double, double &gt;::iterator &amp;point) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetTheta</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a3f568a1ad82699c04a6cac1ea97cdbde</anchor>
      <arglist>(const double &amp;anEnergy) const </arglist>
    </member>
    <member kind="function" protection="protected">
      <type>double</type>
      <name>GetTotalCrossSectionAt</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a257c8fcb2d342f4eabed392a04e86557</anchor>
      <arglist>(const double &amp;anEnergy) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetUpperExtrapolation</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>abc99cdc9b509e26ac4bfb709daeb5d76</anchor>
      <arglist>(const double &amp;anEnergy, std::map&lt; double, double &gt;::iterator &amp;point) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual double</type>
      <name>GetUpperExtrapolationForTotalCrossSection</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a748f555edcedf21b0b8bdf0e8442f170</anchor>
      <arglist>(const double &amp;anEnergy, std::map&lt; double, double &gt;::iterator &amp;point) const </arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>InitializeDifferentialCrossSection</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>ab510fa76acf94ffe30aab6d34ab237c1</anchor>
      <arglist>(unsigned int numOfParameters)</arglist>
    </member>
    <member kind="function" protection="protected" virtualness="virtual">
      <type>virtual void</type>
      <name>InitializeTotalCrossSection</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a041aed8465b6aeb6c229e3ced722414d</anchor>
      <arglist>(unsigned int numOfParameters)</arglist>
    </member>
    <member kind="variable" protection="protected">
      <type>katrin::KMathBilinearInterpolator&lt; double &gt; *</type>
      <name>fDifferentialCrossSectionInterpolator</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgon.html</anchorfile>
      <anchor>a8256cdf0a3e0eaf38d5f2222dccea3ae</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>KSIntCalculatorArgonData</name>
    <filename>classKSIntCalculatorArgonData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonDataReader</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonDataReader.html</filename>
    <member kind="function">
      <type>std::map&lt; double, double &gt; *</type>
      <name>GetData</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDataReader.html</anchorfile>
      <anchor>ab6717f655bf8f30bb2c8645596205582</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt; *</type>
      <name>GetParameters</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDataReader.html</anchorfile>
      <anchor>accf74fd4d63a3cffbe032cabd035d595</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>Read</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDataReader.html</anchorfile>
      <anchor>ac3bcbf5773ba8a9bdb2f71db783a3ec9</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonDifferentialCrossSectionReader</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonDifferentialCrossSectionReader.html</filename>
    <member kind="function">
      <type>std::map&lt; double *, double &gt; *</type>
      <name>GetData</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDifferentialCrossSectionReader.html</anchorfile>
      <anchor>ac9c74d50fa927273b0900121776246d4</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt; *</type>
      <name>GetParameters</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDifferentialCrossSectionReader.html</anchorfile>
      <anchor>a4ecd190e5d7de73646197a48bc59d56c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>Read</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDifferentialCrossSectionReader.html</anchorfile>
      <anchor>a6af2d5bc167455560c86c14412e37230</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>Readlx</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonDifferentialCrossSectionReader.html</anchorfile>
      <anchor>a72239ff747b095b7e3a97d11433082fa</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonDoubleIonisation</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonDoubleIonisation.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonDoubleIonisation, KSIntCalculatorArgon &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonDoubleIonisation, KSIntCalculatorArgon &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonElastic</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonElastic.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonElastic, KSIntCalculatorArgon &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonElastic, KSIntCalculatorArgon &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonExcitation</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonExcitation.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonExcitation, KSIntCalculatorArgon &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonExcitation, KSIntCalculatorArgon &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSIntCalculatorArgonSet</name>
    <filename>classkatrin_1_1KSIntCalculatorArgonSet.html</filename>
    <base>katrin::KSIntCalculatorSet</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonSingleIonisation</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonSingleIonisation.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonSingleIonisation, KSIntCalculatorArgon &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorArgonSingleIonisation, KSIntCalculatorArgon &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorArgonTotalCrossSectionReader</name>
    <filename>classKassiopeia_1_1KSIntCalculatorArgonTotalCrossSectionReader.html</filename>
    <member kind="function">
      <type>std::map&lt; double, double &gt; *</type>
      <name>GetData</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonTotalCrossSectionReader.html</anchorfile>
      <anchor>a3655f0e8991e987c01289542cebb6eb1</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; double &gt; *</type>
      <name>GetParameters</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonTotalCrossSectionReader.html</anchorfile>
      <anchor>ae05d8805051d3e8f1ce5f15ca5bc118b</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>Read</name>
      <anchorfile>classKassiopeia_1_1KSIntCalculatorArgonTotalCrossSectionReader.html</anchorfile>
      <anchor>af974fbc0aa7f8ee3ecd8022f191f2aea</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorConstant</name>
    <filename>classKassiopeia_1_1KSIntCalculatorConstant.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorConstant, KSIntCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSIntCalculatorHydrogenData</name>
    <filename>classKSIntCalculatorHydrogenData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenDissoziation10</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenDissoziation10.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenDissoziation10, KSIntCalculatorHydrogenExcitationBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenDissoziation15</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenDissoziation15.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenDissoziation15, KSIntCalculatorHydrogenExcitationBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenElastic</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenElastic.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenElastic, KSIntCalculatorHydrogenElasticBase &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenElastic, KSIntCalculatorHydrogenElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenElasticBase</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenElasticBase.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenElasticBase, KSIntCalculator &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenElasticBase, KSIntCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenExcitationB</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenExcitationB.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationB, KSIntCalculatorHydrogenExcitationBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenExcitationBase</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenExcitationBase.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationBase, KSIntCalculator &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationBase, KSIntCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenExcitationBC</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenExcitationBC.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationBC, KSIntCalculatorHydrogenExcitationBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenExcitationC</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenExcitationC.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationC, KSIntCalculatorHydrogenExcitationBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenExcitationElectronic</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenExcitationElectronic.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenExcitationElectronic, KSIntCalculatorHydrogenExcitationBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenIonisation</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenIonisation.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenIonisation, KSIntCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenIonisationOld</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenIonisationOld.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenIonisationOld, KSIntCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenRot02</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenRot02.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot02, KSIntCalculatorHydrogenElasticBase &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot02, KSIntCalculatorHydrogenElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenRot13</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenRot13.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot13, KSIntCalculatorHydrogenElasticBase &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot13, KSIntCalculatorHydrogenElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenRot20</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenRot20.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot20, KSIntCalculatorHydrogenElasticBase &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenRot20, KSIntCalculatorHydrogenElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSIntCalculatorHydrogenSet</name>
    <filename>classkatrin_1_1KSIntCalculatorHydrogenSet.html</filename>
    <base>katrin::KSIntCalculatorSet</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorHydrogenVib</name>
    <filename>classKassiopeia_1_1KSIntCalculatorHydrogenVib.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenVib, KSIntCalculatorHydrogenElasticBase &gt;</base>
    <base>KSComponentTemplate&lt; KSIntCalculatorHydrogenVib, KSIntCalculatorHydrogenElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSIntCalculatorKESSSet</name>
    <filename>classkatrin_1_1KSIntCalculatorKESSSet.html</filename>
    <base>katrin::KSIntCalculatorSet</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSIntCalculatorSet</name>
    <filename>classkatrin_1_1KSIntCalculatorSet.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorTritiumElastic</name>
    <filename>classKassiopeia_1_1KSIntCalculatorTritiumElastic.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorTritiumElastic, KSIntCalculatorTritiumElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorTritiumElasticBase</name>
    <filename>classKassiopeia_1_1KSIntCalculatorTritiumElasticBase.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorTritiumElasticBase, KSIntCalculatorHydrogenElasticBase &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorTritiumRot02</name>
    <filename>classKassiopeia_1_1KSIntCalculatorTritiumRot02.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorTritiumRot02, KSIntCalculatorHydrogenRot02 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorTritiumRot13</name>
    <filename>classKassiopeia_1_1KSIntCalculatorTritiumRot13.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorTritiumRot13, KSIntCalculatorHydrogenRot13 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorTritiumRot20</name>
    <filename>classKassiopeia_1_1KSIntCalculatorTritiumRot20.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorTritiumRot20, KSIntCalculatorHydrogenRot20 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntCalculatorTritiumVib</name>
    <filename>classKassiopeia_1_1KSIntCalculatorTritiumVib.html</filename>
    <base>KSComponentTemplate&lt; KSIntCalculatorTritiumVib, KSIntCalculatorHydrogenVib &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecay</name>
    <filename>classKassiopeia_1_1KSIntDecay.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecay, KSSpaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculator</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculator.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorDeathConstRate</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorDeathConstRate.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorDeathConstRate, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorFerencBBRTransition</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorFerencBBRTransition.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorFerencBBRTransition, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorFerencIonisation</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorFerencIonisation.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorFerencIonisation, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorFerencSpontaneous</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorFerencSpontaneous.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorFerencSpontaneous, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorGlukhovDeExcitation</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorGlukhovDeExcitation.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovDeExcitation, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorGlukhovExcitation</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorGlukhovExcitation.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovExcitation, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorGlukhovIonisation</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorGlukhovIonisation.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovIonisation, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDecayCalculatorGlukhovSpontaneous</name>
    <filename>classKassiopeia_1_1KSIntDecayCalculatorGlukhovSpontaneous.html</filename>
    <base>KSComponentTemplate&lt; KSIntDecayCalculatorGlukhovSpontaneous, KSIntDecayCalculator &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSIntDecayCalculatorSet</name>
    <filename>classkatrin_1_1KSIntDecayCalculatorSet.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDensity</name>
    <filename>classKassiopeia_1_1KSIntDensity.html</filename>
    <base>KSComponentTemplate&lt; KSIntDensity &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntDensityConstant</name>
    <filename>classKassiopeia_1_1KSIntDensityConstant.html</filename>
    <base>KSComponentTemplate&lt; KSIntDensityConstant, KSIntDensity &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntScattering</name>
    <filename>classKassiopeia_1_1KSIntScattering.html</filename>
    <base>KSComponentTemplate&lt; KSIntScattering, KSSpaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntSpinFlip</name>
    <filename>classKassiopeia_1_1KSIntSpinFlip.html</filename>
    <base>KSComponentTemplate&lt; KSIntSpinFlip, KSSpaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntSurfaceDiffuse</name>
    <filename>classKassiopeia_1_1KSIntSurfaceDiffuse.html</filename>
    <base>KSComponentTemplate&lt; KSIntSurfaceDiffuse, KSSurfaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntSurfaceMultiplication</name>
    <filename>classKassiopeia_1_1KSIntSurfaceMultiplication.html</filename>
    <base>KSComponentTemplate&lt; KSIntSurfaceMultiplication, KSSurfaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSIntSurfaceSpecular</name>
    <filename>classKassiopeia_1_1KSIntSurfaceSpecular.html</filename>
    <base>KSComponentTemplate&lt; KSIntSurfaceSpecular, KSSurfaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSList</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
    <templarg>XType</templarg>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSElectricField &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSGenCreator &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSGenSpecial &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSMagneticField &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSMathControl &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSMathDifferentiator &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSSpaceInteraction &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSStepModifier &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSTerminator &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSWriter &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>KSList&lt; Kassiopeia::KSWriteROOTCondition &gt;</name>
    <filename>classKassiopeia_1_1KSList.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMagneticField</name>
    <filename>classKassiopeia_1_1KSMagneticField.html</filename>
    <base>KSComponentTemplate&lt; KSMagneticField &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMagneticKEMField</name>
    <filename>classKassiopeia_1_1KSMagneticKEMField.html</filename>
    <base>Kassiopeia::KSMagneticField</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathAdd</name>
    <filename>classKassiopeia_1_1KSMathAdd.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathArray</name>
    <filename>classKassiopeia_1_1KSMathArray.html</filename>
    <templarg>XDimension</templarg>
  </compound>
  <compound kind="class">
    <name>KSMathArray&lt; 10 &gt;</name>
    <filename>classKassiopeia_1_1KSMathArray.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathArray&lt; 12 &gt;</name>
    <filename>classKassiopeia_1_1KSMathArray.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathArray&lt; 5 &gt;</name>
    <filename>classKassiopeia_1_1KSMathArray.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathArray&lt; 8 &gt;</name>
    <filename>classKassiopeia_1_1KSMathArray.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathControl</name>
    <filename>classKassiopeia_1_1KSMathControl.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathControl&lt; KSMathSystem&lt; XValueType, XDerivativeType, XErrorType &gt; &gt;</name>
    <filename>classKassiopeia_1_1KSMathControl_3_01KSMathSystem_3_01XValueType_00_01XDerivativeType_00_01XErrorType_01_4_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathDifferentiator</name>
    <filename>classKassiopeia_1_1KSMathDifferentiator.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathDifferentiator&lt; KSMathSystem&lt; XValueType, XDerivativeType, XErrorType &gt; &gt;</name>
    <filename>classKassiopeia_1_1KSMathDifferentiator_3_01KSMathSystem_3_01XValueType_00_01XDerivativeType_00_01XErrorType_01_4_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathDivide</name>
    <filename>classKassiopeia_1_1KSMathDivide.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathExpression</name>
    <filename>classKassiopeia_1_1KSMathExpression.html</filename>
    <templarg>XLeft</templarg>
    <templarg>XOperation</templarg>
    <templarg>XRight</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathExpression&lt; double, XOperation, XRight &gt;</name>
    <filename>classKassiopeia_1_1KSMathExpression_3_01double_00_01XOperation_00_01XRight_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathExpression&lt; XLeft, XOperation, double &gt;</name>
    <filename>classKassiopeia_1_1KSMathExpression_3_01XLeft_00_01XOperation_00_01double_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathIntegrator</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathIntegrator&lt; KSMathSystem&lt; XValueType, XDerivativeType, XErrorType &gt; &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator_3_01KSMathSystem_3_01XValueType_00_01XDerivativeType_00_01XErrorType_01_4_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajExactTrappedSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>KSMathIntegrator&lt; XSystemType &gt;</name>
    <filename>classKassiopeia_1_1KSMathIntegrator.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathInterpolator</name>
    <filename>classKassiopeia_1_1KSMathInterpolator.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathInterpolator&lt; KSMathSystem&lt; XValueType, XDerivativeType, XErrorType &gt; &gt;</name>
    <filename>classKassiopeia_1_1KSMathInterpolator_3_01KSMathSystem_3_01XValueType_00_01XDerivativeType_00_01XErrorType_01_4_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathMultiply</name>
    <filename>classKassiopeia_1_1KSMathMultiply.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRK65</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK65&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK65&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK65&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK65&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK65&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK65&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK65.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRK8</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRK86</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK86&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK86&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK86&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK86&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK86&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK86&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK86.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRK87</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK87&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK87&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK87&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK87&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK87&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK87&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK87.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajExactTrappedSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactTrappedSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRK8&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRK8.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRKDP54</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP54&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP54&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP54&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP54&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP54&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP54&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRKDP853</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP853&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP853&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP853&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP853&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP853&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKDP853&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKDP853.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathRKF54</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKF54&lt; KSTrajAdiabaticSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKF54&lt; KSTrajAdiabaticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajAdiabaticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKF54&lt; KSTrajElectricSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajElectricSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKF54&lt; KSTrajExactSpinSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSpinSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKF54&lt; KSTrajExactSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathRKF54&lt; KSTrajMagneticSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathRKF54.html</filename>
    <base>KSMathIntegrator&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathSubtract</name>
    <filename>classKassiopeia_1_1KSMathSubtract.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathSym4</name>
    <filename>classKassiopeia_1_1KSMathSym4.html</filename>
    <templarg>XSystemType</templarg>
    <base>KSMathIntegrator&lt; XSystemType &gt;</base>
  </compound>
  <compound kind="class">
    <name>KSMathSym4&lt; KSTrajExactTrappedSystem &gt;</name>
    <filename>classKassiopeia_1_1KSMathSym4.html</filename>
    <base>KSMathIntegrator&lt; KSTrajExactTrappedSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMathSystem</name>
    <filename>classKassiopeia_1_1KSMathSystem.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSModDynamicEnhancement</name>
    <filename>classKassiopeia_1_1KSModDynamicEnhancement.html</filename>
    <base>KSComponentTemplate&lt; KSModDynamicEnhancement, KSStepModifier &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMultiplyExpression</name>
    <filename>classKassiopeia_1_1KSMultiplyExpression.html</filename>
    <templarg>XType</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSMutex</name>
    <filename>classKassiopeia_1_1KSMutex.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSNavMeshedSpace</name>
    <filename>classKassiopeia_1_1KSNavMeshedSpace.html</filename>
    <base>KSComponentTemplate&lt; KSNavMeshedSpace, KSSpaceNavigator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSNavOctreeData</name>
    <filename>classKassiopeia_1_1KSNavOctreeData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSNavSpace</name>
    <filename>classKassiopeia_1_1KSNavSpace.html</filename>
    <base>KSComponentTemplate&lt; KSNavSpace, KSSpaceNavigator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSNavSurface</name>
    <filename>classKassiopeia_1_1KSNavSurface.html</filename>
    <base>KSComponentTemplate&lt; KSNavSurface, KSSurfaceNavigator &gt;</base>
  </compound>
  <compound kind="struct">
    <name>Kassiopeia::KSNumerical</name>
    <filename>structKassiopeia_1_1KSNumerical.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>Kassiopeia::KSNumerical&lt; bool &gt;</name>
    <filename>structKassiopeia_1_1KSNumerical_3_01bool_01_4.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSObject</name>
    <filename>classKassiopeia_1_1KSObject.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSParticle</name>
    <filename>classKassiopeia_1_1KSParticle.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSParticleFactory</name>
    <filename>classKassiopeia_1_1KSParticleFactory.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSPathResolver</name>
    <filename>classKassiopeia_1_1KSPathResolver.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadEventROOT</name>
    <filename>classKassiopeia_1_1KSReadEventROOT.html</filename>
    <base>Kassiopeia::KSReadIteratorROOT</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadFile</name>
    <filename>classKassiopeia_1_1KSReadFile.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadFileROOT</name>
    <filename>classKassiopeia_1_1KSReadFileROOT.html</filename>
    <base>Kassiopeia::KSReadFile</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadIterator</name>
    <filename>classKassiopeia_1_1KSReadIterator.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadIteratorROOT</name>
    <filename>classKassiopeia_1_1KSReadIteratorROOT.html</filename>
    <base>Kassiopeia::KSReadIterator</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadObjectROOT</name>
    <filename>classKassiopeia_1_1KSReadObjectROOT.html</filename>
    <base>Kassiopeia::KSReadIterator</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
    <base>Kassiopeia::KSReadSet</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadRunROOT</name>
    <filename>classKassiopeia_1_1KSReadRunROOT.html</filename>
    <base>Kassiopeia::KSReadIteratorROOT</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadSet</name>
    <filename>classKassiopeia_1_1KSReadSet.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadSet&lt; KSReadValue&lt; XType &gt; &gt;</name>
    <filename>classKassiopeia_1_1KSReadSet_3_01KSReadValue_3_01XType_01_4_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadStepROOT</name>
    <filename>classKassiopeia_1_1KSReadStepROOT.html</filename>
    <base>Kassiopeia::KSReadIteratorROOT</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadTrackROOT</name>
    <filename>classKassiopeia_1_1KSReadTrackROOT.html</filename>
    <base>Kassiopeia::KSReadIteratorROOT</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSReadValue</name>
    <filename>classKassiopeia_1_1KSReadValue.html</filename>
    <templarg>XType</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRoot</name>
    <filename>classKassiopeia_1_1KSRoot.html</filename>
    <base>KSComponentTemplate&lt; KSRoot &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootElectricField</name>
    <filename>classKassiopeia_1_1KSRootElectricField.html</filename>
    <base>KSComponentTemplate&lt; KSRootElectricField, KSElectricField &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootGenerator</name>
    <filename>classKassiopeia_1_1KSRootGenerator.html</filename>
    <base>KSComponentTemplate&lt; KSRootGenerator, KSGenerator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSROOTMagFieldPainter</name>
    <filename>classKassiopeia_1_1KSROOTMagFieldPainter.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootMagneticField</name>
    <filename>classKassiopeia_1_1KSRootMagneticField.html</filename>
    <base>KSComponentTemplate&lt; KSRootMagneticField, KSMagneticField &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSROOTPotentialPainter</name>
    <filename>classKassiopeia_1_1KSROOTPotentialPainter.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootSpace</name>
    <filename>classKassiopeia_1_1KSRootSpace.html</filename>
    <base>KSComponentTemplate&lt; KSRootSpace, KSSpace &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootSpaceInteraction</name>
    <filename>classKassiopeia_1_1KSRootSpaceInteraction.html</filename>
    <base>KSComponentTemplate&lt; KSRootSpaceInteraction, KSSpaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootSpaceNavigator</name>
    <filename>classKassiopeia_1_1KSRootSpaceNavigator.html</filename>
    <base>KSComponentTemplate&lt; KSRootSpaceNavigator, KSSpaceNavigator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootStepModifier</name>
    <filename>classKassiopeia_1_1KSRootStepModifier.html</filename>
    <base>KSComponentTemplate&lt; KSRootStepModifier, KSStepModifier &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootSurfaceInteraction</name>
    <filename>classKassiopeia_1_1KSRootSurfaceInteraction.html</filename>
    <base>KSComponentTemplate&lt; KSRootSurfaceInteraction, KSSurfaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootSurfaceNavigator</name>
    <filename>classKassiopeia_1_1KSRootSurfaceNavigator.html</filename>
    <base>KSComponentTemplate&lt; KSRootSurfaceNavigator, KSSurfaceNavigator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootTerminator</name>
    <filename>classKassiopeia_1_1KSRootTerminator.html</filename>
    <base>KSComponentTemplate&lt; KSRootTerminator, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSROOTTrackPainter</name>
    <filename>classKassiopeia_1_1KSROOTTrackPainter.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootTrajectory</name>
    <filename>classKassiopeia_1_1KSRootTrajectory.html</filename>
    <base>KSComponentTemplate&lt; KSRootTrajectory, KSTrajectory &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRootWriter</name>
    <filename>classKassiopeia_1_1KSRootWriter.html</filename>
    <base>KSComponentTemplate&lt; KSRootWriter, KSWriter &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSRun</name>
    <filename>classKassiopeia_1_1KSRun.html</filename>
    <base>KSComponentTemplate&lt; KSRun &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSide</name>
    <filename>classKassiopeia_1_1KSSide.html</filename>
    <base>KSComponentTemplate&lt; KSSide &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSimulation</name>
    <filename>classKassiopeia_1_1KSSimulation.html</filename>
    <base>KSComponentTemplate&lt; KSSimulation &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSpace</name>
    <filename>classKassiopeia_1_1KSSpace.html</filename>
    <base>KSComponentTemplate&lt; KSSpace &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSpaceInteraction</name>
    <filename>classKassiopeia_1_1KSSpaceInteraction.html</filename>
    <base>KSComponentTemplate&lt; KSSpaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSpaceNavigator</name>
    <filename>classKassiopeia_1_1KSSpaceNavigator.html</filename>
    <base>KSComponentTemplate&lt; KSSpaceNavigator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSStep</name>
    <filename>classKassiopeia_1_1KSStep.html</filename>
    <base>KSComponentTemplate&lt; KSStep &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSStepModifier</name>
    <filename>classKassiopeia_1_1KSStepModifier.html</filename>
    <base>KSComponentTemplate&lt; KSStepModifier &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSubtractExpression</name>
    <filename>classKassiopeia_1_1KSSubtractExpression.html</filename>
    <templarg>XLeft</templarg>
    <templarg>XRight</templarg>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSurface</name>
    <filename>classKassiopeia_1_1KSSurface.html</filename>
    <base>KSComponentTemplate&lt; KSSurface &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSurfaceInteraction</name>
    <filename>classKassiopeia_1_1KSSurfaceInteraction.html</filename>
    <base>KSComponentTemplate&lt; KSSurfaceInteraction &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSSurfaceNavigator</name>
    <filename>classKassiopeia_1_1KSSurfaceNavigator.html</filename>
    <base>KSComponentTemplate&lt; KSSurfaceNavigator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermDeath</name>
    <filename>classKassiopeia_1_1KSTermDeath.html</filename>
    <base>KSComponentTemplate&lt; KSTermDeath, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTerminator</name>
    <filename>classKassiopeia_1_1KSTerminator.html</filename>
    <base>KSComponentTemplate&lt; KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxEnergy</name>
    <filename>classKassiopeia_1_1KSTermMaxEnergy.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxEnergy, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxLength</name>
    <filename>classKassiopeia_1_1KSTermMaxLength.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxLength, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxLongEnergy</name>
    <filename>classKassiopeia_1_1KSTermMaxLongEnergy.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxLongEnergy, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxR</name>
    <filename>classKassiopeia_1_1KSTermMaxR.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxR, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxSteps</name>
    <filename>classKassiopeia_1_1KSTermMaxSteps.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxSteps, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxTime</name>
    <filename>classKassiopeia_1_1KSTermMaxTime.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxTime, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMaxZ</name>
    <filename>classKassiopeia_1_1KSTermMaxZ.html</filename>
    <base>KSComponentTemplate&lt; KSTermMaxZ, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMinDistance</name>
    <filename>classKassiopeia_1_1KSTermMinDistance.html</filename>
    <base>KSComponentTemplate&lt; KSTermMinDistance, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMinEnergy</name>
    <filename>classKassiopeia_1_1KSTermMinEnergy.html</filename>
    <base>KSComponentTemplate&lt; KSTermMinEnergy, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMinLongEnergy</name>
    <filename>classKassiopeia_1_1KSTermMinLongEnergy.html</filename>
    <base>KSComponentTemplate&lt; KSTermMinLongEnergy, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMinR</name>
    <filename>classKassiopeia_1_1KSTermMinR.html</filename>
    <base>KSComponentTemplate&lt; KSTermMinR, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermMinZ</name>
    <filename>classKassiopeia_1_1KSTermMinZ.html</filename>
    <base>KSComponentTemplate&lt; KSTermMinZ, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermOutput</name>
    <filename>classKassiopeia_1_1KSTermOutput.html</filename>
    <templarg>XValueType</templarg>
    <base>KSComponentTemplate&lt; KSTermOutput&lt; XValueType &gt;, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSTermOutputData</name>
    <filename>classkatrin_1_1KSTermOutputData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermSecondaries</name>
    <filename>classKassiopeia_1_1KSTermSecondaries.html</filename>
    <base>KSComponentTemplate&lt; KSTermSecondaries, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermStepsize</name>
    <filename>classKassiopeia_1_1KSTermStepsize.html</filename>
    <base>KSComponentTemplate&lt; KSTermStepsize, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTermTrapped</name>
    <filename>classKassiopeia_1_1KSTermTrapped.html</filename>
    <base>KSComponentTemplate&lt; KSTermTrapped, KSTerminator &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrack</name>
    <filename>classKassiopeia_1_1KSTrack.html</filename>
    <base>KSComponentTemplate&lt; KSTrack &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajAdiabaticDerivative</name>
    <filename>classKassiopeia_1_1KSTrajAdiabaticDerivative.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajAdiabaticError</name>
    <filename>classKassiopeia_1_1KSTrajAdiabaticError.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajAdiabaticParticle</name>
    <filename>classKassiopeia_1_1KSTrajAdiabaticParticle.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajAdiabaticSpinDerivative</name>
    <filename>classKassiopeia_1_1KSTrajAdiabaticSpinDerivative.html</filename>
    <base>KSMathArray&lt; 10 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajAdiabaticSpinError</name>
    <filename>classKassiopeia_1_1KSTrajAdiabaticSpinError.html</filename>
    <base>KSMathArray&lt; 10 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajAdiabaticSpinParticle</name>
    <filename>classKassiopeia_1_1KSTrajAdiabaticSpinParticle.html</filename>
    <base>KSMathArray&lt; 10 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlBChange</name>
    <filename>classKassiopeia_1_1KSTrajControlBChange.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlBChange &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlCyclotron</name>
    <filename>classKassiopeia_1_1KSTrajControlCyclotron.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlCyclotron &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlEnergy</name>
    <filename>classKassiopeia_1_1KSTrajControlEnergy.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlEnergy &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlLength</name>
    <filename>classKassiopeia_1_1KSTrajControlLength.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlLength &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlMagneticMoment</name>
    <filename>classKassiopeia_1_1KSTrajControlMagneticMoment.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlMagneticMoment &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlMDot</name>
    <filename>classKassiopeia_1_1KSTrajControlMDot.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlMDot &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlMomentumNumericalError</name>
    <filename>classKassiopeia_1_1KSTrajControlMomentumNumericalError.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlMomentumNumericalError &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlPositionNumericalError</name>
    <filename>classKassiopeia_1_1KSTrajControlPositionNumericalError.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlPositionNumericalError &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlSpinPrecession</name>
    <filename>classKassiopeia_1_1KSTrajControlSpinPrecession.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlSpinPrecession &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajControlTime</name>
    <filename>classKassiopeia_1_1KSTrajControlTime.html</filename>
    <base>KSComponentTemplate&lt; KSTrajControlTime &gt;</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
    <base>Kassiopeia::KSMathControl</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajectory</name>
    <filename>classKassiopeia_1_1KSTrajectory.html</filename>
    <base>KSComponentTemplate&lt; KSTrajectory &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajElectricDerivative</name>
    <filename>classKassiopeia_1_1KSTrajElectricDerivative.html</filename>
    <base>KSMathArray&lt; 5 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajElectricError</name>
    <filename>classKassiopeia_1_1KSTrajElectricError.html</filename>
    <base>KSMathArray&lt; 5 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajElectricParticle</name>
    <filename>classKassiopeia_1_1KSTrajElectricParticle.html</filename>
    <base>KSMathArray&lt; 5 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactDerivative</name>
    <filename>classKassiopeia_1_1KSTrajExactDerivative.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactError</name>
    <filename>classKassiopeia_1_1KSTrajExactError.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactParticle</name>
    <filename>classKassiopeia_1_1KSTrajExactParticle.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactSpinDerivative</name>
    <filename>classKassiopeia_1_1KSTrajExactSpinDerivative.html</filename>
    <base>KSMathArray&lt; 12 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactSpinError</name>
    <filename>classKassiopeia_1_1KSTrajExactSpinError.html</filename>
    <base>KSMathArray&lt; 12 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactSpinParticle</name>
    <filename>classKassiopeia_1_1KSTrajExactSpinParticle.html</filename>
    <base>KSMathArray&lt; 12 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactTrappedDerivative</name>
    <filename>classKassiopeia_1_1KSTrajExactTrappedDerivative.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactTrappedError</name>
    <filename>classKassiopeia_1_1KSTrajExactTrappedError.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajExactTrappedParticle</name>
    <filename>classKassiopeia_1_1KSTrajExactTrappedParticle.html</filename>
    <base>KSMathArray&lt; 8 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRK54</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRK54.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRK54 &gt;</base>
    <base>KSMathRKF54&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRKF54&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRKF54&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRKF54&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRKF54&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRKF54&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRK65</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRK65.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRK65 &gt;</base>
    <base>KSMathRK65&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRK65&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRK65&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRK65&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRK65&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRK65&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRK8</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRK8.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRK8 &gt;</base>
    <base>KSMathRK8&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRK8&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRK8&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRK8&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRK8&lt; KSTrajExactTrappedSystem &gt;</base>
    <base>KSMathRK8&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRK8&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRK86</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRK86.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRK86 &gt;</base>
    <base>KSMathRK86&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRK86&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRK86&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRK86&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRK86&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRK86&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRK87</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRK87.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRK87 &gt;</base>
    <base>KSMathRK87&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRK87&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRK87&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRK87&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRK87&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRK87&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRKDP54</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRKDP54.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRKDP54 &gt;</base>
    <base>KSMathRKDP54&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRKDP54&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRKDP54&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRKDP54&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRKDP54&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRKDP54&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorRKDP853</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorRKDP853.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorRKDP853 &gt;</base>
    <base>KSMathRKDP853&lt; KSTrajExactSystem &gt;</base>
    <base>KSMathRKDP853&lt; KSTrajExactSpinSystem &gt;</base>
    <base>KSMathRKDP853&lt; KSTrajAdiabaticSpinSystem &gt;</base>
    <base>KSMathRKDP853&lt; KSTrajAdiabaticSystem &gt;</base>
    <base>KSMathRKDP853&lt; KSTrajElectricSystem &gt;</base>
    <base>KSMathRKDP853&lt; KSTrajMagneticSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajIntegratorSym4</name>
    <filename>classKassiopeia_1_1KSTrajIntegratorSym4.html</filename>
    <base>KSComponentTemplate&lt; KSTrajIntegratorSym4 &gt;</base>
    <base>KSMathSym4&lt; KSTrajExactTrappedSystem &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajInterpolatorContinuousRungeKutta</name>
    <filename>classKassiopeia_1_1KSTrajInterpolatorContinuousRungeKutta.html</filename>
    <base>KSComponentTemplate&lt; KSTrajInterpolatorContinuousRungeKutta &gt;</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajInterpolatorFast</name>
    <filename>classKassiopeia_1_1KSTrajInterpolatorFast.html</filename>
    <base>KSComponentTemplate&lt; KSTrajInterpolatorFast &gt;</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajInterpolatorHermite</name>
    <filename>classKassiopeia_1_1KSTrajInterpolatorHermite.html</filename>
    <base>KSComponentTemplate&lt; KSTrajInterpolatorHermite &gt;</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
    <base>Kassiopeia::KSMathInterpolator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajMagneticDerivative</name>
    <filename>classKassiopeia_1_1KSTrajMagneticDerivative.html</filename>
    <base>KSMathArray&lt; 5 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajMagneticError</name>
    <filename>classKassiopeia_1_1KSTrajMagneticError.html</filename>
    <base>KSMathArray&lt; 5 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajMagneticParticle</name>
    <filename>classKassiopeia_1_1KSTrajMagneticParticle.html</filename>
    <base>KSMathArray&lt; 5 &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTermConstantForcePropagation</name>
    <filename>classKassiopeia_1_1KSTrajTermConstantForcePropagation.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTermConstantForcePropagation &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTermDrift</name>
    <filename>classKassiopeia_1_1KSTrajTermDrift.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTermDrift &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTermGravity</name>
    <filename>classKassiopeia_1_1KSTrajTermGravity.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTermGravity &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTermGyration</name>
    <filename>classKassiopeia_1_1KSTrajTermGyration.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTermGyration &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTermPropagation</name>
    <filename>classKassiopeia_1_1KSTrajTermPropagation.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTermPropagation &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTermSynchrotron</name>
    <filename>classKassiopeia_1_1KSTrajTermSynchrotron.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTermSynchrotron &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryAdiabatic</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryAdiabatic.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryAdiabatic, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryAdiabaticSpin</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryAdiabaticSpin.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryAdiabaticSpin, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryElectric</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryElectric.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryElectric, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryExact</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryExact.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryExact, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryExactSpin</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryExactSpin.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryExactSpin, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryExactTrapped</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryExactTrapped.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryExactTrapped, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryLinear</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryLinear.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryLinear, KSTrajectory &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSTrajTrajectoryMagnetic</name>
    <filename>classKassiopeia_1_1KSTrajTrajectoryMagnetic.html</filename>
    <base>KSComponentTemplate&lt; KSTrajTrajectoryMagnetic, KSTrajectory &gt;</base>
    <base>Kassiopeia::KSMathDifferentiator</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSVTKTrackPainter</name>
    <filename>classKassiopeia_1_1KSVTKTrackPainter.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSVTKTrackTerminatorPainter</name>
    <filename>classKassiopeia_1_1KSVTKTrackTerminatorPainter.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteASCII</name>
    <filename>classKassiopeia_1_1KSWriteASCII.html</filename>
    <base>KSComponentTemplate&lt; KSWriteASCII, KSWriter &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriter</name>
    <filename>classKassiopeia_1_1KSWriter.html</filename>
    <base>KSComponentTemplate&lt; KSWriter &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteROOT</name>
    <filename>classKassiopeia_1_1KSWriteROOT.html</filename>
    <base>KSComponentTemplate&lt; KSWriteROOT, KSWriter &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteROOTCondition</name>
    <filename>classKassiopeia_1_1KSWriteROOTCondition.html</filename>
    <base>KSComponentTemplate&lt; KSWriteROOTCondition &gt;</base>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteROOTConditionOutput</name>
    <filename>classKassiopeia_1_1KSWriteROOTConditionOutput.html</filename>
    <templarg>XValueType</templarg>
    <base>KSComponentTemplate&lt; KSWriteROOTConditionOutput&lt; XValueType &gt;, KSWriteROOTCondition &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSWriteROOTConditionOutputData</name>
    <filename>classkatrin_1_1KSWriteROOTConditionOutputData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteROOTConditionStep</name>
    <filename>classKassiopeia_1_1KSWriteROOTConditionStep.html</filename>
    <templarg>XValueType</templarg>
    <base>KSComponentTemplate&lt; KSWriteROOTConditionStep&lt; XValueType &gt;, KSWriteROOTCondition &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSWriteROOTConditionStepData</name>
    <filename>classkatrin_1_1KSWriteROOTConditionStepData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteROOTConditionTerminator</name>
    <filename>classKassiopeia_1_1KSWriteROOTConditionTerminator.html</filename>
    <base>KSComponentTemplate&lt; KSWriteROOTConditionTerminator, KSWriteROOTCondition &gt;</base>
  </compound>
  <compound kind="class">
    <name>katrin::KSWriteROOTConditionTerminatorData</name>
    <filename>classkatrin_1_1KSWriteROOTConditionTerminatorData.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::KSWriteVTK</name>
    <filename>classKassiopeia_1_1KSWriteVTK.html</filename>
    <base>KSComponentTemplate&lt; KSWriteVTK, KSWriter &gt;</base>
  </compound>
  <compound kind="struct">
    <name>Kassiopeia::KSGenRelaxation::line_struct</name>
    <filename>structKassiopeia_1_1KSGenRelaxation_1_1line__struct.html</filename>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::QuadGaussLegendre</name>
    <filename>classKassiopeia_1_1QuadGaussLegendre.html</filename>
    <member kind="function" static="yes">
      <type>static double</type>
      <name>IntegrateH</name>
      <anchorfile>classKassiopeia_1_1QuadGaussLegendre.html</anchorfile>
      <anchor>ac5aed5f75fb5e03157592e6eff247a00</anchor>
      <arglist>(XFunctorType &amp;f, double step1, double xlimit, double tol, int N)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>Kassiopeia::RydbergCalculator</name>
    <filename>classKassiopeia_1_1RydbergCalculator.html</filename>
    <member kind="function">
      <type>void</type>
      <name>BBRTransitionGenerator</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>a7f9e336ec7d145c8c7637c8d548e0ba6</anchor>
      <arglist>(double T, int n, int l, double &amp;PBBRtotal, int &amp;np, int &amp;lp)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>HypergeometricFHoangBinh</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>aa3aa53409e39f6216b6e70af4e1f7672</anchor>
      <arglist>(int a, int b, int c, double x, double &amp;E)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>PBBR</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>a19eb33940f17faeb874d16a49476a30f</anchor>
      <arglist>(double T, int n, int l, int np, int sign)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>PBBRdecay</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>ac3f798875d4747d83b5e97d9b0c39ad1</anchor>
      <arglist>(double T, int n, int l)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>PBBRexcitation</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>a8d8f00d09fbb08907c4abaeda2a07411</anchor>
      <arglist>(double T, int n, int l, int npmax)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>PBBRionization</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>ac7dc4816dfe29bca5455a6059ba9be43</anchor>
      <arglist>(double T, int n, int l, double step1factor, double tol, int Ninteg)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Psp</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>adb60650479e799bbda82c37ba9a6f9c2</anchor>
      <arglist>(int n, int l, int np, int sign)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>Pspsum</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>a120945151e87292fde6cb0945d2c2d44</anchor>
      <arglist>(int n, int l)</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>RadInt2Gordon</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>af43dc762eda62c0fcdc231987bd368be</anchor>
      <arglist>(int n, int l, int np, int sign)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>SpontaneousEmissionGenerator</name>
      <anchorfile>classKassiopeia_1_1RydbergCalculator.html</anchorfile>
      <anchor>aa21cd0cce1e59560d92d5ae3c330b087</anchor>
      <arglist>(int n, int l, double &amp;Psptotal, int &amp;np, int &amp;lp)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>XFirstParentType</name>
    <filename>classXFirstParentType.html</filename>
  </compound>
</tagfile>
