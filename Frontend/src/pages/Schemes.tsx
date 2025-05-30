
import { useState } from "react";
import Footer from "../components/Footer";
import SchemeCard, { SchemeType } from "../components/SchemeCard";
import { useLanguage } from "../contexts/LanguageContext";

// Sample schemes data with translations
const schemesData: {
  id: number;
  title: { en: string; mr: string };
  ministry: { en: string; mr: string };
  eligibility: { en: string; mr: string };
  benefits: { en: string; mr: string };
  applicationProcess: { en: string; mr: string };
  lastDate?: string;
  website: string;
}[] = [
  {
    id: 1,
    title: {
      en: "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
      mr: "प्रधानमंत्री किसान सम्मान निधि (पीएम-किसान)"
    },
    ministry: {
      en: "Ministry of Agriculture and Farmers Welfare",
      mr: "कृषि आणि शेतकरी कल्याण मंत्रालय"
    },
    eligibility: {
      en: "All landholding farmer families with cultivable land, subject to certain exclusions.",
      mr: "मशागतीयोग्य जमिनीसह सर्व भूमिधारक शेतकरी कुटुंबे, काही अपवर्जनांच्या अधीन."
    },
    benefits: {
      en: "Financial benefit of Rs. 6,000 per year in three equal installments of Rs. 2,000 each, transferred directly into farmers' bank accounts every four months.",
      mr: "शेतकऱ्यांच्या बँक खात्यांमध्ये थेट दर चार महिन्यांनी रु. २,००० च्या तीन समान हप्त्यांमध्ये वर्षाकाठी रु. ६,००० चा आर्थिक लाभ."
    },
    applicationProcess: {
      en: "Registration through local revenue officials or Common Service Centers. Required documents include Aadhaar, bank details, and land documents.",
      mr: "स्थानिक महसूल अधिकारी किंवा सामाईक सेवा केंद्रांद्वारे नोंदणी. आवश्यक दस्तऐवजांमध्ये आधार, बँक तपशील आणि जमिनीचे दस्तऐवज यांचा समावेश आहे."
    },
    website: "https://pmkisan.gov.in/"
  },
  {
    id: 2,
    title: {
      en: "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
      mr: "प्रधानमंत्री फसल बीमा योजना (पीएमएफबीवाय)"
    },
    ministry: {
      en: "Ministry of Agriculture and Farmers Welfare",
      mr: "कृषि आणि शेतकरी कल्याण मंत्रालय"
    },
    eligibility: {
      en: "All farmers, including sharecroppers and tenant farmers, growing notified crops in notified areas are eligible.",
      mr: "अधिसूचित क्षेत्रांमध्ये अधिसूचित पिके वाढवणारे सर्व शेतकरी, शेतमजूर आणि भाडेकरू शेतकरी यांचा समावेश."
    },
    benefits: {
      en: "Comprehensive risk coverage for pre-sowing to post-harvest losses due to natural calamities, pests, and diseases at low premium rates.",
      mr: "कमी प्रीमियम दरांसह नैसर्गिक आपत्ती, किटक आणि रोगांमुळे होणाऱ्या पेरणीपूर्व ते कापणीनंतरच्या नुकसानीसाठी सर्वसमावेशक जोखीम कवच."
    },
    applicationProcess: {
      en: "Registration through bank branches, Common Service Centers, or online. Farmers need to provide land details, bank account, and Aadhaar number.",
      mr: "बँक शाखा, सामाईक सेवा केंद्रे किंवा ऑनलाईन नोंदणी. शेतकऱ्यांना जमीन तपशील, बँक खाते आणि आधार क्रमांक प्रदान करणे आवश्यक आहे."
    },
   
    website: "https://pmfby.gov.in/"
  },
  {
    id: 3,
    title: {
      en: "Soil Health Card Scheme",
      mr: "मृदा आरोग्य कार्ड योजना"
    },
    ministry: {
      en: "Department of Agriculture and Cooperation",
      mr: "कृषि आणि सहकार विभाग"
    },
    eligibility: {
      en: "All farmers across India.",
      mr: "संपूर्ण भारतातील सर्व शेतकरी."
    },
    benefits: {
      en: "Provides Soil Health Card with crop-wise recommendations of nutrients and fertilizers to help farmers improve productivity through proper use of inputs.",
      mr: "इनपुटचा योग्य वापर करून शेतकऱ्यांना उत्पादकता सुधारण्यास मदत करण्यासाठी पिकांनुसार पोषक तत्वे आणि खतांच्या शिफारसींसह मृदा आरोग्य कार्ड प्रदान करते."
    },
    applicationProcess: {
      en: "Apply at the nearest agriculture department office or Krishi Vigyan Kendra. Soil samples from the farmer's field will be collected for testing.",
      mr: "जवळच्या कृषि विभागाच्या कार्यालयात किंवा कृषी विज्ञान केंद्रात अर्ज करा. शेतकऱ्याच्या शेतातून मातीचे नमुने चाचणीसाठी गोळा केले जातील."
    },
    website: "https://soilhealth.dac.gov.in/"
  },
  {
    id: 4,
    title: {
      en: "Kisan Credit Card (KCC)",
      mr: "किसान क्रेडिट कार्ड (केसीसी)"
    },
    ministry: {
      en: "Ministry of Finance",
      mr: "वित्त मंत्रालय"
    },
    eligibility: {
      en: "All farmers, tenant farmers, sharecroppers, and individuals engaged in animal husbandry and fisheries.",
      mr: "सर्व शेतकरी, भाडेकरू शेतकरी, शेतमजूर आणि पशुसंवर्धन आणि मत्स्यपालनामध्ये गुंतलेले व्यक्ती."
    },
    benefits: {
      en: "Provides access to short-term credit needs for cultivation of crops, post-harvest expenses, and working capital for animal husbandry and fisheries.",
      mr: "पिकांच्या लागवडीसाठी अल्पकालीन क्रेडिट गरजा, कापणी नंतरचे खर्च आणि पशुसंवर्धन आणि मत्स्यपालनासाठी कार्यशील भांडवल यांसाठी प्रवेश प्रदान करते."
    },
    applicationProcess: {
      en: "Apply at any bank branch with land documents, ID proof, and passport-sized photographs.",
      mr: "जमीन दस्तऐवज, ओळखपत्र आणि पासपोर्ट आकाराचे फोटो यांसह कोणत्याही बँक शाखेत अर्ज करा."
    },
    website: "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1607417"
  },
  {
    id: 5,
    title: {
      en: "Micro Irrigation Fund (MIF)",
      mr: "सूक्ष्म सिंचन निधी (एमआयएफ)"
    },
    ministry: {
      en: "Ministry of Agriculture and Farmers Welfare",
      mr: "कृषि आणि शेतकरी कल्याण मंत्रालय"
    },
    eligibility: {
      en: "State Governments implementing micro-irrigation projects.",
      mr: "सूक्ष्म सिंचन प्रकल्प राबवणाऱ्या राज्य सरकारांसाठी."
    },
    benefits: {
      en: "Facilitates access to financing to expand coverage of micro-irrigation systems by providing interest subsidy on loans.",
      mr: "कर्जावरील व्याज सबसिडी प्रदान करून सूक्ष्म सिंचन प्रणालींचे कव्हरेज विस्तारण्यासाठी वित्तपुरवठ्यास प्रवेश सुलभ करते."
    },
    applicationProcess: {
      en: "State Governments need to submit proposals to NABARD. Individual farmers benefit through state-implemented micro-irrigation projects.",
      mr: "राज्य सरकारांनी नाबार्डकडे प्रस्ताव सादर करणे आवश्यक आहे. व्यक्तिगत शेतकरी राज्य-कार्यान्वित सूक्ष्म सिंचन प्रकल्पांद्वारे लाभ मिळवतात."
    },
    website: "https://pmksy.gov.in/"
  },
  {
    id: 6,
    title: {
      en: "Agriculture Infrastructure Fund",
      mr: "कृषी पायाभूत सुविधा निधी"
    },
    ministry: {
      en: "Ministry of Agriculture and Farmers Welfare",
      mr: "कृषि आणि शेतकरी कल्याण मंत्रालय"
    },
    eligibility: {
      en: "Farmers, FPOs, PACS, Marketing Cooperative Societies, Self Help Groups, Joint Liability Groups, Multipurpose Cooperative Societies, Agri-entrepreneurs, Start-ups, and Central/State agency or Local Body sponsored Public-Private Partnership Projects.",
      mr: "शेतकरी, एफपीओ, पीएसीएस, मार्केटिंग सहकारी संस्था, स्वयंसहायता गट, संयुक्त देयता गट, बहुउद्देशीय सहकारी संस्था, कृषि उद्योजक, स्टार्ट-अप आणि केंद्र/राज्य संस्था किंवा स्थानिक संस्थेद्वारे प्रायोजित सार्वजनिक-खासगी भागीदारी प्रकल्प."
    },
    benefits: {
      en: "Rs. 1 lakh crore financing facility with 3% annual interest subvention and credit guarantee coverage for loans up to Rs. 2 crore.",
      mr: "वार्षिक ३% व्याज सबव्हेंशन आणि रु. २ कोटी पर्यंतच्या कर्जांसाठी क्रेडिट हमी कव्हरेजसह रु. १ लाख कोटी वित्तपुरवठा सुविधा."
    },
    applicationProcess: {
      en: "Apply through participating banks or online via the AIF Portal with a detailed project report and necessary documentation.",
      mr: "सविस्तर प्रकल्प अहवाल आणि आवश्यक कागदपत्रांसह सहभागी बँकांद्वारे किंवा एआयएफ पोर्टलद्वारे ऑनलाइन अर्ज करा."
    },
    lastDate: "Active until 2029",
    website: "https://agriinfra.dac.gov.in/"
  }
];

const Schemes = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const { language, t } = useLanguage();
  
  // Transform schemes data based on current language
  const transformedSchemesData: SchemeType[] = schemesData.map(scheme => ({
    id: scheme.id,
    title: scheme.title[language],
    ministry: scheme.ministry[language],
    eligibility: scheme.eligibility[language],
    benefits: scheme.benefits[language],
    applicationProcess: scheme.applicationProcess[language],
    lastDate: scheme.lastDate,
    website: scheme.website
  }));
  
  const filteredSchemes = transformedSchemesData.filter((scheme) =>
    scheme.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    scheme.ministry.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-grow">
        {/* Header */}
        <div className="bg-farm-green text-white py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-4xl font-bold mb-4">{t("schemes.title")}</h1>
            <p className="text-xl max-w-2xl">
              {t("schemes.description")}
            </p>
          </div>
        </div>
        
        {/* Search */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="bg-white shadow-md rounded-lg p-6 mb-8">
            <label htmlFor="scheme-search" className="block text-sm font-medium text-gray-700 mb-1">
              {t("schemes.search")}
            </label>
            <input
              type="text"
              id="scheme-search"
              placeholder={t("schemes.searchPlaceholder")}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-farm-green focus:border-transparent"
            />
          </div>
          
          {/* Scheme Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
            {filteredSchemes.length > 0 ? (
              filteredSchemes.map((scheme) => (
                <SchemeCard key={scheme.id} scheme={scheme} />
              ))
            ) : (
              <div className="col-span-full text-center py-8">
                <p className="text-gray-500 text-lg">{t("schemes.noResults")}</p>
              </div>
            )}
          </div>
          
          {/* Info Section */}
          <div className="bg-farm-green-light/10 rounded-lg p-6 mb-12">
            <h2 className="text-2xl font-bold text-farm-green mb-4">{t("schemes.apply.title")}</h2>
            <div className="space-y-4">
              <p>
                {t("schemes.apply.documents")}
              </p>
              <ul className="list-disc ml-6 space-y-2">
                <li>{t("schemes.apply.doc1")}</li>
                <li>{t("schemes.apply.doc2")}</li>
                <li>{t("schemes.apply.doc3")}</li>
                <li>{t("schemes.apply.doc4")}</li>
                <li>{t("schemes.apply.doc5")}</li>
              </ul>
              <p>
                {t("schemes.apply.methods")}
              </p>
              <ul className="list-disc ml-6 space-y-2">
                <li>{t("schemes.apply.method1")}</li>
                <li>{t("schemes.apply.method2")}</li>
                <li>{t("schemes.apply.method3")}</li>
                <li>{t("schemes.apply.method4")}</li>
              </ul>
              <p className="text-sm italic">
                {t("schemes.apply.note")}
              </p>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default Schemes;
