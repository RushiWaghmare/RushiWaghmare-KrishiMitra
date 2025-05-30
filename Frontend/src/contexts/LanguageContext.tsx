
import React, { createContext, useState, useContext, ReactNode } from 'react';

type LanguageContextType = {
  language: 'en' | 'mr';
  setLanguage: (language: 'en' | 'mr') => void;
  t: (key: string) => string;
};

const translations = {
  en: {
    // Home page
    "hero.title": "Your Crop Companion",
    "hero.description": "Access crop information, government schemes, and seasonal advisories to improve your farm's productivity.",
    "hero.button.kisanBazaar": "Marketplace for Farmers",
    "hero.button.viewSchemes": "View Schemes",
    
    // Navigation
    "nav.home": "Home",
    "nav.cropDatabase": "Crop Database",
    "nav.govtSchemes": "Govt Schemes",
    "nav.diseaseDetection": "Disease Detection",
    "nav.cropMonitoring": "Crop Monitoring",
    "nav.fertilizers": "Fertilizers",
    "nav.cropPricePrediction": "Crop Price Prediction",
    
    // Features section
    "features.heading": "How KrishiMitra Helps You",
    "features.cropInfo.title": "Crop Information",
    "features.cropInfo.description": "Access detailed information about various crops, including growing seasons, water requirements, and expected yields.",
    "features.cropInfo.link": "Explore Crops →",
    "features.seasonalGuides.title": "Seasonal Guides",
    "features.seasonalGuides.description": "Know what to plant and when with our seasonal planting guides for Kharif, Rabi, and Zaid seasons.",
    "features.seasonalGuides.link": "View Guide →",
    "features.govtSchemes.title": "Government Schemes",
    "features.govtSchemes.description": "Stay updated on the latest government schemes and subsidies available for farmers.",
    "features.govtSchemes.link": "View Schemes →",
    
    // CTA section
    "cta.title": "Ready to Improve Your Farming Practice?",
    "cta.description": "Explore our comprehensive database of crops and discover government schemes that can help you increase productivity and income.",
    "cta.button.browseCrops": "Browse Crop Database",
    "cta.button.viewSchemes": "View Government Schemes",
    
    // Footer
    "footer.about": "Empowering farmers with knowledge and resources to improve agricultural productivity.",
    "footer.quickLinks": "Quick Links",
    "footer.home": "Home",
    "footer.cropDatabase": "Crop Database",
    "footer.govtSchemes": "Government Schemes",
    "footer.contact": "Contact",
    "footer.contactText": "For any queries or suggestions, please contact us at:",
    "footer.rights": "All rights reserved.",
    
    // 404 page
    "notFound.title": "404",
    "notFound.description": "Oops! Page not found",
    "notFound.link": "Return to Home",
    
    // Crops page
    "crops.title": "Crop Database",
    "crops.description": "Explore detailed information about various crops, including growing seasons, water requirements, soil preferences, and expected yields.",
    "crops.search": "Search Crops",
    "crops.searchPlaceholder": "Search by name or scientific name...",
    "crops.filterSeason": "Filter by Season",
    "crops.allSeasons": "All Seasons",
    "crops.kharif": "Kharif (Monsoon)",
    "crops.rabi": "Rabi (Winter)",
    "crops.zaid": "Zaid (Summer)",
    "crops.noResults": "No crops matching your search criteria. Try adjusting your filters.",
    
    // Crop card
    "crop.season": "Season:",
    "crop.water": "Water:",
    "crop.soil": "Soil:",
    "crop.yield": "Yield:",
    "crop.showMore": "Show More",
    "crop.showLess": "Show Less",
    
    // Seasonal Guide
    "seasonal.title": "Seasonal Planting Guide",
    "seasonal.months": "Months:",
    "seasonal.recommendedCrops": "Recommended Crops:",
    "seasonal.kharif": "Kharif (Monsoon)",
    "seasonal.rabi": "Rabi (Winter)",
    "seasonal.zaid": "Zaid (Summer)",
    "seasonal.kharif.description": "The Kharif crops are sown at the beginning of the monsoon season and harvested at the end of the monsoon season.",
    "seasonal.rabi.description": "The Rabi crops are sown at the beginning of winter and harvested in spring.",
    "seasonal.zaid.description": "The Zaid crops are grown in the short duration between Rabi and Kharif seasons.",
    
    // Weather
    "weather.loading": "Loading weather data...",
    "weather.error": "Failed to load weather data",
    "weather.fallback": "Showing default weather data instead.",
    "weather.in": "Weather in",
    "weather.humidity": "Humidity:",
    "weather.rainfall": "Rainfall:",
    "weather.forecast": "5-Day Forecast",
    "weather.unavailable": "Forecast unavailable",
    
    // Schemes
    "schemes.title": "Government Schemes",
    "schemes.description": "Stay updated on the latest government schemes, subsidies, and initiatives designed to support farmers and improve agricultural productivity.",
    "schemes.search": "Search Schemes",
    "schemes.searchPlaceholder": "Search by name or ministry...",
    "schemes.noResults": "No schemes matching your search criteria. Try adjusting your search term.",
    "schemes.apply.title": "How to Apply for Schemes",
    "schemes.apply.documents": "Government agricultural schemes typically require the following documents:",
    "schemes.apply.doc1": "Aadhaar Card",
    "schemes.apply.doc2": "Land Records/Ownership Documents",
    "schemes.apply.doc3": "Bank Account Details",
    "schemes.apply.doc4": "Passport-sized Photographs",
    "schemes.apply.doc5": "Mobile Number (for receiving updates)",
    "schemes.apply.methods": "For most schemes, you can apply through:",
    "schemes.apply.method1": "Online portals of respective ministries",
    "schemes.apply.method2": "Common Service Centers (CSCs) in your area",
    "schemes.apply.method3": "Local agriculture offices or Krishi Vigyan Kendras",
    "schemes.apply.method4": "Bank branches (for credit-related schemes)",
    "schemes.apply.note": "Note: The application process and requirements may vary for each scheme. Please refer to the official websites for the most accurate and up-to-date information.",
    
    // Scheme card
    "scheme.ministry": "Ministry:",
    "scheme.eligibility": "Eligibility:",
    "scheme.benefits": "Benefits:",
    "scheme.applicationProcess": "Application Process:",
    "scheme.lastDate": "Last Date:",
    "scheme.website": "Visit Official Website",
    
    // Fertilizers page
    "fertilizers.title": "Fertilizers & Pesticides",
    "fertilizers.description": "Find the recommended fertilizers and pesticides for different crops to improve yield and protect your plants.",
    "fertilizers.search": "Search for a crop",
    "fertilizers.searchPlaceholder": "Enter crop name...",
    "fertilizers.noResults": "No crops found matching your search. Try a different crop name.",
    "fertilizers.recommended": "Recommended Products",
    "fertilizers.fertilizersSection": "Fertilizers",
    "fertilizers.pesticidesSection": "Pesticides",
    "fertilizers.buyNow": "Buy Now",
    "fertilizers.type": "Type:",
    "fertilizers.usage": "Usage:",
    "fertilizers.benefits": "Benefits:",
    "fertilizers.application": "Application:",
    "fertilizers.price": "Price:",
    "fertilizers.loadMore": "Load More",

    //Disease Detection
    "disease.title": "Disease Detection",
    "disease.description" : "Upload images of your plants to quickly identify diseases, pests, and nutritional deficiencies. Get detailed information and treatment recommendations.",
    "disease.heading" : "Plant Disease Detection Tool",
    "disease.imageUpload" : "Upload Plant Image",
    "disease.drag" : "Click to select or drag an image",
    "disease.span" : "Support formats: JPG, PNG, WEBP",
    "disease.toggle": "Language",
    "disease.imagePreview" : "Image preview will appear here",
    "disease.result" : "Analysis Results",
    "disease.button" : "Analyze Plant",
    "disease.loading" : "Analyzing",

    //Crop Price Prediction
    "cropPrice.title" : "Indian Crop Price Prediction",
    "cropPrice.cropName" : "Crop Name",
    "cropPrice.region" : "Region",
    "cropPrice.placeholder" : "Enter Crop Name",
    "cropPrice.basePrice" : "Base Price (₹/quintal)",
    "cropPrice.button" : "Update Prediction",
    "cropPrice.forecastHeading" : "Price Trends & Predictions for next 3 months",
    "cropPrice.forecast": "Forecast",
    "cropPrice.market" : "Market Insights",
    "cropPrice.result" : "Based on our ARIMA model analysis, the crop prices in the selected region are expected to change over the next three months. According to the model, by the end of this period, the price per quintal will change compared to the current prices.",


  },
  mr: {
    // Home page
    "hero.title": "तुमचा पीक साथीदार",
    "hero.description": "आपल्या शेतीची उत्पादकता सुधारण्यासाठी पीक माहिती, सरकारी योजना आणि हंगामी सल्ला प्राप्त करा.",
    "hero.button.kisanBazaar": "शेतमाल बाजार",
    "hero.button.viewSchemes": "योजना पहा",
    
    // Navigation
    "nav.home": "मुख्यपृष्ठ",
    "nav.cropDatabase": "पीक डेटाबेस",
    "nav.govtSchemes": "सरकारी योजना",
    "nav.diseaseDetection": "रोग शोध",
    "nav.cropMonitoring": "पीक निरीक्षण",
    "nav.fertilizers": "खते आणि कीटकनाशके",
    "nav.cropPricePrediction": "पीक दराचा अंदाज",
    
    // Features section
    "features.heading": "फार्मइन्फो तुम्हाला कसे मदत करते",
    "features.cropInfo.title": "पीक माहिती",
    "features.cropInfo.description": "विविध पिकांबद्दल तपशीलवार माहिती मिळवा, ज्यामध्ये वाढीचे हंगाम, पाण्याची आवश्यकता आणि अपेक्षित उत्पन्न यांचा समावेश आहे.",
    "features.cropInfo.link": "पिके एक्स्प्लोर करा →",
    "features.seasonalGuides.title": "हंगामी मार्गदर्शक",
    "features.seasonalGuides.description": "खरीप, रबी आणि झाईड हंगामांसाठी आमच्या हंगामी लागवड मार्गदर्शकांसह काय लावायचे आणि कधी हे जाणून घ्या.",
    "features.seasonalGuides.link": "मार्गदर्शक पहा →",
    "features.govtSchemes.title": "सरकारी योजना",
    "features.govtSchemes.description": "शेतकऱ्यांसाठी उपलब्ध नवीनतम सरकारी योजना आणि अनुदानांबद्दल अद्ययावत रहा.",
    "features.govtSchemes.link": "योजना पहा →",
    
    // CTA section
    "cta.title": "तुमची शेती पद्धती सुधारण्यास तयार आहात?",
    "cta.description": "आमचा व्यापक पीक डेटाबेस एक्स्प्लोर करा आणि सरकारी योजना शोधा ज्या तुम्हाला उत्पादकता आणि उत्पन्न वाढविण्यास मदत करू शकतात.",
    "cta.button.browseCrops": "पीक डेटाबेस ब्राउझ करा",
    "cta.button.viewSchemes": "सरकारी योजना पहा",
    
    // Footer
    "footer.about": "शेतकऱ्यांना कृषी उत्पादकता सुधारण्यासाठी ज्ञान आणि संसाधनांसह सशक्त बनवणे.",
    "footer.quickLinks": "द्रुत दुवे",
    "footer.home": "मुख्यपृष्ठ",
    "footer.cropDatabase": "पीक डेटाबेस",
    "footer.govtSchemes": "सरकारी योजना",
    "footer.contact": "संपर्क",
    "footer.contactText": "कोणत्याही प्रश्नांसाठी किंवा सूचनांसाठी, कृपया आमच्याशी संपर्क साधा:",
    "footer.rights": "सर्व हक्क राखीव.",
    
    // 404 page
    "notFound.title": "४०४",
    "notFound.description": "अरेरे! पृष्ठ सापडले नाही",
    "notFound.link": "मुख्यपृष्ठावर परत जा",
    
    // Crops page
    "crops.title": "पीक डेटाबेस",
    "crops.description": "विविध पिकांबद्दल तपशीलवार माहिती एक्स्प्लोर करा, ज्यामध्ये वाढीचे हंगाम, पाण्याची आवश्यकता, मातीची प्राधान्ये आणि अपेक्षित उत्पन्न यांचा समावेश आहे.",
    "crops.search": "पिके शोधा",
    "crops.searchPlaceholder": "नाव किंवा वैज्ञानिक नावाने शोधा...",
    "crops.filterSeason": "हंगामानुसार फिल्टर करा",
    "crops.allSeasons": "सर्व हंगाम",
    "crops.kharif": "खरीप (पावसाळा)",
    "crops.rabi": "रबी (हिवाळा)",
    "crops.zaid": "झाईड (उन्हाळा)",
    "crops.noResults": "तुमच्या शोध निकषांशी जुळणारी पिके नाहीत. तुमचे फिल्टर समायोजित करण्याचा प्रयत्न करा.",
    
    // Crop card
    "crop.season": "हंगाम:",
    "crop.water": "पाणी:",
    "crop.soil": "माती:",
    "crop.yield": "उत्पादन:",
    "crop.showMore": "अधिक दाखवा",
    "crop.showLess": "कमी दाखवा",
    
    // Seasonal Guide
    "seasonal.title": "हंगामी लागवड मार्गदर्शक",
    "seasonal.months": "महिने:",
    "seasonal.recommendedCrops": "शिफारस केलेली पिके:",
    "seasonal.kharif": "खरीप (पावसाळा)",
    "seasonal.rabi": "रबी (हिवाळा)",
    "seasonal.zaid": "झाईड (उन्हाळा)",
    "seasonal.kharif.description": "खरीप पिके पावसाळ्याच्या सुरुवातीला पेरली जातात आणि पावसाळ्याच्या शेवटी कापणी केली जाते.",
    "seasonal.rabi.description": "रबी पिके हिवाळ्याच्या सुरुवातीला पेरली जातात आणि वसंतात कापणी केली जाते.",
    "seasonal.zaid.description": "झाईड पिके रबी आणि खरीप हंगामांदरम्यान अल्प कालावधीत वाढवली जातात.",
    
    // Weather
    "weather.loading": "हवामान डेटा लोड करत आहे...",
    "weather.error": "हवामान डेटा लोड करण्यात अयशस्वी",
    "weather.fallback": "त्याऐवजी डीफॉल्ट हवामान डेटा दाखवत आहे.",
    "weather.in": "येथे हवामान",
    "weather.humidity": "आर्द्रता:",
    "weather.rainfall": "पाऊस:",
    "weather.forecast": "५-दिवसांचा अंदाज",
    "weather.unavailable": "अंदाज उपलब्ध नाही",
    
    // Schemes
    "schemes.title": "सरकारी योजना",
    "schemes.description": "शेतकऱ्यांना पाठिंबा देण्यासाठी आणि कृषी उत्पादकता सुधारण्यासाठी डिझाइन केलेल्या नवीनतम सरकारी योजना, अनुदाने आणि उपक्रमांबद्दल अद्ययावत रहा.",
    "schemes.search": "योजना शोधा",
    "schemes.searchPlaceholder": "नाव किंवा मंत्रालयाद्वारे शोधा...",
    "schemes.noResults": "तुमच्या शोध मापदंडांशी जुळणाऱ्या योजना नाहीत. तुमचा शोध शब्द समायोजित करण्याचा प्रयत्न करा.",
    "schemes.apply.title": "योजनांसाठी अर्ज कसा करावा",
    "schemes.apply.documents": "सरकारी कृषी योजनांसाठी सामान्यत: खालील दस्तऐवजांची आवश्यकता असते:",
    "schemes.apply.doc1": "आधार कार्ड",
    "schemes.apply.doc2": "जमिनीचे रेकॉर्ड/मालकी दस्तऐवज",
    "schemes.apply.doc3": "बँक खाते तपशील",
    "schemes.apply.doc4": "पासपोर्ट आकाराचे फोटो",
    "schemes.apply.doc5": "मोबाईल नंबर (अपडेट्स प्राप्त करण्यासाठी)",
    "schemes.apply.methods": "बहुतेक योजनांसाठी, आपण खालीलद्वारे अर्ज करू शकता:",
    "schemes.apply.method1": "संबंधित मंत्रालयांचे ऑनलाइन पोर्टल",
    "schemes.apply.method2": "आपल्या क्षेत्रातील सामाईक सेवा केंद्रे (CSC)",
    "schemes.apply.method3": "स्थानिक कृषी कार्यालये किंवा कृषी विज्ञान केंद्रे",
    "schemes.apply.method4": "बँक शाखा (क्रेडिट-संबंधित योजनांसाठी)",
    "schemes.apply.note": "टीप: अर्ज प्रक्रिया आणि आवश्यकता प्रत्येक योजनेसाठी भिन्न असू शकतात. कृपया सर्वात अचूक आणि अद्ययावत माहितीसाठी अधिकृत वेबसाइट्स पहा.",
    
    // Scheme card
    "scheme.ministry": "मंत्रालय:",
    "scheme.eligibility": "पात्रता:",
    "scheme.benefits": "लाभ:",
    "scheme.applicationProcess": "अर्ज प्रक्रिया:",
    "scheme.lastDate": "अंतिम तारीख:",
    "scheme.website": "अधिकृत वेबसाइट भेट द्या",
    
    // Fertilizers page
    "fertilizers.title": "खते आणि कीटकनाशके",
    "fertilizers.description": "उत्पादन सुधारण्यासाठी आणि आपल्या रोपांचे संरक्षण करण्यासाठी विविध पिकांसाठी शिफारस केलेली खते आणि कीटकनाशके शोधा.",
    "fertilizers.search": "पीक शोधा",
    "fertilizers.searchPlaceholder": "पीक नाव प्रविष्ट करा...",
    "fertilizers.noResults": "तुमच्या शोधाशी जुळणारी पिके आढळली नाहीत. वेगळे पीक नाव वापरून पहा.",
    "fertilizers.recommended": "शिफारस केलेले उत्पादने",
    "fertilizers.fertilizersSection": "खते",
    "fertilizers.pesticidesSection": "कीटकनाशके",
    "fertilizers.buyNow": "आता खरेदी करा",
    "fertilizers.type": "प्रकार:",
    "fertilizers.usage": "वापर:",
    "fertilizers.benefits": "फायदे:",
    "fertilizers.application": "अर्ज:",
    "fertilizers.price": "किंमत:",
    "fertilizers.loadMore": "अधिक लोड करा",

    //Disease Detection
    "disease.title": "रोग शोध",
    "disease.description": "आपल्या वनस्पतींचे रोग, कीड आणि पोषणतत्त्वांच्या कमतरतेची ओळख पटवण्यासाठी प्रतिमा अपलोड करा. तपशीलवार माहिती आणि उपचार शिफारसी मिळवा.",
    "disease.heading": "वनस्पती रोग शोध साधन",
    "disease.imageUpload": "वनस्पती प्रतिमा अपलोड करा",
    "disease.drag": "प्रतिमा निवडण्यासाठी क्लिक करा किंवा ड्रॅग करा",
    "disease.span": "समर्थित फॉरमॅट्स: JPG, PNG, WEBP",
    "disease.toggle": "भाषा",
    "disease.imagePreview": "प्रतिमेचा पूर्वावलोकन येथे दिसेल",
    "disease.result": "विश्लेषणाचे परिणाम",
    "disease.button": "वनस्पतीचे विश्लेषण करा",
    "disease.loading": "विश्लेषण सुरू आहे",

    //Crop Price Prediction
    "cropPrice.title": "भारतीय पीक किंमत भविष्यवाणी",
    "cropPrice.cropName": "पीकाचे नाव",
    "cropPrice.region": "प्रदेश",
    "cropPrice.placeholder": "पीकाचे नाव प्रविष्ट करा",
    "cropPrice.basePrice": "मूल्य प्रति क्विंटल (₹)",
    "cropPrice.button": "नवीन भविष्यवाणी मिळवा",
    "cropPrice.forecastHeading": "पुढील ३ महिन्यांसाठी पिकांच्या किंमतीतील प्रवृत्ती व अंदाज",
    "cropPrice.forecast": "अंदाज",
    "cropPrice.market": "बाजारातील अंतर्दृष्टी",
    "cropPrice.result": "आमच्या ARIMA मॉडेल विश्लेषणानुसार, निवडलेल्या प्रदेशातील पीक किंमती पुढील तीन महिन्यांत बदलण्याची शक्यता आहे. मॉडेलनुसार, या कालावधीच्या अखेरीस प्रति क्विंटल किंमत सध्याच्या किमतींच्या तुलनेत बदलली जाईल."



  }
};

const LanguageContext = createContext<LanguageContextType>({
  language: 'en',
  setLanguage: () => {},
  t: () => '',
});

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  const [language, setLanguage] = useState<'en' | 'mr'>('mr');

  const t = (key: string): string => {
    return translations[language][key as keyof typeof translations[typeof language]] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext);
